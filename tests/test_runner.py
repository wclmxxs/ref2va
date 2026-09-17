import json
import os
from pathlib import Path
import sys
import time

import pytest

from openvdn_comfy import runner
from openvdn_comfy.config import Settings, encode_command, inference_command


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setattr(runner, "WORKER_PYTHON", Path(sys.executable))
    monkeypatch.setattr(runner, "UPSTREAM", tmp_path)
    return tmp_path


def fake_success(calls):
    def run(command, log_path, **kwargs):
        calls.append(command)
        if "--out" in command:
            Path(command[command.index("--out") + 1]).write_bytes(b"encoded-tensors")
        else:
            cfg = json.loads(Path(command[-1]).read_text())
            assert cfg["render"]["num_steps"] == 8
            output = Path(cfg["render"]["out"])
            output.write_bytes(b"video")
            Path(str(output) + ".inference.json").write_text(json.dumps({"parallel": {"world_size": 8}}))
        return .1
    return run


def test_reference_cache_and_seed_reuse(runtime):
    reference = runtime / "image one.png"
    reference.write_bytes(b"first")
    calls = []
    def render(index, settings=Settings()):
        return runner.generate(prompt="literal `$(do not run)` 中文", refs=[reference], settings=settings,
                               output=runtime / f"{index}.mp4", process_runner=fake_success(calls))
    assert not render(1)["conditioning_cache_hit"]
    assert render(2, Settings(seed=99, fp8=False))["conditioning_cache_hit"]
    reference.write_bytes(b"second")
    assert not render(3)["conditioning_cache_hit"]
    assert len(calls) == 5  # 2 encodings, 3 independent eight-rank jobs
    assert calls[0][calls[0].index("--prompt") + 1] == "literal `$(do not run)` 中文"
    assert "--nproc_per_node=8" in calls[1]


def test_partial_encoding_never_becomes_cache(runtime):
    def fail(command, *args, **kwargs):
        Path(command[command.index("--out") + 1]).write_bytes(b"partial")
        raise RuntimeError("failed encoder")
    with pytest.raises(RuntimeError, match="failed encoder"):
        runner.generate(prompt="test", output=runtime / "x.mp4", process_runner=fail)
    assert not list((runtime / "runtime/conditioning").glob("*.pt"))
    records = list((runtime / "runtime/jobs").glob("*/result.json"))
    assert json.loads(records[0].read_text())["status"] == "failed"
    calls = []
    runner.generate(prompt="test", output=runtime / "y.mp4", process_runner=fake_success(calls))
    assert len(calls) == 2  # A failure must also release the GPU lock.


def test_preencoded_prompt_skips_conditioner(runtime):
    cache = runtime / "ready prompt.pt"
    cache.write_bytes(b"cached")
    calls = []
    result = runner.generate(prompt_file=cache, output=runtime / "x.mp4", process_runner=fake_success(calls))
    assert len(calls) == 1
    assert result["encode_seconds"] == 0
    with pytest.raises(ValueError):
        runner.generate(prompt_file=cache, prompt="ambiguous", process_runner=fake_success(calls))


def test_failed_inference_is_not_reported_as_success(runtime):
    cache = runtime / "cache.pt"
    cache.write_bytes(b"cached")
    with pytest.raises(RuntimeError, match="without producing a video"):
        runner.generate(prompt_file=cache, output=runtime / "missing.mp4", process_runner=lambda *a, **k: 0)


@pytest.mark.parametrize("kwargs", [{"num_frames": 120}, {"num_frames": 346}, {"warmup_steps": -1},
                                     {"softmax_ranks": 8}, {"reference_short_edge": 770},
                                     {"fp8": "false"}, {"softmax_backend": "sol"}])
def test_invalid_request_rejected(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs).validate()


def test_official_config_and_no_shell_interpolation(tmp_path):
    config = Settings().inference_config(tmp_path / "ref [1], x.pt", tmp_path / "out a.mp4")
    assert config["render"]["num_steps"] == 8
    assert config["render"]["num_frames"] == 345
    assert config["parallel"]["softmax_ranks"] == 6
    assert config["render"]["prompt_file"].endswith("ref [1], x.pt")
    with pytest.raises(ValueError, match="OmegaConf"):
        Settings().inference_config(tmp_path / "${x}.pt", tmp_path / "out.mp4")
    assert "--nproc_per_node=8" in inference_command(tmp_path / "a b.json")
    command = encode_command("a ' $() prompt", ["a b.png"], "out.pt", 768)
    assert command[-2:] == ["--refs", "a b.png"]


def test_cancel_kills_process_group(runtime):
    if os.name != "posix":
        pytest.skip("POSIX process groups")
    pid_file = runtime / "child.pid"
    command = [sys.executable, "-c", "import subprocess,time,pathlib,sys; "
               "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
               "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)", str(pid_file)]
    class Cancelled(Exception):
        pass
    def interrupt():
        if pid_file.exists():
            raise Cancelled()
    with pytest.raises(Cancelled):
        runner.run_process(command, runtime / "log", interrupt=interrupt, timeout=10)
    import psutil
    child_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
            break
        time.sleep(.1)
    else:
        pytest.fail("Cancelled worker left a live child")


def test_nonzero_subprocess_reports_log(runtime):
    with pytest.raises(RuntimeError, match="code 7.*failure.log"):
        runner.run_process([sys.executable, "-c", "raise SystemExit(7)"], runtime / "failure.log")
