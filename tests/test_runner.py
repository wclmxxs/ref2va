import json
import os
from pathlib import Path
import sys
import time

import pytest

from openvdn_comfy import runner
from openvdn_comfy.config import Settings


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setattr(runner, "WORKER_PYTHON", Path(sys.executable))
    monkeypatch.setattr(runner, "UPSTREAM", tmp_path)
    return tmp_path


def fake_success(calls):
    def run(request, **kwargs):
        calls.append(request)
        cache = Path(request["prompt_file"])
        hit = cache.exists()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(b"encoded-tensors")
        output = Path(request["output"])
        output.write_bytes(b"video")
        Path(str(output) + ".inference.json").write_text(json.dumps({"parallel": {"world_size": 8}}))
        return {"conditioning_cache_hit": hit, "encode_seconds": 0 if hit else .1, "inference_process_seconds": .2}
    return run


def test_reference_cache_and_seed_reuse(runtime):
    reference = runtime / "image one.png"
    reference.write_bytes(b"first")
    calls = []
    def render(index, settings=Settings()):
        return runner.generate(prompt="literal `$(do not run)` 中文", refs=[reference], settings=settings,
                               output=runtime / f"{index}.mp4", worker_call=fake_success(calls))
    assert not render(1)["conditioning_cache_hit"]
    assert render(2, Settings(seed=99, fp8=False))["conditioning_cache_hit"]
    reference.write_bytes(b"second")
    assert not render(3)["conditioning_cache_hit"]
    assert len(calls) == 3  # Three requests, no per-job process launch.
    assert calls[0]["prompt"] == "literal `$(do not run)` 中文"
    assert calls[0]["prompt_file"] == calls[1]["prompt_file"]


def test_partial_encoding_never_becomes_cache(runtime):
    def fail(request, **kwargs):
        raise RuntimeError("failed encoder")
    with pytest.raises(RuntimeError, match="failed encoder"):
        runner.generate(prompt="test", output=runtime / "x.mp4", worker_call=fail)
    assert not list((runtime / "runtime/conditioning").glob("*.pt"))
    records = list((runtime / "runtime/jobs").glob("*/result.json"))
    assert json.loads(records[0].read_text())["status"] == "failed"
    calls = []
    runner.generate(prompt="test", output=runtime / "y.mp4", worker_call=fake_success(calls))
    assert len(calls) == 1  # A failure must also release the GPU lock.


def test_preencoded_prompt_skips_conditioner(runtime):
    cache = runtime / "ready prompt.pt"
    cache.write_bytes(b"cached")
    calls = []
    result = runner.generate(prompt_file=cache, output=runtime / "x.mp4", worker_call=fake_success(calls))
    assert len(calls) == 1
    assert result["encode_seconds"] == 0
    with pytest.raises(ValueError):
        runner.generate(prompt_file=cache, prompt="ambiguous", worker_call=fake_success(calls))


def test_failed_inference_is_not_reported_as_success(runtime):
    cache = runtime / "cache.pt"
    cache.write_bytes(b"cached")
    with pytest.raises(RuntimeError, match="without producing a video"):
        runner.generate(prompt_file=cache, output=runtime / "missing.mp4", worker_call=lambda *a, **k: 0)


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
    with pytest.raises(RuntimeError, match="code 7.*failure.log") as exc:
        runner.run_process([sys.executable, "-c", "import sys; print('worker diagnostic: 测试错误', file=sys.stderr); raise SystemExit(7)"],
                           runtime / "failure.log")
    assert "worker diagnostic: 测试错误" in str(exc.value)


def test_log_tail_is_bounded_and_handles_missing_logs(runtime):
    log = runtime / "large.log"
    log.write_text("old-output\n" * 200000 + "last diagnostic\n")
    text = runner.log_tail(log)
    assert text.endswith("last diagnostic")
    assert len(text.splitlines()) == 160
    assert len(text) <= 32768
    assert "could not read worker log" in runner.log_tail(runtime / "missing.log")
    log.write_text("")
    assert runner.log_tail(log) == "(worker log is empty)"
