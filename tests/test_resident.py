from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
import types

import psutil
import pytest
import torch

from openvdn_comfy import backend, gpu_cleanup
from openvdn_comfy.config import Settings, atomic_json
from openvdn_comfy.resident_geometry import GeometryCache


def test_actual_upstream_ulysses_changes_geometry_without_reloading(monkeypatch):
    # Exercise the actual pinned runtime's pure configuration code on CPU. Triton
    # kernel bodies/collectives are not executed in this test.
    path = Path(__file__).resolve().parents[1] / "work/upstream/openvdn/src/inference/utils/ulysses_runtime.py"
    if not path.exists():
        path = Path(__file__).resolve().parents[1] / ".deps/openvdn/src/inference/utils/ulysses_runtime.py"
    if not path.exists():
        pytest.skip("Run install_sources.py to fetch the pinned upstream source")
    triton = types.ModuleType("triton")
    triton.jit = lambda f: f
    language = types.ModuleType("triton.language")
    triton.language = language
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    spec = importlib.util.spec_from_file_location("tested_ulysses", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    resets = []
    cache = GeometryCache(lambda: resets.append(True), max_shapes=2)
    runtime = module.UlyssesRuntime(0, 8, 0, torch.device("cpu"), "gloo")
    runtime.enable_branch_parallel(6)
    plan = Settings(duration=10, ratio="9:16", resolution=720).render_plan()
    for length in (200, 201, 201, 202):
        cache.prepare(runtime, plan, torch.zeros(length, 3), torch.ones(length, dtype=torch.long), None)
        runtime.configure(length, 56)
        assert sum(runtime.splits) == length
        assert runtime.softmax_ranks == 6
        assert sum(runtime.softmax_head_splits) == sum(runtime.linear_head_splits) == 56
    assert resets == [True]


@pytest.fixture
def resident(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "BACKEND", tmp_path)
    profile = {key: getattr(Settings(), key) for key in backend.PROFILE_FIELDS}
    atomic_json(tmp_path / "owner.json", {"pid": psutil.Process().pid, "created": psutil.Process().create_time(), "instance": "test"})
    atomic_json(tmp_path / "state.json", {"instance": "test", "status": "ready", "profile": profile})
    return tmp_path


def test_mailbox_reuses_one_worker_and_propagates_errors(resident):
    def worker():
        handled = set()
        while len(handled) < 3:
            command = backend.read_json(resident / "command.json", {})
            token = command.get("token")
            if token and token not in handled:
                handled.add(token)
                result = {"ok": True, "metrics": {"call": len(handled)}} if len(handled) < 3 else {"ok": False, "error": "CUDA OOM detail"}
                atomic_json(resident / "results" / f"{token}.json", result)
            time.sleep(.01)
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    request = {"settings": asdict(Settings()), "prompt": "test"}
    assert backend.call_worker(request)["call"] == 1
    assert backend.call_worker(request)["call"] == 2
    with pytest.raises(RuntimeError, match="CUDA OOM detail"):
        backend.call_worker(request)
    thread.join(timeout=2)
    assert not (resident / "cancel.json").exists()
    assert backend.health()["ready"]


def test_cancel_requests_group_restart_and_stale_owner_is_not_ready(resident):
    def interrupt():
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        backend.call_worker({"settings": asdict(Settings())}, interrupt=interrupt)
    assert backend.read_json(resident / "cancel.json")["instance"] == "test"
    owner = backend.read_json(resident / "owner.json")
    owner["created"] -= 1
    atomic_json(resident / "owner.json", owner)
    assert not backend.health()["ready"]


def test_worker_crash_is_not_mistaken_for_user_cancellation(resident, monkeypatch):
    original = backend.health
    checks = 0
    def health():
        nonlocal checks
        checks += 1
        return {**original(), "ready": checks == 1}
    monkeypatch.setattr(backend, "health", health)
    atomic_json(resident / "errors/test-7.json", {"traceback": "torch.OutOfMemoryError: GPU 7 exhausted"})
    with pytest.raises(RuntimeError, match="GPU 7 exhausted"):
        backend.call_worker({"settings": asdict(Settings())})
    assert not (resident / "cancel.json").exists()


@pytest.mark.parametrize("preload_fails", [True, False])
def test_supervisor_gates_ui_on_preload_and_reaps_worker(tmp_path, monkeypatch, preload_fails):
    path = Path(__file__).resolve().parents[1] / "scripts/serve.py"
    spec = importlib.util.spec_from_file_location("serve_under_test", path)
    serve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve)
    monkeypatch.setattr(serve, "BACKEND", tmp_path)
    monkeypatch.setattr(serve, "retire_previous_server", lambda: None)
    monkeypatch.setattr(serve, "clear_gpu_applications", lambda: None)
    monkeypatch.setattr(serve.signal, "signal", lambda *a: None)
    monkeypatch.setattr(serve.subprocess, "run", lambda *a, **k: None)
    worker = types.SimpleNamespace(name="worker")
    ui = types.SimpleNamespace(name="ui", poll=lambda: 0, returncode=0)
    monkeypatch.setattr(serve, "launch_worker", lambda: (worker, "test"))
    ready = []
    def preload(process):
        assert process is worker
        if preload_fails:
            raise RuntimeError("preload failed")
        ready.append(True)
    monkeypatch.setattr(serve, "await_ready", preload)
    started = []
    def launch_ui(command, **kwargs):
        assert ready == [True]
        assert "--cpu" in command and "--database-url" in command
        started.append(True)
        return ui
    monkeypatch.setattr(serve.subprocess, "Popen", launch_ui)
    stopped = []
    monkeypatch.setattr(serve, "stop_group", lambda process: stopped.append(process.name))
    if preload_fails:
        with pytest.raises(RuntimeError, match="preload failed"):
            serve.main()
        assert started == [] and stopped == ["worker"]
    else:
        serve.main()
        assert started == [True] and stopped == ["ui", "worker"]


@pytest.mark.parametrize("cgroup,expected", [
    ("0::/system.slice/sglang.service", "sglang.service"),
    ("0::/system.slice/ssh.service", None),
    ("0::/system.slice/docker.service", None),
    ("0::/user.slice/user-0.slice/session-8.scope", None),
    ("0::/system.slice/systemd-logind.service", None),
    ("0::/system.slice/supervisor.service", None),
    ("0::/system.slice/kubelet.service", None),
    ("0::/system.slice/amazon-ssm-agent.service", None),
])
def test_cleanup_identifies_application_service_only(cgroup, expected):
    assert gpu_cleanup.service_owner(cgroup) == expected


def test_cleanup_stops_supervisor_and_requeries_gpu_processes(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_cleanup, "RUNTIME", tmp_path)
    monkeypatch.setenv("REF2VA_CLEAR_GPU_APPS", "1")
    monkeypatch.setattr(gpu_cleanup, "selected_gpus", lambda: [(str(i), f"GPU-{i}", "140000") for i in range(8)])
    apps = [{998877: ["GPU-0", "998877", "sgl_diffusion::scheduler_U0", "60000"]}, {}]
    monkeypatch.setattr(gpu_cleanup, "compute_apps", lambda uuids: apps.pop(0))
    monkeypatch.setattr(gpu_cleanup.psutil, "Process", lambda pid: types.SimpleNamespace(pid=pid))
    original_read = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **k:
                        "0::/system.slice/sglang.service" if str(path) == "/proc/998877/cgroup" else
                        "0::/user.slice/session-1.scope" if str(path) == "/proc/self/cgroup" else original_read(path, *a, **k))
    commands = []
    monkeypatch.setattr(gpu_cleanup.subprocess, "run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(gpu_cleanup.time, "sleep", lambda seconds: None)
    checked = []
    monkeypatch.setattr(gpu_cleanup, "ensure_free_gpus", lambda: checked.append(True))
    gpu_cleanup.clear_gpu_applications()
    assert commands == [["systemctl", "stop", "sglang.service"]]
    assert checked == [True]
    assert json.loads((tmp_path / "gpu-cleanup.json").read_text())["actions"][0]["stopped"]
