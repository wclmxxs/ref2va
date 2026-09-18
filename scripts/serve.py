"""Own the resident GPU group and CPU UI; expose the UI only after warmup."""
from dataclasses import asdict
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psutil
from openvdn_comfy.backend import BACKEND, health, read_json, same_process, startup_settings, failure_detail, parallel_vae_enabled
from openvdn_comfy.config import RUNTIME, UPSTREAM, WORKER_PYTHON, atomic_json
from openvdn_comfy.compile_cache import cache_settings
from openvdn_comfy.exact_runtime import enabled
from openvdn_comfy.gpu_cleanup import clear_gpu_applications, stop_tree
from openvdn_comfy.gpu_check import ensure_free_gpus
from openvdn_comfy.runner import stop_group, worker_environment
from openvdn_comfy.hardware import Hardware, install_workflows
from openvdn_comfy.supervision import Policy, WorkerWatchdog, fail_pending


def process_record(process, instance):
    return {"pid": process.pid, "created": psutil.Process(process.pid).create_time(), "instance": instance}


def retire_previous_server():
    # Only stop a controller from this exact checkout, with verified PID identity.
    previous = read_json(BACKEND / "server.json", {})
    if same_process(previous):
        process = psutil.Process(previous["pid"])
        if str(ROOT / "scripts/serve.py") not in process.cmdline():
            raise RuntimeError("Existing controller identity does not match this checkout")
        process.terminate()
        try:
            process.wait(timeout=30)
        except psutil.TimeoutExpired:
            stop_tree(process)
    # Migration from the earlier foreground ComfyUI launcher (no controller PID).
    for process in psutil.process_iter(["pid", "cmdline", "cwd"]):
        try:
            args = process.info["cmdline"] or []
            if (process.pid != os.getpid() and any(arg.endswith("ComfyUI/main.py") for arg in args)
                    and "--user-directory" in args and str(RUNTIME / "comfy-user") in args
                    and process.info["cwd"] == str(ROOT)):
                print(f"Stopping previous ComfyUI from this checkout: {process.pid}", flush=True)
                stop_tree(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def launch_worker():
    settings = startup_settings()
    instance = uuid.uuid4().hex
    atomic_json(BACKEND / "launch.json", {"instance": instance, "settings": asdict(settings),
                                        "parallel_vae": parallel_vae_enabled(), "compile_cache": cache_settings()})
    atomic_json(BACKEND / "inference.json", settings.inference_config(BACKEND / "warmup.pt", BACKEND / "warmup.mp4"))
    atomic_json(BACKEND / "state.json", {"instance": instance, "status": "loading", "phase": "starting_ranks"})
    command = [str(WORKER_PYTHON), "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc_per_node={Hardware.from_env().world_size}", str(ROOT / "scripts/resident_worker.py")]
    with (BACKEND / "worker.log").open("a") as log:
        log.write(f"\n=== Starting resident worker {instance} ===\n")
        log.flush()
        process = subprocess.Popen(command, cwd=UPSTREAM, env=worker_environment(), stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    atomic_json(BACKEND / "owner.json", process_record(process, instance))
    return process, instance


def main():
    policy = Policy.from_env()
    startup_settings()  # Validate before stopping an already running deployment.
    parallel_vae_enabled()
    cache_settings()
    enabled("REF2VA_EXACT_RUNTIME")
    enabled("REF2VA_ASYNC_OUTPUT")
    enabled("REF2VA_PIPELINE_OUTPUT")
    BACKEND.mkdir(parents=True, exist_ok=True)
    retire_previous_server()
    lock = (BACKEND / "serve.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    controller = process_record(psutil.Process(), uuid.uuid4().hex)
    atomic_json(BACKEND / "server.json", controller)
    worker = ui = None
    instance = None

    def terminate(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    restart_count, consecutive = 0, 0
    next_attempt = 0.
    watchdog = None
    last_error = None
    status = None

    def report(value, **extra):
        nonlocal status
        status = value
        atomic_json(BACKEND / "supervisor.json", {
            "controller": controller, "instance": instance, "status": value,
            "restart_count": restart_count, "consecutive_failures": consecutive,
            "last_error": last_error, "policy": asdict(policy), "updated_at": time.time(), **extra})

    def recover(reason, cancelled=False):
        nonlocal worker, consecutive, restart_count, next_attempt, last_error
        restart_count += 1
        consecutive = 0 if cancelled else consecutive + 1
        last_error = reason
        report("recovering")  # Reject new requests before stopping any CUDA ranks.
        if worker is not None:
            stop_group(worker)
            # Capture diagnostics after all ranks exit. Preserve the original
            # result if rank zero has already reported the failing request.
            detail = reason + "\n" + failure_detail(instance)
            fail_pending(BACKEND, instance, detail)
            atomic_json(BACKEND / "failures" / f"{instance}.json",
                        {"instance": instance, "error": detail, "time": time.time(), "cancelled": cancelled})
            worker = None
        delay = 1. if cancelled else policy.delay(consecutive)
        next_attempt = time.monotonic() + delay
        report("recovering", retry_at=time.time() + delay)
        print(f"OpenVDN recovery #{restart_count}: {reason[-2000:]}\nRetrying worker in {delay:g}s; UI remains available.", flush=True)

    try:
        report("loading")
        clear_gpu_applications()
        subprocess.run([str(WORKER_PYTHON), str(ROOT / "scripts/doctor.py"), "--nccl"], check=True, cwd=ROOT)
        phase = None
        while True:
            if ui is not None and ui.poll() is not None:
                raise RuntimeError(f"ComfyUI exited with code {ui.returncode}")
            if worker is None:
                if time.monotonic() < next_attempt:
                    time.sleep(.3)
                    continue
                try:
                    # Reclaim only our old group during recovery; do not kill
                    # other services which acquired a GPU after initial startup.
                    ensure_free_gpus()
                    worker, instance = launch_worker()
                    watchdog = WorkerWatchdog(BACKEND, instance, policy)
                    phase = None
                    report("loading")
                except Exception:
                    recover("Worker launch failed: " + traceback.format_exc())
                    continue
            current = health()
            cancel = read_json(BACKEND / "cancel.json", {})
            if cancel.get("instance") == instance:
                timed_out = cancel.get("reason") == "timeout"
                recover("GPU request timed out" if timed_out else "Request cancelled", cancelled=not timed_out)
                continue
            reason = watchdog.check(worker, current)
            if reason:
                recover(reason)
                continue
            if current.get("phase") != phase:
                phase = current.get("phase")
                if not current["ready"]:
                    print(f"OpenVDN startup: {phase}; log: {BACKEND / 'worker.log'}", flush=True)
            if current["ready"]:
                if status != "ready":
                    report("ready")
                    print("OpenVDN ready: all models resident; startup checks complete.", flush=True)
                if consecutive and time.monotonic() - watchdog.ready_since >= policy.stable_seconds:
                    consecutive = 0
                    report("ready")
                if ui is None:
                    install_workflows(ROOT, RUNTIME / 'comfy-user', startup_settings())
                    command = [str(ROOT / ".venv-ui/bin/python"), str(ROOT / ".deps/ComfyUI/main.py"), "--cpu", "--disable-dynamic-vram",
                               "--listen", os.environ.get("REF2VA_LISTEN", "0.0.0.0"), "--port", os.environ.get("REF2VA_PORT", "8188"),
                               "--output-directory", str(ROOT / "output"), "--input-directory", str(ROOT / "input"),
                               "--user-directory", str(RUNTIME / "comfy-user"),
                               "--temp-directory", str(RUNTIME / "comfy-temp"),
                               "--database-url", f"sqlite:///{RUNTIME / 'comfy-user/comfyui.db'}", *sys.argv[1:]]
                    ui = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
                    atomic_json(BACKEND / "ui.json", process_record(ui, instance))
            time.sleep(.3)
    finally:
        # Ignore repeated shutdown signals while reaping children.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        report("stopped")
        if ui is not None:
            stop_group(ui)
        if worker is not None:
            stop_group(worker)
        atomic_json(BACKEND / "state.json", {"instance": instance, "status": "stopped", "phase": "stopped"})
        lock.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
