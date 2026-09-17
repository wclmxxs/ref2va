"""Own the resident GPU group and CPU UI; expose the UI only after warmup."""
from dataclasses import asdict
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
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
from openvdn_comfy.runner import log_tail, stop_group, worker_environment


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
                    and "--user-directory" in args and str(ROOT / ".runtime/comfy-user") in args
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
               "--nproc_per_node=8", str(ROOT / "scripts/resident_worker.py")]
    with (BACKEND / "worker.log").open("a") as log:
        log.write(f"\n=== Starting resident worker {instance} ===\n")
        log.flush()
        process = subprocess.Popen(command, cwd=UPSTREAM, env=worker_environment(), stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    atomic_json(BACKEND / "owner.json", process_record(process, instance))
    return process, instance


def await_ready(process, timeout=3600):
    started, phase = time.monotonic(), None
    while process.poll() is None:
        current = health()
        if current["ready"]:
            print("OpenVDN ready: all models resident; 8-step video/audio warmup complete.", flush=True)
            return
        if current.get("phase") != phase:
            phase = current.get("phase")
            print(f"OpenVDN startup: {phase}; log: {BACKEND / 'worker.log'}", flush=True)
        if time.monotonic() - started > timeout:
            raise TimeoutError("OpenVDN startup exceeded one hour\n" + log_tail(BACKEND / "worker.log"))
        time.sleep(.5)
    raise RuntimeError(f"OpenVDN preload failed (code {process.returncode})\n" + failure_detail(health().get("instance")))


def main():
    startup_settings()  # Validate before stopping an already running deployment.
    parallel_vae_enabled()
    cache_settings()
    enabled("REF2VA_EXACT_RUNTIME")
    enabled("REF2VA_ASYNC_OUTPUT")
    BACKEND.mkdir(parents=True, exist_ok=True)
    retire_previous_server()
    lock = (BACKEND / "serve.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic_json(BACKEND / "server.json", process_record(psutil.Process(), uuid.uuid4().hex))
    worker = ui = None
    instance = None

    def terminate(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        clear_gpu_applications()
        subprocess.run([str(WORKER_PYTHON), str(ROOT / "scripts/doctor.py"), "--nccl"], check=True, cwd=ROOT)
        worker, instance = launch_worker()
        await_ready(worker)
        command = [str(ROOT / ".venv-ui/bin/python"), str(ROOT / ".deps/ComfyUI/main.py"), "--cpu", "--disable-dynamic-vram",
                   "--listen", os.environ.get("REF2VA_LISTEN", "0.0.0.0"), "--port", os.environ.get("REF2VA_PORT", "8188"),
                   "--output-directory", str(ROOT / "output"), "--input-directory", str(ROOT / "input"),
                   "--user-directory", str(RUNTIME / "comfy-user"),
                   "--database-url", f"sqlite:///{RUNTIME / 'comfy-user/comfyui.db'}", *sys.argv[1:]]
        ui = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        failed = False
        while ui.poll() is None:
            cancel = read_json(BACKEND / "cancel.json", {})
            if cancel.get("instance") == instance:
                print("Request cancelled; releasing all ranks and reloading the resident backend.", flush=True)
                atomic_json(BACKEND / "state.json", {"instance": instance, "status": "loading", "phase": "restarting_after_cancel"})
                stop_group(worker)
                ensure_free_gpus()  # Never kill unrelated applications during request handling.
                worker, instance = launch_worker()
                await_ready(worker)
                failed = False
            elif worker.poll() is not None and not failed:
                stop_group(worker)  # Reap surviving ranks even if torchrun already exited.
                atomic_json(BACKEND / "state.json", {"instance": instance, "status": "failed", "phase": "failed",
                            "error": failure_detail(instance)})
                print("OpenVDN failed. UI remains available for diagnostics; restart with bash deploy.sh start.", flush=True)
                failed = True
            time.sleep(.3)
        if ui.returncode:
            raise RuntimeError(f"ComfyUI exited with code {ui.returncode}")
    finally:
        # Ignore repeated shutdown signals while reaping children.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
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
