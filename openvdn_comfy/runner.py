from __future__ import annotations

import fcntl
import atexit
import hashlib
import json
import os
import signal
import subprocess
import time
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from .config import (ROOT, RUNTIME, UPSTREAM, WORKER_PYTHON, Settings, atomic_json,
                     source_lock)
from .backend import call_worker, wait_output
from .hardware import Hardware
from .exact_runtime import enabled

_active = set()
_active_lock = threading.Lock()


@atexit.register
def shutdown():
    with _active_lock:
        processes = list(_active)
    for process in processes:
        if process.poll() is None:
            stop_group(process)


def worker_environment():
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "RANK", "LOCAL_RANK", "WORLD_SIZE",
                "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK"):
        env.pop(key, None)
    # ComfyUI uses --cpu. These child processes alone own CUDA.
    env.update(PYTHONPATH=str(UPSTREAM), PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", TORCH_NCCL_ASYNC_ERROR_HANDLING="1")
    hardware = Hardware.from_env()
    env.setdefault("CUDA_VISIBLE_DEVICES", ",".join(hardware.visible_devices()))
    # This H200 deployment reports NVLS multicast bind CUDA error 401 with
    # NCCL 2.29. Use the same workaround for the probe and resident ranks.
    # This only disables NVLink SHARP offload, not P2P/NVLink transport.
    # An explicit setting can re-enable it after the host fabric is repaired.
    if hardware.gpu_type == "h200":
        env.setdefault("NCCL_NVLS_ENABLE", "0")
    env.setdefault("OMP_NUM_THREADS", "8")
    cache_root = RUNTIME if hardware.gpu_type == "h200" and hardware.world_size == 8 else RUNTIME / "compile" / f"{hardware.gpu_type}-{hardware.world_size}"
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_root / "inductor"))
    env.setdefault("TRITON_CACHE_DIR", str(cache_root / "triton"))
    env.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
    return env


def stop_group(process):
    # torchrun and all ranks share this group; a surviving rank would retain GPUs.
    if getattr(process, "_openvdn_group_stopped", False):
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
    process._openvdn_group_stopped = True


def log_tail(path, max_bytes=32768, max_lines=160):
    """Include worker diagnostics in the UI/API without loading an unbounded log."""
    try:
        with Path(path).open("rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            stream.seek(max(0, size - max_bytes))
            data = stream.read(max_bytes)
        if size > max_bytes:
            data = data.partition(b"\n")[2] or data
        text = "\n".join(data.decode("utf-8", errors="replace").splitlines()[-max_lines:]).strip()
        return text or "(worker log is empty)"
    except OSError as error:
        return f"(could not read worker log: {error})"


def run_process(command, log_path, interrupt=lambda: None, timeout=3600):
    started = time.monotonic()
    with Path(log_path).open("ab") as log:
        process = subprocess.Popen(command, cwd=UPSTREAM, env=worker_environment(),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        with _active_lock:
            _active.add(process)
        try:
            while process.poll() is None:
                interrupt()
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"GPU stage exceeded {timeout}s; see {log_path}")
                time.sleep(0.2)
            if process.returncode:
                raise RuntimeError(f"OpenVDN exited with code {process.returncode}; see {log_path}\n"
                                   f"--- worker log tail ---\n{log_tail(log_path)}")
        except BaseException:
            stop_group(process)
            raise
        finally:
            with _active_lock:
                _active.discard(process)
    return time.monotonic() - started


@contextmanager
def gpu_lock(interrupt):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    # Shared by CLI and ComfyUI in this checkout. Keep the inode for the lifetime
    # of the deployment so competing requests cannot acquire different locks.
    with (RUNTIME / "gpu.lock").open("a") as lock:
        while True:
            interrupt()
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.2)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def conditioning_key(prompt, refs, short_edge, anchors=None, canvas=None):
    from .keyframes import normalize_anchors
    anchors = normalize_anchors(len(refs), anchors)
    identity = {"prompt": prompt, "reference_short_edge": short_edge, "sources": source_lock(), "images": []}
    if anchors and anchors[0] != 'ref':
        if canvas is None:
            raise ValueError('Keyframe conditioning cache needs the generation canvas')
        identity.pop('reference_short_edge')
        identity.update(image_anchors=anchors, keyframe_canvas=canvas, keyframe_preprocess_version=1)
    for path in refs:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        identity["images"].append(digest.hexdigest())
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def generate(*, prompt="", refs=(), image_anchors=None, settings=None, output=None, prompt_file=None,
             interrupt=lambda: None, worker_call=call_worker, progress=lambda phase: None, defer_output=False):
    settings = (settings or Settings()).validate()
    refs = [Path(path).resolve() for path in refs]
    from .keyframes import normalize_anchors, conditioning_mode
    anchors = normalize_anchors(len(refs), image_anchors)
    if prompt_file is not None and (refs or prompt or image_anchors):
        raise ValueError("Choose a pre-encoded prompt_file OR prompt + references")
    if prompt_file is None and not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    if len(refs) > 9:
        raise ValueError("This wrapper accepts up to 9 reference images")
    for path in refs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not WORKER_PYTHON.is_file():
        raise RuntimeError("Run ./deploy.sh install on the GPU server first")
    job_id = uuid.uuid4().hex
    job = RUNTIME / "jobs" / job_id
    job.mkdir(parents=True)
    output = Path(output or ROOT / "output" / f"vdn_{job_id}.mp4").resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    request = {"job_id": job_id, "settings": asdict(settings), "sources": source_lock(),
               "render_plan": settings.render_plan().metadata(),
               "mode": "cached_prompt" if prompt_file else conditioning_mode(anchors), "image_anchors": list(anchors),
               "references": list(map(str, refs)), "prompt": prompt, "output": str(output),
               "defer_output": enabled('REF2VA_PIPELINE_OUTPUT')}
    atomic_json(job / "request.json", request)
    started = time.monotonic()
    try:
        progress("waiting_for_gpus")
        with gpu_lock(interrupt):
            queue_seconds = time.monotonic() - started
            if prompt_file:
                cache = Path(prompt_file).resolve()
                if not cache.is_file():
                    raise FileNotFoundError(cache)
            else:
                plan = settings.render_plan()
                key = conditioning_key(prompt, refs, settings.reference_short_edge, anchors,
                                       (plan.generation_width, plan.generation_height))
                cache = RUNTIME / "conditioning" / f"{key}.pt"
            atomic_json(job / "inference.json", settings.inference_config(cache, output))
            progress("inference")
            metrics = worker_call({**request, "prompt_file": str(cache)}, interrupt=interrupt, progress=progress)
        def finalize(metrics):
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError("Inference succeeded without producing a video")
            upstream_record = json.loads(Path(str(output) + ".inference.json").read_text())
            if upstream_record["parallel"]["world_size"] != Hardware.from_env().world_size:
                raise RuntimeError("Upstream result did not use the configured GPU count")
            result = {**request, "status": "complete", "metrics_schema_version": 10, "log_directory": str(job),
                      "conditioning_cache_hit": metrics["conditioning_cache_hit"], "prompt_file": str(cache),
                      "queue_seconds": queue_seconds, "encode_seconds": metrics["encode_seconds"],
                      "inference_process_seconds": metrics["inference_process_seconds"], "resident": True,
                      "request_wall_seconds": time.monotonic() - started,
                      "upstream": upstream_record, "timings": {**upstream_record["timings"],
                          "gpu_queue_seconds": queue_seconds, "generation_wall_seconds": time.monotonic() - started}}
            atomic_json(str(output) + ".metrics.json", result)
            atomic_json(job / "result.json", result)
            return result
        def failed(error):
            atomic_json(job / "result.json", {**request, "status": "failed", "error": str(error),
                                             "request_wall_seconds": time.monotonic() - started})
        pending = PendingGeneration(metrics, finalize, failed, interrupt if not defer_output else lambda: None)
        return pending if defer_output else pending.finish()
    except BaseException as error:
        atomic_json(job / "result.json", {**request, "status": "failed", "error": str(error),
                                         "request_wall_seconds": time.monotonic() - started})
        raise


class PendingGeneration:
    def __init__(self, metrics, finalize, failed, interrupt=lambda: None):
        self.metrics, self.finalize, self.failed, self.interrupt = metrics, finalize, failed, interrupt

    def finish(self):
        try:
            metrics = wait_output(self.metrics['_output_ticket'], self.interrupt) if isinstance(self.metrics, dict) and '_output_ticket' in self.metrics else self.metrics
            return self.finalize(metrics)
        except BaseException as error:
            self.failed(error)
            raise
