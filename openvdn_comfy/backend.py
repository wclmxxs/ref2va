"""Private local mailbox between CPU ComfyUI and the resident torchrun group."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import uuid

import psutil

from .config import RUNTIME, Settings, atomic_json

BACKEND = RUNTIME / "backend"
PROFILE_FIELDS = ("fp8", "inference_kernels", "softmax_backend")
REQUEST_DEFAULT_FIELDS = ("softmax_ranks", "profile")


def parallel_vae_enabled():
    value = os.environ.get("REF2VA_VAE_PARALLEL", "1")
    if value not in ("0", "1"):
        raise ValueError("REF2VA_VAE_PARALLEL must be 0 or 1")
    return value == "1"


def startup_settings():
    def boolean(name, default):
        value = os.environ.get(name, str(int(default)))
        if value not in ("0", "1"):
            raise ValueError(f"{name} must be 0 or 1")
        return value == "1"
    return Settings(
        duration=float(os.environ.get("REF2VA_WARMUP_DURATION", "10")),
        ratio=os.environ.get("REF2VA_WARMUP_RATIO", "9:16"),
        resolution=int(os.environ.get("REF2VA_WARMUP_RESOLUTION", "768")),
        reference_short_edge=int(os.environ.get("REF2VA_REFERENCE_SHORT_EDGE", "768")),
        fp8=boolean("REF2VA_FP8", True), inference_kernels=boolean("REF2VA_INFERENCE_KERNELS", True),
        softmax_backend=os.environ.get("REF2VA_SOFTMAX_BACKEND", "flex"),
        softmax_ranks=int(os.environ.get("REF2VA_SOFTMAX_RANKS", "6")),
        profile=boolean("REF2VA_PROFILE", False), warmup_steps=8).validate()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def same_process(record):
    try:
        process = psutil.Process(record["pid"])
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE and abs(process.create_time() - record["created"]) < .01
    except (psutil.NoSuchProcess, KeyError):
        return False


def health():
    state = read_json(BACKEND / "state.json", {})
    owner = read_json(BACKEND / "owner.json", {})
    ready = (same_process(owner) and state.get("instance") == owner.get("instance")
             and state.get("status") in ("ready", "busy"))
    return {**state, "ready": bool(ready)}


def failure_detail(instance):
    from .runner import log_tail
    errors = []
    for rank in range(8):
        error = read_json(BACKEND / "errors" / f"{instance}-{rank}.json")
        if error:
            errors.append(f"rank {rank}: {error['traceback'][-8000:]}")
    return "\n".join(errors)[:48000] + "\n--- worker log tail ---\n" + log_tail(BACKEND / "worker.log")


def validate_profile(settings, state=None):
    state = state if state is not None else health()
    if not state.get("ready"):
        raise RuntimeError("OpenVDN backend is not ready; start/restart with bash deploy.sh start. " + state.get("error", ""))
    mismatches = [name for name in PROFILE_FIELDS if getattr(settings, name) != state["profile"][name]]
    if mismatches:
        raise ValueError("Resident model profile differs for " + ", ".join(mismatches) +
                         "; use the active profile from /openvdn/health or change REF2VA_* and restart.")


def call_worker(request, interrupt=lambda: None, progress=lambda phase: None, timeout=3600):
    # Caller holds gpu.lock, which serializes UI, REST and CLI requests in this checkout.
    state = health()
    validate_profile(Settings(**request["settings"]), state)
    token = uuid.uuid4().hex
    command = {**request, "token": token, "instance": state["instance"]}
    result_path = BACKEND / "results" / f"{token}.json"
    started = time.monotonic()
    atomic_json(BACKEND / "command.json", command)
    previous_phase = None
    cancel_needed = False
    try:
        while True:
            try:
                interrupt()
            except BaseException:
                cancel_needed = True
                raise
            result = read_json(result_path)
            if result is not None:
                if not result.get("ok"):
                    raise RuntimeError(result["error"])
                return result["metrics"]
            current = health()
            if not current["ready"] or current.get("instance") != state["instance"]:
                raise RuntimeError("Resident OpenVDN worker exited. " + current.get("error", "") +
                                   "\n" + failure_detail(state["instance"]))
            if current.get("phase") != previous_phase:
                previous_phase = current.get("phase")
                progress(previous_phase or "inference")
            if time.monotonic() - started > timeout:
                cancel_needed = True
                raise TimeoutError(f"OpenVDN request exceeded {timeout}s")
            time.sleep(.2)
    except BaseException:
        # Supervisor terminates the entire group on cancellation, then preloads again.
        # Do not signal a PID supplied by an HTTP request or a stale state record.
        if cancel_needed and not result_path.exists() and health().get("instance") == state["instance"]:
            atomic_json(BACKEND / "cancel.json", {"instance": state["instance"], "token": token})
        raise
