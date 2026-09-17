"""Startup-only GPU reclamation, explicitly enabled for this dedicated eight-GPU host."""
import csv
import io
import json
import os
from pathlib import Path
import re
import subprocess
import time

import psutil

from .config import RUNTIME, atomic_json
from .gpu_check import selected_gpus, ensure_free_gpus


def compute_apps(uuids):
    result = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    return {int(row[1]): row for row in csv.reader(io.StringIO(result), skipinitialspace=True)
            if len(row) == 4 and row[0] in uuids and row[1].isdigit()}


def service_owner(cgroup):
    """Only a system application service; never stop login/session or host runtimes."""
    for path in cgroup.splitlines():
        match = re.search(r"/system\.slice/([^/]+\.service)(?:/|$)", path)
        if match:
            unit = match[1]
            if unit not in {"ssh.service", "sshd.service", "docker.service", "containerd.service",
                            "systemd-logind.service", "cron.service", "supervisor.service", "supervisord.service",
                            "kubelet.service", "k3s.service", "k3s-agent.service", "ecs.service",
                            "amazon-ssm-agent.service"} and not unit.startswith("systemd-"):
                return unit
    return None


def application_root(process):
    """Climb only through inference launchers, stopping before shells/SSH/tmux/init."""
    current = process
    while True:
        parent = current.parent()
        if parent is None or parent.pid <= 1 or parent.pid == os.getpid():
            return current
        args = parent.cmdline()
        if not args:
            return current
        executable = Path(args[0]).name
        inference = any(word in " ".join(args).lower() for word in
                        ("sglang", "sgl_diffusion", "vllm", "torch.distributed.run", "infer_ulysses"))
        if executable in {"bash", "sh", "zsh", "sshd", "tmux", "sudo", "systemd", "tini"} or not inference:
            return current
        current = parent


def stop_tree(process):
    # psutil checks PID creation time before signalling; a reused PID is not killed.
    children = process.children(recursive=True)
    targets = [process, *children]
    for target in targets:
        try:
            target.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=10)
    for target in alive:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=5)


def clear_gpu_applications():
    if os.environ.get("REF2VA_CLEAR_GPU_APPS", "1") != "1":
        ensure_free_gpus()
        return
    uuids = {row[1] for row in selected_gpus()}
    controller_cgroup = Path("/proc/self/cgroup").read_text()
    audit = {"started_at": time.time(), "actions": []}
    audit_path = RUNTIME / "gpu-cleanup.json"
    # Re-query to handle process exits and service restarts; never reuse stale PIDs.
    try:
        for attempt in range(3):
            apps = compute_apps(uuids)
            if not apps:
                ensure_free_gpus()
                return
            handled = set()
            for pid, row in apps.items():
                try:
                    process = psutil.Process(pid)
                    if pid <= 1 or pid in {os.getpid(), os.getppid()}:
                        raise RuntimeError(f"Refusing to stop the deployment controller (PID {pid})")
                    cgroup = Path(f"/proc/{pid}/cgroup").read_text()
                    container = re.search(r"(?:docker[-/])([0-9a-f]{64})(?:\.scope)?(?:/|$)", cgroup, re.M)
                    unit = service_owner(cgroup)
                    if container and container[1] not in controller_cgroup:
                        key = ("container", container[1])
                        command = ["docker", "stop", "--time", "15", container[1]]
                    elif unit and unit != service_owner(controller_cgroup):
                        key = ("service", unit)
                        command = ["systemctl", "stop", unit]
                    else:
                        root = application_root(process)
                        key = ("process_tree", root.pid)
                        command = None
                    if key in handled:
                        continue
                    handled.add(key)
                    action = {"kind": key[0], "target": key[1], "gpu_process": row, "time": time.time()}
                    audit["actions"].append(action)
                    atomic_json(audit_path, audit)
                    print(f"Releasing GPUs: stopping {key[0]} {key[1]} ({row[2]})", flush=True)
                    if command:
                        subprocess.run(command, check=True, timeout=45)
                    else:
                        stop_tree(root)
                    action["stopped"] = True
                except (psutil.NoSuchProcess, FileNotFoundError):
                    # A disappeared /proc entry is benign; missing control tools aren't.
                    if psutil.pid_exists(pid):
                        raise
            time.sleep(2)
        remaining = compute_apps(uuids)
        if remaining:
            raise RuntimeError(f"GPU applications keep restarting; stop their supervisor: {json.dumps(remaining)}")
        ensure_free_gpus()
    finally:
        audit["finished_at"] = time.time()
        atomic_json(audit_path, audit)
