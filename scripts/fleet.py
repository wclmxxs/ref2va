"""Manage one eight-GPU instance or two four-GPU instances, with readiness gating."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openvdn_comfy.config import atomic_json
from openvdn_comfy.backend import read_json
from openvdn_comfy.hardware import GPU_SPECS
from openvdn_comfy.host_identity import host_identity
from openvdn_comfy.supervision import Policy

MANIFEST = ROOT / '.runtime/fleet.json'
PYTHON = ROOT / '.venv-ui/bin/python'


def plans(gpu_type, gpus, port, devices=None):
    if gpu_type not in GPU_SPECS or gpus not in (4, 8) or not 1 <= port <= 65535 - (gpus == 4):
        raise ValueError('Choose h200|b200|b300, 4|8 GPUs per worker and a valid base port')
    devices = devices or ','.join(map(str, range(8)))
    ids = [x.strip() for x in devices.split(',')]
    if len(ids) != 8 or len(set(ids)) != 8 or not all(ids):
        raise ValueError('This launcher needs 8 distinct visible GPUs: one 8-GPU worker or two 4-GPU workers')
    return [{'REF2VA_GPU_TYPE': gpu_type, 'REF2VA_GPUS': str(gpus),
             'REF2VA_PORT': str(port + i), 'CUDA_VISIBLE_DEVICES': ','.join(ids[i*gpus:(i+1)*gpus]),
             'REF2VA_INSTANCE': '' if gpus == 8 else f'worker-{i}'} for i in range(8//gpus)]


def invoke(action, plan, args=(), capture=False):
    # Never replay addresses stored in a manifest copied from another machine.
    env = {**os.environ, **{k: v for k, v in plan.items() if k != 'PUBLIC_BASE_URL'},
           'REF2VA_MANAGED_INSTANCE': '1'}
    if plan.get('REF2VA_INSTANCE'):
        index = plan['REF2VA_INSTANCE'].rsplit('-', 1)[1]
        env['PUBLIC_BASE_URL'] = os.environ.get(f'REF2VA_PUBLIC_BASE_URL_{index}', '')
    return subprocess.run([str(PYTHON), str(ROOT / 'scripts/service.py'), action, *args], cwd=ROOT,
                          env=env, check=True, text=True, capture_output=capture)


def stop_all():
    for instance in ('', 'worker-0', 'worker-1'):
        invoke('stop', {'REF2VA_INSTANCE': instance})


def runtime(plan):
    return ROOT / '.runtime' / 'instances' / plan['REF2VA_INSTANCE'] if plan['REF2VA_INSTANCE'] else ROOT / '.runtime'


def fingerprint(desired, ui_args):
    digest = hashlib.sha256()
    files = [ROOT / 'deploy.sh', ROOT / 'sources.lock.json', *ROOT.glob('constraints-*.txt')]
    files += list((ROOT / 'scripts').glob('*.py')) + list((ROOT / 'openvdn_comfy').rglob('*.py'))
    # Native kernel changes must restart an otherwise-ready cloned deployment.
    files += list((ROOT / 'openvdn_comfy').rglob('*.cu')) + list((ROOT / 'openvdn_comfy').rglob('*.cuh'))
    for path in sorted(files):
        digest.update(str(path.relative_to(ROOT)).encode()); digest.update(path.read_bytes())
    config = {key: value for key, value in os.environ.items()
              if key.startswith(('REF2VA_', 'NCCL_', 'TORCH', 'TRITON_')) or key in ('PUBLIC_BASE_URL', 'OMP_NUM_THREADS', 'CUDA_HOME')}
    # Internal process marker does not change runtime behavior.
    config.pop('REF2VA_MANAGED_INSTANCE', None)
    digest.update(json.dumps([desired, ui_args, config], sort_keys=True).encode())
    return digest.hexdigest()


def get_status(plan):
    return json.loads(invoke('status', plan, capture=True).stdout)


def check_ports(selected):
    listen = os.environ.get('REF2VA_LISTEN', '0.0.0.0')
    for plan in selected:
        family = socket.AF_INET6 if ':' in listen else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((listen, int(plan['REF2VA_PORT'])))


def wait_ready(selected, timeout):
    started, reported = time.monotonic(), {}
    while True:
        ready = True
        for plan in selected:
            status = get_status(plan)
            port = plan['REF2VA_PORT']
            if not status['running']:
                raise RuntimeError(f'API {port}: controller exited; see {runtime(plan) / "service.log"}')
            supervision = status.get('supervision', {})
            if supervision.get('status') in ('recovering', 'stopped'):
                raise RuntimeError(f'API {port}: startup failed: {supervision.get("last_error", "worker stopped")}')
            ready = ready and status.get('ready', False)
            phase = 'ready' if status.get('ready') else status.get('phase') or 'checking_environment'
            previous, last = reported.get(port, (None, started - 15))
            now = time.monotonic()
            if phase != previous or now - last >= 15:
                print(f'API {port} / {plan["REF2VA_INSTANCE"] or "worker"}: {phase} ({now - started:.0f}s)', flush=True)
                reported[port] = (phase, now)
        if ready:
            print('All models warmed up and APIs ready. Services remain running in the background.', flush=True)
            for plan in selected:
                print(f'API port {plan["REF2VA_PORT"]}; use this machine\'s current IP or DNS name.', flush=True)
            return
        if time.monotonic() - started >= timeout:
            raise TimeoutError(f'Startup exceeded {timeout:g}s before every API became ready')
        time.sleep(2)


def diagnostics(selected):
    for plan in selected:
        for name in ('service.log', 'backend/worker.log'):
            path = runtime(plan) / name
            if path.is_file():
                with path.open('rb') as stream:
                    stream.seek(max(0, path.stat().st_size - 12000))
                    tail = stream.read().decode(errors='replace')
                print(f'\n==> {path} <==\n{tail}', file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['ensure', 'up', 'restart', 'stop', 'status', 'logs', 'foreground'])
    parser.add_argument('--gpu-type', choices=list(GPU_SPECS), default=os.environ.get('REF2VA_GPU_TYPE', 'h200'))
    parser.add_argument('--gpus', type=int, choices=[4,8], default=int(os.environ.get('REF2VA_GPUS','8')))
    parser.add_argument('--port', type=int, default=int(os.environ.get('REF2VA_PORT','8188')))
    parser.add_argument('--wait-timeout', type=float, default=None)
    args, ui_args = parser.parse_known_args()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with (MANIFEST.parent / 'fleet.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        saved = read_json(MANIFEST, {})
        existing = saved.get('instances', [])
        if args.action == 'stop':
            stop_all()
            return
        desired = existing if args.action in ('status', 'logs') and existing else plans(
            args.gpu_type, args.gpus, args.port, os.environ.get('CUDA_VISIBLE_DEVICES'))
        if args.action in ('status', 'logs'):
            if args.action == 'status':
                for plan in desired:
                    print(f"{plan['REF2VA_GPU_TYPE']} / {plan['REF2VA_GPUS']} GPUs / port {plan['REF2VA_PORT']}", flush=True)
                    invoke('status', plan)
                return
            paths = [runtime(plan) / 'service.log' for plan in desired]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True); path.touch(exist_ok=True)
            fcntl.flock(lock, fcntl.LOCK_UN)
            os.execvp('tail', ['tail', '-n', '50', '-F', *map(str, paths)])
        timeout = args.wait_timeout if args.wait_timeout is not None else float(os.environ.get(
            'REF2VA_LAUNCH_TIMEOUT', Policy.from_env().startup_timeout + 300))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('--wait-timeout / REF2VA_LAUNCH_TIMEOUT must be positive and finite')
        if len(desired) == 2 and os.environ.get('PUBLIC_BASE_URL'):
            if not all(os.environ.get(f'REF2VA_PUBLIC_BASE_URL_{i}') for i in range(2)):
                raise ValueError('Two APIs need separate REF2VA_PUBLIC_BASE_URL_0 and REF2VA_PUBLIC_BASE_URL_1; '
                                 'or unset PUBLIC_BASE_URL to use each request origin')
        for plan in desired:
            invoke('validate', plan)
        signature, host = fingerprint(desired, ui_args), host_identity()
        reuse = (args.action in ('ensure', 'up') and saved.get('fingerprint') == signature
                 and saved.get('host') == host and existing == desired
                 and all(get_status(plan)['running'] for plan in desired))
        if not reuse:
            stop_all()
            check_ports(desired)
        else:
            print('Current code/configuration already running; verifying API readiness.', flush=True)
        atomic_json(MANIFEST, {'instances': desired, 'fingerprint': signature, 'host': host})
        started = []
        try:
            for plan in desired:
                started.append(plan)
                invoke('up', plan, ui_args)
            # Bootstrap's deployment lock still serializes other unified starts.
            # Let status/logs/stop work from another terminal during model load.
            fcntl.flock(lock, fcntl.LOCK_UN)
            wait_ready(desired, timeout)
        except BaseException:
            diagnostics(desired)
            for plan in started:
                try:
                    invoke('stop', plan)
                except Exception as error:
                    print(f'Failed to stop {plan["REF2VA_INSTANCE"]}: {error}', file=sys.stderr)
            raise
    if args.action == 'foreground':
        def interrupted(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        try:
            while True:
                if not all(get_status(plan)['running'] for plan in desired):
                    raise RuntimeError('An instance controller exited; see bash deploy.sh logs')
                time.sleep(2)
        except KeyboardInterrupt:
            pass
        finally:
            for plan in desired:
                invoke('stop', plan)


if __name__ == '__main__':
    def cancelled(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, cancelled)
    signal.signal(signal.SIGHUP, cancelled)
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f'Startup failed: {error}', file=sys.stderr)
        sys.exit(1)
