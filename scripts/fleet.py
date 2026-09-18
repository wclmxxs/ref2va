"""One command manages either one eight-GPU instance or two four-GPU instances."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openvdn_comfy.config import atomic_json
from openvdn_comfy.backend import read_json

MANIFEST = ROOT / '.runtime/fleet.json'
PYTHON = ROOT / '.venv-ui/bin/python'


def plans(gpu_type, gpus, port, devices=None):
    if gpu_type not in ('h200', 'b200') or gpus not in (4, 8) or not 1 <= port <= 65535 - (gpus == 4):
        raise ValueError('Choose h200|b200, 4|8 GPUs per worker and a valid base port')
    devices = devices or ','.join(map(str, range(8)))
    ids = [x.strip() for x in devices.split(',')]
    if len(ids) != 8 or len(set(ids)) != 8 or not all(ids):
        raise ValueError('This launcher needs 8 distinct visible GPUs: one 8-GPU worker or two 4-GPU workers')
    return [{'REF2VA_GPU_TYPE': gpu_type, 'REF2VA_GPUS': str(gpus),
             'REF2VA_PORT': str(port + i), 'CUDA_VISIBLE_DEVICES': ','.join(ids[i*gpus:(i+1)*gpus]),
             'REF2VA_INSTANCE': '' if gpus == 8 else f'worker-{i}'} for i in range(8//gpus)]


def invoke(action, plan, args=(), capture=False):
    env = {**os.environ, **plan, 'REF2VA_MANAGED_INSTANCE': '1'}
    return subprocess.run([str(PYTHON), str(ROOT / 'scripts/service.py'), action, *args], cwd=ROOT,
                          env=env, check=True, text=True, capture_output=capture)


def stop_all():
    # The only three supported runtime namespaces; also catches a legacy
    # foreground deployment which predates the fleet manifest.
    for instance in ('', 'worker-0', 'worker-1'):
        invoke('stop', {'REF2VA_INSTANCE': instance})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['up', 'restart', 'stop', 'status', 'logs', 'foreground'])
    parser.add_argument('--gpu-type', choices=['h200', 'b200'], default=os.environ.get('REF2VA_GPU_TYPE', 'h200'))
    parser.add_argument('--gpus', type=int, choices=[4,8], default=int(os.environ.get('REF2VA_GPUS','8')),
                        help='GPUs per worker; 4 launches two workers on an eight-GPU host')
    parser.add_argument('--port', type=int, default=int(os.environ.get('REF2VA_PORT','8188')),
                        help='First API port; second worker uses port+1')
    args, ui_args = parser.parse_known_args()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with (MANIFEST.parent / 'fleet.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = read_json(MANIFEST, {}).get('instances', [])
        if args.action == 'stop':
            stop_all()
            return
        # Status and logs use the saved topology even from a shell with a
        # different CUDA_VISIBLE_DEVICES or deployment configuration.
        desired = existing if args.action in ('status', 'logs') and existing else plans(
            args.gpu_type, args.gpus, args.port, os.environ.get('CUDA_VISIBLE_DEVICES'))
        if args.action in ('status', 'logs'):
            selected = existing or desired
            if args.action == 'status':
                for plan in selected:
                    print(f"{plan['REF2VA_GPU_TYPE']} / {plan['REF2VA_GPUS']} GPUs / port {plan['REF2VA_PORT']}", flush=True)
                    invoke('status', plan)
                return
            paths = [ROOT / '.runtime' / 'instances' / p['REF2VA_INSTANCE'] / 'service.log'
                     if p['REF2VA_INSTANCE'] else ROOT / '.runtime/service.log' for p in selected]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch(exist_ok=True)
            fcntl.flock(lock, fcntl.LOCK_UN)
            os.execvp('tail', ['tail', '-n', '50', '-F', *map(str, paths)])
        if len(desired) == 2:
            for index, plan in enumerate(desired):
                base = os.environ.get(f'REF2VA_PUBLIC_BASE_URL_{index}', '')
                if os.environ.get('PUBLIC_BASE_URL') and not base:
                    raise ValueError('Two APIs need separate REF2VA_PUBLIC_BASE_URL_0 and REF2VA_PUBLIC_BASE_URL_1; '
                                     'or unset PUBLIC_BASE_URL to use each request origin')
                if base:
                    plan['PUBLIC_BASE_URL'] = base
        # Validate every child before replacing any running instance.
        for plan in desired:
            invoke('validate', plan)
        if args.action == 'up' and existing and existing != desired:
            raise RuntimeError('Deployment selection changed; use restart --gpu-type ... --gpus ...')
        if args.action in ('restart', 'foreground'):
            stop_all()
        elif not existing and args.gpus == 4:
            legacy = json.loads(invoke('status', {'REF2VA_INSTANCE': ''}, capture=True).stdout)
            if legacy['running']:
                raise RuntimeError('An eight-GPU service is running; use restart --gpus 4 to replace it')
        atomic_json(MANIFEST, {'instances': desired})
        started = []
        try:
            for plan in desired:
                invoke('up', plan, ui_args)
                started.append(plan)
                print(f"API: http://localhost:{plan['REF2VA_PORT']}  GPUs: {plan['CUDA_VISIBLE_DEVICES']}", flush=True)
        except BaseException:
            for plan in started:
                invoke('stop', plan)
            raise
    if args.action == 'foreground':
        def interrupted(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        try:
            while True:
                for plan in desired:
                    if not json.loads(invoke('status', plan, capture=True).stdout)['running']:
                        raise RuntimeError('An instance controller exited; see bash deploy.sh logs')
                time.sleep(2)
        except KeyboardInterrupt:
            pass
        finally:
            for plan in desired:
                invoke('stop', plan)


if __name__ == '__main__':
    main()
