"""Detached service controls. PID creation times prevent signalling reused PIDs."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psutil
from openvdn_comfy.backend import BACKEND, health, read_json, same_process, startup_settings, parallel_vae_enabled
from openvdn_comfy.config import RUNTIME, atomic_json
from openvdn_comfy.gpu_cleanup import stop_tree
from openvdn_comfy.supervision import Policy
from openvdn_comfy.compile_cache import cache_settings
from openvdn_comfy.exact_runtime import enabled

LOG = RUNTIME / 'service.log'
RECORD = BACKEND / 'service.json'


def controller():
    # service.json also covers the doctor phase before serve.py records its PID.
    for path in (RECORD, BACKEND / 'server.json'):
        record = read_json(path, {})
        if not same_process(record):
            continue
        try:
            process = psutil.Process(record['pid'])
            args = process.cmdline()
            if str(ROOT / 'scripts/serve.py') in args or str(ROOT / 'deploy.sh') in args:
                return process
        except psutil.NoSuchProcess:
            continue
    return None


def validate_start():
    Policy.from_env()
    startup_settings()
    parallel_vae_enabled()
    cache_settings()
    enabled('REF2VA_EXACT_RUNTIME')
    enabled('REF2VA_ASYNC_OUTPUT')


def stop():
    process = controller()
    if process is not None:
        print(f'Stopping Ref2VA controller {process.pid}', flush=True)
        try:
            process.terminate()
            process.wait(timeout=35)
        except psutil.NoSuchProcess:
            pass
        except psutil.TimeoutExpired:
            stop_tree(process)
    # Recover owned groups if the controller itself was SIGKILLed previously.
    for name in ('owner.json', 'ui.json'):
        record = read_json(BACKEND / name, {})
        if same_process(record):
            try:
                process = psutil.Process(record['pid'])
                args = process.cmdline()
                expected = ROOT / ('scripts/resident_worker.py' if name == 'owner.json' else '.deps/ComfyUI/main.py')
                if str(expected) not in args or os.getpgid(process.pid) != process.pid:
                    raise RuntimeError(f'Refusing to stop an unrecognized process from {name}')
                # Stop descendants before parent identity disappears; all were ours.
                stop_tree(process)
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass
    print('Ref2VA stopped.', flush=True)


def start(args):
    validate_start()  # Fail immediately for invalid deployment/watchdog configuration.
    process = controller()
    if process is not None:
        print(f'Ref2VA already running (PID {process.pid}); use bash deploy.sh restart to reload code/config.')
        return
    # Clean orphaned owned ranks before startup's general GPU cleanup runs.
    stop()
    with LOG.open('ab') as output:
        output.write(f'\n=== Ref2VA background start {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n'.encode())
        output.flush()
        process = subprocess.Popen(['bash', str(ROOT / 'deploy.sh'), 'start', *args], cwd=ROOT,
                                   stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    atomic_json(RECORD, {'pid': process.pid, 'created': psutil.Process(process.pid).create_time()})
    time.sleep(.2)
    if process.poll() is not None:
        raise RuntimeError(f'Startup exited with code {process.returncode}; see {LOG}')
    print(f'Ref2VA starting in background (PID {process.pid}); model loading is asynchronous.\n'
          f'GPU worker auto-recovery enabled. Log: {LOG}\n'
          'Check readiness: bash deploy.sh status', flush=True)


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if action not in ('up', 'restart', 'stop', 'status', 'logs'):
        raise ValueError('Expected up, restart, stop, status or logs')
    BACKEND.mkdir(parents=True, exist_ok=True)
    if action == 'logs':
        LOG.touch(exist_ok=True)
        os.execvp('tail', ['tail', '-n', '100', '-F', str(LOG)])
    if action == 'status':
        process = controller()
        state = health()
        print(json.dumps({'running': process is not None, 'controller_pid': process.pid if process else None,
                          'ready': state['ready'], 'phase': state.get('phase'),
                          'supervision': state.get('supervision', {}), 'log': str(LOG)}, ensure_ascii=False, indent=2))
        return
    with (BACKEND / 'service.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if action in ('up', 'restart'):
            validate_start()  # Never stop a healthy service for invalid new settings.
        if action in ('restart', 'stop'):
            stop()
        if action in ('up', 'restart'):
            start(sys.argv[2:])


if __name__ == '__main__':
    main()
