"""Detached service controls. PID creation times prevent signalling reused PIDs."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psutil
from openvdn_comfy.backend import BACKEND, health, read_json, same_process, startup_settings, parallel_vae_enabled
from openvdn_comfy.config import RUNTIME, atomic_json
from openvdn_comfy.gpu_cleanup import stop_tree
from openvdn_comfy.supervision import Policy
from openvdn_comfy.compile_cache import cache_settings
from openvdn_comfy.exact_runtime import enabled
from openvdn_comfy.host_identity import host_identity
from openvdn_comfy.network import listen_value, health_urls, check_bindings

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
    check_bindings([(listen_value(), 0)])  # Fail before stopping workers or loading any models.
    Policy.from_env()
    startup_settings()
    parallel_vae_enabled()
    cache_settings()
    enabled('REF2VA_EXACT_RUNTIME')
    enabled('REF2VA_ASYNC_OUTPUT')
    enabled('REF2VA_PIPELINE_OUTPUT')


def stop():
    process = controller()
    if process is not None:
        print(f'Stopping Ref2VA controller {process.pid}', flush=True)
        # During doctor/bootstrap the controller is still a shell. A shell can
        # exit on TERM without reaping its foreground Python children.
        children = process.children(recursive=True)
        try:
            process.terminate()
            process.wait(timeout=35)
        except psutil.NoSuchProcess:
            pass
        except psutil.TimeoutExpired:
            stop_tree(process)
        for child in children:
            try:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    stop_tree(child)
            except psutil.NoSuchProcess:
                pass
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
        log_offset = output.tell()
        log_inode = os.fstat(output.fileno()).st_ino
        process = subprocess.Popen(['bash', str(ROOT / 'deploy.sh'), 'start', *args], cwd=ROOT,
                                   stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    atomic_json(RECORD, {'pid': process.pid, 'created': psutil.Process(process.pid).create_time(), 'host': host_identity(),
                         'log_offset': log_offset, 'log_inode': log_inode, 'listen': listen_value()})
    time.sleep(.2)
    if process.poll() is not None:
        raise RuntimeError(f'Startup exited with code {process.returncode}; see {LOG}\n{startup_log_tail()}')
    print(f'Ref2VA loading (PID {process.pid}). Log: {LOG}', flush=True)


def active_listen():
    record = read_json(RECORD, {})
    if same_process(record):
        # Legacy records lack this field. Use the explicit environment or old
        # IPv4 default when inspecting a controller started before this change.
        return record.get('listen', os.environ.get('REF2VA_LISTEN', '0.0.0.0'))
    return listen_value()


def api_ready(state):
    if not state.get('ready') or not same_process(read_json(BACKEND / 'ui.json', {})):
        return False
    try:
        # Ignore inherited HTTP_PROXY settings: readiness must reach this local API.
        for url in health_urls(os.environ.get('REF2VA_PORT', '8188'), active_listen()):
            with build_opener(ProxyHandler({})).open(url, timeout=2) as response:
                actual = json.load(response)
            if actual.get('ready') is not True or actual.get('instance') != state.get('instance'):
                return False
        return True
    except (OSError, ValueError):
        return False


def startup_log_tail():
    """Only this boot's last start, never the old worker's SIGTERM/log history."""
    record = read_json(RECORD, {})
    offset = record.get('log_offset')
    if record.get('host') != host_identity() or not isinstance(offset, int) or offset < 0:
        return ''
    try:
        with LOG.open('rb') as stream:
            stat = os.fstat(stream.fileno())
            if stat.st_ino != record.get('log_inode') or offset > stat.st_size:
                return ''
            stream.seek(max(offset, stat.st_size - 8000))
            return stream.read().decode(errors='replace').strip()
    except OSError:
        return ''


def snapshot():
    process = controller()
    state = health()
    supervision = state.get('supervision', {})
    supervisor_owner = supervision.get('controller', {})
    current = bool(process and supervisor_owner.get('pid') == process.pid and same_process(supervisor_owner))
    return {'running': process is not None, 'controller_pid': process.pid if process else None,
            'listen': active_listen(), 'health_urls': health_urls(os.environ.get('REF2VA_PORT', '8188'), active_listen()),
            'ready': bool(current and api_ready(state)), 'worker_ready': bool(current and state['ready']),
            'phase': state.get('phase') if current else 'checking_environment',
            'supervision': supervision if current else {}, 'log': str(LOG),
            'startup_error': startup_log_tail() if process is None else None}


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if action not in ('up', 'restart', 'stop', 'status', 'logs', 'validate'):
        raise ValueError('Expected up, restart, stop, status or logs')
    if action == 'validate':
        validate_start()
        return
    BACKEND.mkdir(parents=True, exist_ok=True)
    if action == 'logs':
        LOG.touch(exist_ok=True)
        os.execvp('tail', ['tail', '-n', '100', '-F', str(LOG)])
    if action == 'status':
        print(json.dumps(snapshot(), ensure_ascii=False, indent=2))
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
