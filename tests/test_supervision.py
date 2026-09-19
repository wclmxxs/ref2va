import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import psutil
import pytest

from openvdn_comfy import backend
from openvdn_comfy.config import atomic_json
from openvdn_comfy.supervision import Policy, WorkerWatchdog, fail_pending

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name + '_under_test', ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Clock:
    now = 0.

    def __call__(self):
        return self.now


def test_policy_validation_and_capped_backoff(monkeypatch):
    policy = Policy.from_env()
    assert [policy.delay(n) for n in range(1, 9)] == [5, 10, 20, 40, 60, 60, 60, 60]
    for value in ('0', '-1', 'nan', 'inf', 'bad'):
        monkeypatch.setenv('REF2VA_REQUEST_TIMEOUT', value)
        with pytest.raises(ValueError):
            Policy.from_env()
    monkeypatch.setenv('REF2VA_REQUEST_TIMEOUT', '900')
    assert Policy.from_env().request_timeout == 900
    monkeypatch.setenv('REF2VA_RESTART_MAX_DELAY', '1')
    with pytest.raises(ValueError):
        Policy.from_env()


@pytest.fixture
def watch(tmp_path):
    clock = Clock()
    policy = Policy(startup_timeout=30, request_timeout=20, idle_timeout=5)
    watcher = WorkerWatchdog(tmp_path, 'new', policy, clock)
    worker = types.SimpleNamespace(poll=lambda: None)
    state = {'instance': 'new', 'status': 'ready', 'ready': True, 'phase': 'idle'}
    return watcher, worker, state, clock, tmp_path


def test_startup_has_separate_deadline_and_ignores_old_errors(watch):
    watcher, worker, state, clock, path = watch
    atomic_json(path / 'errors/old-1.json', {'traceback': 'old OOM'})
    state.update(ready=False, status='loading', phase='warming_up')
    clock.now = 21
    assert watcher.check(worker, state) is None
    clock.now = 31
    assert 'startup exceeded 30s' in watcher.check(worker, state)


def test_idle_watches_all_ranks_without_gpu_utilization_checks(watch):
    watcher, worker, state, clock, path = watch
    for t in range(12):
        clock.now = t
        for rank in range(8):
            if rank != 7 or t < 5:
                atomic_json(path / f'heartbeats/new-{rank}.json', {'sequence': t})
        result = watcher.check(worker, state)
        if t <= 9:
            assert result is None
        else:
            assert 'Rank 7 idle heartbeat stalled' in result


def test_idle_liveness_clock_resets_after_long_successful_request(watch):
    watcher, worker, state, clock, path = watch
    assert watcher.check(worker, state) is None
    atomic_json(path / 'command.json', {'instance': 'new', 'token': 'a'})
    state.update(status='busy', phase='denoising')
    clock.now = 1
    assert watcher.check(worker, state) is None
    clock.now = 19
    assert watcher.check(worker, state) is None
    atomic_json(path / 'results/a.json', {'ok': True})
    state.update(status='ready', phase='idle')
    assert watcher.check(worker, state) is None
    clock.now = 23
    assert watcher.check(worker, state) is None


def test_stuck_gpu_request_times_out_even_with_fresh_heartbeats(watch):
    watcher, worker, state, clock, path = watch
    atomic_json(path / 'command.json', {'instance': 'new', 'token': 'a'})
    assert watcher.check(worker, state) is None
    for rank in range(8):
        atomic_json(path / f'heartbeats/new-{rank}.json', {'sequence': 99})
    clock.now = 21
    state['phase'] = 'denoising'
    assert 'GPU request exceeded 20s' in watcher.check(worker, state)
    fail_pending(path, 'new', 'watchdog timeout')
    assert json.loads((path / 'results/a.json').read_text())['error'] == 'watchdog timeout'
    assert watcher.check(worker, state) is None


def test_failure_result_is_not_overwritten_or_replayed(watch):
    watcher, worker, state, clock, path = watch
    atomic_json(path / 'command.json', {'instance': 'old', 'token': 'old'})
    fail_pending(path, 'new', 'oom')
    assert not (path / 'results/old.json').exists()
    atomic_json(path / 'command.json', {'instance': 'new', 'token': 'a'})
    for result in ({'ok': True, 'metrics': {}}, {'ok': False, 'error': 'original OOM'}):
        atomic_json(path / 'results/a.json', result)
        fail_pending(path, 'new', 'restart')
        assert json.loads((path / 'results/a.json').read_text()) == result


@pytest.mark.parametrize('kind', ['exit', 'rank_error', 'failed_state'])
def test_gpu_failures_detected_without_waiting_for_request_deadline(watch, kind):
    watcher, worker, state, clock, path = watch
    if kind == 'exit':
        worker.poll = lambda: -9
    elif kind == 'rank_error':
        atomic_json(path / 'errors/new-5.json', {'traceback': 'torch.OutOfMemoryError'})
    else:
        state.update(status='failed', error='CUDA illegal access')
    assert watcher.check(worker, state)


def test_health_gated_by_supervisor_and_controller_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, 'BACKEND', tmp_path)
    identity = {'pid': os.getpid(), 'created': psutil.Process().create_time()}
    atomic_json(tmp_path / 'owner.json', {**identity, 'instance': 'worker'})
    atomic_json(tmp_path / 'state.json', {'instance': 'worker', 'status': 'ready'})
    assert backend.health()['ready']
    supervisor = {'controller': identity, 'instance': 'worker', 'status': 'recovering', 'restart_count': 1}
    atomic_json(tmp_path / 'supervisor.json', supervisor)
    assert not backend.health()['ready']
    supervisor['status'] = 'loading'
    atomic_json(tmp_path / 'supervisor.json', supervisor)
    assert backend.health()['ready']
    supervisor['controller']['created'] -= 10
    atomic_json(tmp_path / 'supervisor.json', supervisor)
    assert not backend.health()['ready']


@pytest.mark.parametrize('failure', ['oom', 'startup', 'timeout', 'cancel'])
def test_supervisor_recovers_worker_keeps_ui_and_does_not_replay(tmp_path, monkeypatch, failure):
    serve = load_script('serve')
    clock = Clock()
    monkeypatch.setattr(serve, 'BACKEND', tmp_path)
    monkeypatch.setattr(backend, 'BACKEND', tmp_path)
    monkeypatch.setattr(serve, 'retire_previous_server', lambda: None)
    monkeypatch.setattr(serve.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(serve.subprocess, 'run', lambda *args, **kwargs: None)
    monkeypatch.setattr(serve.Policy, 'from_env', lambda: Policy(startup_timeout=2, request_timeout=2,
                                                               idle_timeout=60, restart_delay=1))
    monkeypatch.setattr(serve.time, 'monotonic', clock)
    monkeypatch.setattr(serve, 'WorkerWatchdog', lambda *args: WorkerWatchdog(*args, clock=clock))
    cleanup, free, launched, stopped, ui_launches = [], [], [], [], []
    monkeypatch.setattr(serve, 'clear_gpu_applications', lambda: cleanup.append(True))
    monkeypatch.setattr(serve, 'ensure_free_gpus', lambda: free.append(True))
    worker = types.SimpleNamespace(pid=os.getpid(), poll=lambda: None, name='worker')
    ui = types.SimpleNamespace(pid=os.getpid(), poll=lambda: None, name='ui')
    identity = {'pid': os.getpid(), 'created': psutil.Process().create_time()}

    def launch_worker():
        instance = str(len(launched) + 1)
        launched.append(instance)
        atomic_json(tmp_path / 'owner.json', {**identity, 'instance': instance})
        loading = instance == '1' and failure == 'startup'
        atomic_json(tmp_path / 'state.json', {'instance': instance, 'status': 'loading' if loading else 'ready', 'phase': 'idle'})
        return worker, instance

    monkeypatch.setattr(serve, 'launch_worker', launch_worker)
    monkeypatch.setattr(serve, 'stop_group', lambda p: stopped.append(p.name))

    def launch_ui(command, **kwargs):
        assert '--cpu' in command
        ui_launches.append(True)
        return ui

    monkeypatch.setattr(serve.subprocess, 'Popen', launch_ui)
    injected = False

    def tick(seconds):
        nonlocal injected
        clock.now += seconds
        if len(launched) == 2 and backend.health()['ready'] and len(ui_launches) == 1:
            assert cleanup == [True]  # No repeated unrelated GPU/application cleanup.
            assert stopped == ['worker']
            assert backend.read_json(tmp_path / 'supervisor.json')['restart_count'] == 1
            if failure != 'startup':
                assert backend.read_json(tmp_path / 'results/request.json')['ok'] is False
                assert backend.read_json(tmp_path / 'command.json')['instance'] == '1'
            raise KeyboardInterrupt
        if not injected and failure != 'startup' and ui_launches:
            injected = True
            atomic_json(tmp_path / 'command.json', {'instance': '1', 'token': 'request'})
            if failure == 'oom':
                atomic_json(tmp_path / 'errors/1-4.json', {'traceback': 'CUDA out of memory'})
            if failure == 'cancel':
                atomic_json(tmp_path / 'cancel.json', {'instance': '1', 'token': 'request'})
        assert clock.now < 20

    monkeypatch.setattr(serve.time, 'sleep', tick)
    monkeypatch.setattr(serve, 'failure_detail', lambda instance: 'original diagnostics')
    with pytest.raises(KeyboardInterrupt):
        serve.main()
    assert launched == ['1', '2']
    assert len(ui_launches) == 1
    assert free == [True, True]
    assert stopped == ['worker', 'ui', 'worker']
    assert backend.read_json(tmp_path / 'supervisor.json')['status'] == 'stopped'
    assert (tmp_path / 'failures/1.json').is_file()


def test_background_start_is_detached_idempotent_and_stoppable(tmp_path, monkeypatch):
    service = load_script('service')
    root = tmp_path / 'path with spaces'
    root.mkdir()
    # exec is the same PID transition used by real deploy.sh -> serve.py.
    (root / 'scripts').mkdir()
    (root / 'scripts/serve.py').write_text('import time\nwhile True: time.sleep(0.05)\n')
    (root / 'deploy.sh').write_text('#!/bin/bash\nexec ' + sys.executable + ' "$(dirname "$0")/scripts/serve.py"\n')
    directory = root / '.runtime/backend'
    directory.mkdir(parents=True)
    monkeypatch.setattr(service, 'ROOT', root)
    monkeypatch.setattr(service, 'BACKEND', directory)
    monkeypatch.setattr(service, 'LOG', root / '.runtime/service.log')
    monkeypatch.setattr(service, 'RECORD', directory / 'service.json')
    try:
        service.start([])
        record = backend.read_json(directory / 'service.json')
        process = service.controller()
        assert process.pid == record['pid']
        assert os.getpgid(process.pid) == process.pid
        assert os.getsid(process.pid) == process.pid
        service.start([])
        assert service.controller().pid == process.pid
        service.stop()
        assert service.controller() is None
    finally:
        service.stop()


def test_service_ignores_reused_and_unrelated_pid_records(tmp_path, monkeypatch):
    service = load_script('service')
    monkeypatch.setattr(service, 'BACKEND', tmp_path)
    monkeypatch.setattr(service, 'RECORD', tmp_path / 'service.json')
    for created in (psutil.Process().create_time() - 1, psutil.Process().create_time()):
        atomic_json(tmp_path / 'service.json', {'pid': os.getpid(), 'created': created})
        assert service.controller() is None


def test_invalid_restart_config_does_not_stop_existing_service(tmp_path, monkeypatch):
    service = load_script('service')
    monkeypatch.setattr(service, 'BACKEND', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['service.py', 'restart'])
    monkeypatch.setenv('REF2VA_REQUEST_TIMEOUT', 'nan')
    stops = []
    monkeypatch.setattr(service, 'stop', lambda: stops.append(True))
    with pytest.raises(ValueError):
        service.main()
    assert stops == []


def test_client_timeout_signals_failure_recovery(tmp_path, monkeypatch):
    from dataclasses import asdict
    from openvdn_comfy.config import Settings
    monkeypatch.setattr(backend, 'BACKEND', tmp_path)
    identity = {'pid': os.getpid(), 'created': psutil.Process().create_time()}
    atomic_json(tmp_path / 'owner.json', {**identity, 'instance': 'worker'})
    atomic_json(tmp_path / 'state.json', {'instance': 'worker', 'status': 'ready',
                'profile': {key: getattr(Settings(), key) for key in backend.PROFILE_FIELDS}})
    with pytest.raises(TimeoutError):
        backend.call_worker({'settings': asdict(Settings())}, timeout=.001)
    command = backend.read_json(tmp_path / 'command.json')
    cancel = backend.read_json(tmp_path / 'cancel.json')
    assert cancel == {'instance': 'worker', 'token': command['token'], 'reason': 'timeout'}


def test_stop_during_preflight_reaps_shell_child(tmp_path, monkeypatch):
    import time
    import shlex
    service = load_script('service')
    root = tmp_path/'service';(root/'scripts').mkdir(parents=True)
    pidfile = root/'doctor-pid'
    (root/'scripts/doctor.py').write_text(
        'import os,time\nfrom pathlib import Path\n'
        f'Path({str(pidfile)!r}).write_text(str(os.getpid()))\ntime.sleep(30)\n')
    (root/'deploy.sh').write_text('#!/bin/bash\n'+shlex.quote(sys.executable)+' '+shlex.quote(str(root/'scripts/doctor.py'))+'\n')
    directory=root/'.runtime/backend';directory.mkdir(parents=True)
    monkeypatch.setattr(service,'ROOT',root);monkeypatch.setattr(service,'BACKEND',directory)
    monkeypatch.setattr(service,'LOG',root/'.runtime/service.log');monkeypatch.setattr(service,'RECORD',directory/'service.json')
    child = None
    try:
        service.start([])
        deadline=time.monotonic()+3
        while not pidfile.exists() and time.monotonic()<deadline: time.sleep(.01)
        child=psutil.Process(int(pidfile.read_text()))
        service.stop()
        assert not child.is_running() or child.status()==psutil.STATUS_ZOMBIE
    finally:
        service.stop()
        if child and child.is_running():
            try: child.kill()
            except psutil.NoSuchProcess: pass
