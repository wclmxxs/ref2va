"""CPU-only watchdog: deadlines and main-loop liveness, without GPU probes."""
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import time

from .backend import read_json
from .hardware import Hardware


@dataclass(frozen=True)
class Policy:
    startup_timeout: float = 3600
    request_timeout: float = 1800
    idle_timeout: float = 60
    restart_delay: float = 5
    restart_max_delay: float = 60
    stable_seconds: float = 300

    @classmethod
    def from_env(cls):
        values = {}
        for name, default in asdict(cls()).items():
            key = 'REF2VA_' + name.upper()
            value = float(os.environ.get(key, default))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be a finite positive number of seconds')
            values[name] = value
        if values['restart_max_delay'] < values['restart_delay']:
            raise ValueError('REF2VA_RESTART_MAX_DELAY must be >= REF2VA_RESTART_DELAY')
        return cls(**values)

    def delay(self, consecutive):
        return min(self.restart_max_delay, self.restart_delay * 2 ** min(max(0, consecutive - 1), 16))


class WorkerWatchdog:
    def __init__(self, directory, instance, policy, clock=time.monotonic):
        self.directory, self.instance, self.policy, self.clock = Path(directory), instance, policy, clock
        self.started = clock()
        self.ready_since = None
        self.token = None
        self.request_started = None
        self.idle_since = None
        self.beats = {}

    def check(self, process, state):
        now = self.clock()
        code = process.poll()
        if code is not None:
            return f'Worker exited with code {code}'
        for rank in range(Hardware.from_env().world_size):
            error = read_json(self.directory / 'errors' / f'{self.instance}-{rank}.json')
            if error:
                return f'Rank {rank} failed: {error.get("traceback", "unknown error")[-8000:]}'
        if state.get('instance') == self.instance and state.get('status') == 'failed':
            return 'Worker reported failure: ' + state.get('error', 'unknown error')
        if self.ready_since is None:
            if state.get('instance') == self.instance and state.get('ready'):
                self.ready_since = now
            elif now - self.started > self.policy.startup_timeout:
                return f'Worker startup exceeded {self.policy.startup_timeout:g}s ({state.get("phase")})'
            else:
                return None
        command = read_json(self.directory / 'command.json', {})
        token = command.get('token') if command.get('instance') == self.instance else None
        pending = token and not (self.directory / 'results' / f'{token}.json').exists() and not (self.directory / 'gpu_results' / f'{token}.json').exists()
        if pending:
            if token != self.token:
                self.token, self.request_started = token, now
            self.idle_since = None
            self.beats.clear()
            if now - self.request_started > self.policy.request_timeout:
                return f'GPU request exceeded {self.policy.request_timeout:g}s ({state.get("phase")})'
            return None
        # Idle is healthy without GPU utilization. Only the worker's actual main
        # loop renews these heartbeats; a background heartbeat thread could hide a hang.
        self.token, self.request_started = None, None
        if self.idle_since is None:
            self.idle_since = now
        for rank in range(Hardware.from_env().world_size):
            beat = read_json(self.directory / 'heartbeats' / f'{self.instance}-{rank}.json', {})
            signature = beat.get('sequence')
            previous, seen = self.beats.get(rank, (None, self.idle_since))
            if signature is not None and signature != previous:
                self.beats[rank] = (signature, now)
            elif now - seen > self.policy.idle_timeout:
                return f'Rank {rank} idle heartbeat stalled for {self.policy.idle_timeout:g}s'
        return None


def fail_pending(directory, instance, error):
    """Called after ranks are reaped, so a late success cannot race this result."""
    from .config import atomic_json
    directory = Path(directory)
    command = read_json(directory / 'command.json', {})
    token = command.get('token')
    if command.get('instance') == instance and token:
        result = directory / 'results' / f'{token}.json'
        if not result.exists():
            atomic_json(result, {'ok': False, 'error': error, 'worker_restarted': True})

    # A worker restart also invalidates CPU outputs owned by that process.
    for path in (directory / 'gpu_results').glob('*.json'):
        receipt = read_json(path, {})
        result = directory / 'results' / path.name
        if receipt.get('instance') == instance and not result.exists():
            atomic_json(result, {'ok': False, 'error': error, 'worker_restarted': True})
