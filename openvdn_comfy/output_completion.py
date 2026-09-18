"""Bounded CPU completion after all CUDA work and device-to-host copies finish."""
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore
import time
import traceback

from .config import atomic_json


class OutputCompletions:
    def __init__(self, backend, instance, capacity=2):
        self.backend, self.instance = backend, instance
        self.slots = BoundedSemaphore(capacity)
        self.pool = ThreadPoolExecutor(max_workers=capacity, thread_name_prefix='output-completion')

    def reserve(self):
        self.slots.acquire()

    def submit(self, request, metrics, pending):
        def complete():
            try:
                output_timings, encoding = pending.finish()
                record = metrics['upstream']
                timings = record['timings']
                timings.update(output_timings)
                timings['output_wall_seconds'] = time.perf_counter() - pending.decode_started
                timings['decode_and_encode_seconds'] = timings['output_wall_seconds']
                timings['cpu_output_tail_seconds'] = time.monotonic() - submitted
                timings['worker_wall_seconds'] = timings['gpu_worker_seconds'] + timings['cpu_output_tail_seconds']
                record['output_encoding'] = encoding
                metrics['inference_process_seconds'] = timings['denoise_seconds'] + timings['decode_and_encode_seconds']
                atomic_json(request['output'] + '.inference.json', record)
                result = {'ok': True, 'metrics': metrics}
            except BaseException:
                # Encoder errors are job failures, not reasons to reset healthy GPUs.
                result = {'ok': False, 'error': 'CPU output failed: ' + traceback.format_exc()}
            try:
                atomic_json(self.backend / 'results' / f"{request['token']}.json", result)
            finally:
                self.slots.release()
        submitted = time.monotonic()
        future = self.pool.submit(complete)
        atomic_json(self.backend / 'gpu_results' / f"{request['token']}.json",
                    {'ok': True, 'pending_output': True, 'instance': self.instance,
                     'token': request['token'], 'gpu_worker_seconds': metrics['upstream']['timings']['gpu_worker_seconds']})
        return future
