"""Bounded video writer consuming temporally blended VAE chunks as they arrive."""
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue, Empty, Full
from types import SimpleNamespace
import time

from .fast_output import pixel_chunk, write_mp4


class StreamingMP4:
    def __init__(self, plan, output, sample_rate, mean, std, *, verify=False):
        self.plan, self.mean, self.std = plan, mean, std
        self.timings, self.sent, self.closed = {}, 0, False
        self.queue, self.audio = Queue(maxsize=4), Future()
        self.captured = [] if verify else None
        self.started = time.perf_counter()
        self.first_chunk = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='vae-mp4')
        self.writer = self.pool.submit(write_mp4, None, self.audio.result, sample_rate, plan, output,
                                       mean, std, self.timings, pixel_chunks=self.chunks())

    def chunks(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            yield item

    def put(self, item):
        while True:
            if self.writer.done():
                self.writer.result()  # Raise the encoder failure, not a queue deadlock.
                raise RuntimeError('Video writer finished before all frames were submitted')
            try:
                self.queue.put(item, timeout=.05)
                return
            except Full:
                continue

    def submit(self, chunk):
        """Called only after native temporal blending; clip tails are trimmed here."""
        if self.closed:
            raise RuntimeError('Video writer is closed')
        count = min(chunk.shape[2], self.plan.output_frames - self.sent)
        plan = SimpleNamespace(output_frames=count, height=self.plan.height, width=self.plan.width)
        for start in range(0, count, 8):
            pixels = pixel_chunk(chunk, start, plan, self.mean, self.std, self.timings)
            if self.first_chunk is None:
                self.first_chunk = time.perf_counter() - self.started
            if self.captured is not None:
                self.captured.append((self.sent + start, pixels))
            self.put((self.sent + start, pixels))
        self.sent += count

    def check(self, video):
        import torch
        checked = 0
        if self.captured is not None:
            for start, actual in self.captured:
                expected = pixel_chunk(video, start, self.plan, self.mean, self.std, {}, chunk_size=len(actual))
                if not torch.equal(actual, expected):
                    raise RuntimeError('Streaming output pixel parity failed')
                checked += 1
        return {'checked': self.captured is not None, 'exact': True if self.captured is not None else None,
                'chunks_checked': checked}

    def finish(self, audio, video):
        if self.sent != self.plan.output_frames:
            raise RuntimeError(f'Output has {self.sent} frames, expected {self.plan.output_frames}')
        parity = self.check(video)
        self.audio.set_result(audio)
        self.put(None)
        encoding = self.writer.result()
        self.closed = True
        self.pool.shutdown(wait=True)
        self.captured = None
        encoding.update(streaming_pixel_parity=parity, host_queue_chunks=4)
        self.timings['first_pixel_chunk_seconds'] = self.first_chunk
        self.timings['output_pipeline_wall_seconds'] = time.perf_counter() - self.started
        return self.timings, encoding

    def abort(self, error):
        if self.closed:
            return
        self.closed = True
        if not self.audio.done():
            self.audio.set_exception(error)
        # Discard queued frames, wake a blocked consumer, then join it before
        # removing the temporary file. No GPU work is performed by the consumer.
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break
        self.queue.put_nowait(None)
        try:
            self.writer.result()
        except BaseException:
            pass
        self.pool.shutdown(wait=True)
        self.captured = None
