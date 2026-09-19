"""Opt-in sampling of VIDEO frame statistics in the pinned linear-attention branch.

Sampling happens after the original features/convolutions, before A/B statistics.
Q, the text-state statistics, scans, gates and full-resolution readout stay native.
This is approximate; it does not reduce QKV projection or NCCL payload sizes.
"""
from contextlib import contextmanager
import hashlib
import inspect
import linecache
import textwrap
import types

RATIOS = (1.0, 0.5, 0.25)
SOURCE_HASH = 'bced583d3b29059e2b82a6c462f94a1439dcc9a58f9c088410e5bc1e89de734f'
STATS_OLD = '''        A, B = frame_statistics(key_by_frame, value_by_frame, beta, a_fp32=self.a_fp32,
                                inference=True)'''
STATS_NEW = '''        A, B = self._ref2va_linear_kv.statistics(
            key_by_frame, value_by_frame, beta, a_fp32=self.a_fp32, inference=True)'''


def validate_ratio(value):
    if type(value) not in (int, float) or value not in RATIOS:
        raise ValueError('linear_kv_keep_ratio must be 1.0, 0.5 or 0.25')


def kept_tokens(tokens, ratio):
    validate_ratio(ratio)
    if type(tokens) is not int or tokens < 1:
        raise ValueError('Video statistics require at least one spatial token per frame')
    stride = round(1 / ratio)
    return (tokens + stride - 1) // stride


def sample_indices(tokens, ratio, device):
    """One midpoint from each equal-width stratum of the flattened spatial grid.

    Integer arithmetic is deterministic, independent of seed/frame/rank/NFE, and
    does not advance the inference RNG. Odd grid sizes use ceil(S * ratio).
    """
    import torch
    count = kept_tokens(tokens, ratio)
    return ((2 * torch.arange(count, device=device, dtype=torch.int64) + 1) * tokens) // (2 * count)


def rewrite_readout(function):
    source = inspect.getsource(function)
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_HASH or source.count(STATS_OLD) != 1:
        raise RuntimeError('Pinned OpenVDN linear readout source changed; audit linear K/V sampling')
    source = textwrap.dedent(source.replace(STATS_OLD, STATS_NEW))
    path = '<ref2va-linear-kv/_readout_inference.py>'
    linecache.cache[path] = (len(source), None, source.splitlines(keepends=True), path)
    namespace = dict(function.__globals__)
    exec(compile(source, path, 'exec'), namespace)
    return namespace[function.__name__]


class LinearKVRuntime:
    def __init__(self, hybrids, runtime):
        self.runtime = runtime
        self.bindings = []
        rewritten = {}
        self.native_statistics = None
        for attn in hybrids:
            branch = attn.linear_attention
            native = branch._readout_inference
            function = native.__func__
            if function not in rewritten:
                rewritten[function] = rewrite_readout(function)
            statistics = function.__globals__['frame_statistics']
            if self.native_statistics is not None and statistics is not self.native_statistics:
                raise RuntimeError('Expected one native frame-statistics implementation')
            self.native_statistics = statistics
            branch._ref2va_linear_kv = self
            self.bindings.append((branch, native, types.MethodType(rewritten[function], branch)))
        self.active = False
        self.instrumented = False
        self.ratio = 1.0
        self.indices = {}
        self.observations = {}

    @contextmanager
    def request(self, ratio):
        validate_ratio(ratio)
        if self.active:
            raise RuntimeError('Linear K/V sampling cannot share concurrent requests')
        self.ratio, self.active = float(ratio), True
        self.instrumented = ratio < 1 or self.runtime.profile_enabled
        self.indices, self.observations = {}, {}
        try:
            for branch, native, sampled in self.bindings:
                # Full ratio without profiling takes the original bound method:
                # no sampling, extra tensor math or per-layer callback overhead.
                branch._readout_inference = sampled if self.instrumented else native
            yield self
        finally:
            for branch, native, _ in self.bindings:
                branch._readout_inference = native
            self.indices.clear()
            self.observations.clear()
            self.ratio, self.active = 1.0, False
            self.instrumented = False

    def statistics(self, key, value, beta, *, a_fp32, inference):
        import torch
        if not self.active or not inference:
            raise RuntimeError('Linear K/V sampling requires an active video inference request')
        if key.ndim != 4 or value.ndim != 4 or key.shape[:3] != value.shape[:3] or key.shape[:3] != beta.shape:
            raise ValueError('Expected per-frame K/V [F,H,S,d] and beta [F,H,S]')
        if torch.is_grad_enabled():
            raise RuntimeError('Linear K/V sampling is inference-only')
        frames, heads, tokens = key.shape[:3]
        count = kept_tokens(tokens, self.ratio)
        shape = (frames, heads, tokens, count)
        self.observations[shape] = self.observations.get(shape, 0) + 1
        if count < tokens:
            start = self.runtime.profile_start()
            identity = (tokens, key.device)
            if identity not in self.indices:
                self.indices[identity] = sample_indices(tokens, self.ratio, key.device)
            ids = self.indices[identity]
            key, value, beta = key.index_select(2, ids), value.index_select(2, ids), beta.index_select(2, ids)
            self.runtime.profile_end('linear_kv_select', start)
        start = self.runtime.profile_start()
        A, B = self.native_statistics(key, value, beta, a_fp32=a_fp32, inference=True)
        self.runtime.profile_end('linear_frame_statistics', start)
        if count < tokens:
            start = self.runtime.profile_start()
            # Keep the original scan/backend normalization at S, and estimate its
            # full sums. Scaling BOTH FP32 outputs preserves the A/B relative scale;
            # do not instead change beta or the delta backend's token count.
            weight = tokens / count
            A.mul_(weight)
            B.mul_(weight)
            self.runtime.profile_end('linear_kv_rescale', start)
        return A, B

    def report(self):
        if not self.active:
            raise RuntimeError('Linear K/V report requires an active request')
        shapes = [{'frames': f, 'heads': h, 'original_tokens_per_frame': s,
                   'kept_tokens_per_frame': k, 'actual_keep_ratio': k / s,
                   'statistics_weight': s / k, 'calls': calls}
                  for (f, h, s, k), calls in self.observations.items()]
        return {'rank': self.runtime.rank, 'requested_keep_ratio': self.ratio,
                'instrumented': self.instrumented,
                'statistics_calls': sum(self.observations.values()) if self.instrumented else None, 'shapes': shapes,
                'approximate': any(k < s for _, _, s, k in self.observations)}


def summarize_linear_kv(records):
    ratio = records[0]['requested_keep_ratio']
    if any(r['requested_keep_ratio'] != ratio for r in records):
        raise RuntimeError('Linear K/V keep ratio differs across GPU ranks')
    return {'requested_keep_ratio': ratio, 'enabled': ratio < 1,
            'approximate': any(r['approximate'] for r in records),
            'sampling': 'uniform_frame_strata_midpoint_v1',
            'scope': 'video K/V/beta statistics only; text state and full-resolution Q/readout unchanged',
            'correction': 'A and B multiplied by original_tokens / kept_tokens; original delta normalization',
            'by_rank': records}
