"""Per-request exact-window kernel comparison and optional padding-key isolation.

Isolation uses BlockMask (full valid blocks plus boundary blocks), not score_mod.
It deliberately remains opt-in until latency and video quality are measured.
"""

def isolated_mask_mod(layout, bounds, prefix_tokens, device):
    import torch
    lo = torch.tensor([a for a, _ in bounds], device=device)
    hi = torch.tensor([b for _, b in bounds], device=device)
    # Capture tensor metadata, not the actual prefix length as a Python guard.
    prefix = torch.tensor(prefix_tokens, dtype=torch.int32, device=device)
    start, end, tpf, frames = layout.video_start, layout.video_end, layout.tokens_per_frame, layout.num_frames

    def mask(batch, head, q, k):
        qv = (q >= start) & (q < end)
        kv = (k >= start) & (k < end)
        qf = torch.clamp((q - start) // tpf, 0, frames - 1)
        kf = (k - start) // tpf
        window = (kf >= lo[qf]) & (kf <= hi[qf])
        return qv, kv, qf, kf, window, (k < prefix) | (k >= start)
    return mask


def build_isolated_mask(layout, bounds, prefix_tokens, device, anchors):
    import torch
    from src.models.softmax_attention import flex_attention as module
    block_size = (256, 128) if torch.cuda.get_device_capability(device)[0] >= 10 else 128
    key = ('ref2va_isolated', layout.seq_len, layout.video_start, layout.video_end,
           layout.num_frames, layout.tokens_per_frame, tuple(bounds), prefix_tokens, anchors, str(device))
    if key in module._MASK_CACHE:
        module._MASK_CACHE.move_to_end(key)
        return module._MASK_CACHE[key]
    mod = isolated_mask_mod(layout, bounds, prefix_tokens, device)
    def mask(batch, head, q, k):
        qv, kv, qf, kf, window, valid_key = mod(batch, head, q, k)
        if anchors in ('columns', 'both'):
            window = window | (kf == 0) | (kf == layout.num_frames - 1)
        if anchors in ('rows', 'both'):
            window = window | (qf == 0) | (qf == layout.num_frames - 1)
        return ((~(qv & kv)) | window) & valid_key
    result = module.create_block_mask(mask, B=None, H=None, Q_LEN=layout.seq_len, KV_LEN=layout.seq_len,
                                      device=device, BLOCK_SIZE=block_size, _compile=True)
    module._MASK_CACHE[key] = result
    while len(module._MASK_CACHE) > module.MAX_CACHED_MASKS:
        module._MASK_CACHE.popitem(last=False)
    return result


def isolated(attn):
    state = getattr(attn, '_ref2va_attention', None)
    return bool(state and state.isolate_padding and state.buckets.active and state.buckets.current.padding)


def window_attention(attn, q, k, v, layout, bounds, scale):
    state = attn._ref2va_attention
    full_cover = state.kernel == 'sol' and all(lo <= 0 and hi >= layout.num_frames-1 for lo, hi in bounds)
    if full_cover:
        state.native_reasons['full_cover'] += 1
    if state.kernel == 'sol' and not full_cover and state.use_sol(attn._ref2va_sol_layer):
        prefix = state.buckets.current.prefix_tokens if isolated(attn) else None
        return state.sol(q, k, v, layout, bounds, scale, anchors=attn.anchor_frames,
                         tau=state.sol_tau, valid_prefix=prefix)
    if isolated(attn):
        from src.models.softmax_attention.flex_attention import window_softmax_flex
        mask = build_isolated_mask(layout, bounds, state.buckets.current.prefix_tokens, v.device, attn.anchor_frames)
        import torch
        if torch.cuda.get_device_capability(v.device)[0] >= 10:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        return window_softmax_flex(q, k, v, mask, scale, inference=attn.inference_mode)
    if state.kernel == 'decomposed':
        from src.models.softmax_attention.decomposed import window_softmax_decomposed
        return window_softmax_decomposed(q, k, v, layout, bounds, scale, anchor_frames=attn.anchor_frames)
    return state.native(attn, q, k, v, layout, bounds, scale)


class AttentionRuntime:
    def __init__(self, hybrids, buckets, native):
        self.buckets, self.native = buckets, native
        self.kernel, self.isolate_padding = 'native', False
        self.sol = None
        self.step = -1
        self.transitions = 0
        for index, attn in enumerate(hybrids):
            attn._ref2va_attention = self
            attn._ref2va_sol_layer = index

    def select(self, settings):
        from src.models.linear_attention import scan
        self.kernel, self.isolate_padding = settings.attention_kernel, settings.isolate_padding
        self.linear_chunk = scan.STATS_CHUNK_FRAMES = settings.linear_stats_chunk_frames
        self.sol_tau, self.dense_steps, self.dense_layers = settings.sol_tau, settings.sol_dense_steps, settings.sol_dense_layers
        self.step = -1
        self.transitions = 0
        self.native_reasons = {'initial_steps': 0, 'initial_layers': 0, 'full_cover': 0}
        self.verification = None
        self.verification_seconds = 0.
        self.verification_first_run = False
        self.sol_before = None
        if self.sol is not None:
            self.sol.reset()
        if self.kernel == 'sol' and self.dense_steps < 8 and self.dense_layers < 50:
            import time
            import torch
            from .sol_attention import WindowSol
            if self.sol is None:
                self.sol = WindowSol()
            device = torch.device('cuda', torch.cuda.current_device())
            first = str(device) not in self.sol.kernel.validated
            self.verification_first_run = first
            started = time.perf_counter()
            self.verification = self.sol.kernel.verify(device)
            self.verification_seconds = time.perf_counter()-started if first else 0.
            self.sol_before = self.sol.kernel.snapshot()
        if self.kernel == 'decomposed':
            from src.models.softmax_attention.decomposed import varlen_kernel
            varlen_kernel()  # Fail explicitly if FA4 varlen is not installed.

    def begin_step(self, step):
        # Clear residuals on the SAME step on every rank, even linear-only ranks.
        changed = (self.kernel == 'sol' and self.dense_layers < 50 and self.step >= 0
                   and (self.step < self.dense_steps) != (step < self.dense_steps))
        self.step = step
        self.transitions += int(changed)
        return changed

    def use_sol(self, layer):
        if self.step < 0:
            raise RuntimeError('Sol attention requires the request step context')
        if self.step < self.dense_steps:
            self.native_reasons['initial_steps'] += 1
            return False
        if layer < self.dense_layers:
            self.native_reasons['initial_layers'] += 1
            return False
        return True

    def signature(self):
        base = (self.kernel, self.isolate_padding, self.linear_chunk)
        if self.kernel != 'sol':
            return base
        prefix = self.buckets.current.prefix_tokens if self.isolate_padding and self.buckets.active else None
        return (*base, self.sol_tau, self.dense_steps, self.dense_layers, prefix)

    def report(self):
        result = {'kernel': self.kernel, 'linear_stats_chunk_frames': self.linear_chunk, 'isolate_padding': self.isolate_padding,
                'padding_method': 'block_mask_key_exclusion' if self.isolate_padding else 'unmasked',
                'score_mod': False}
        if self.kernel == 'sol':
            from .sol_kernel import BACKEND, REVISION
            result['sol'] = {
                'backend': BACKEND, 'source_revision': REVISION, 'tau': self.sol_tau,
                'dense_steps': self.dense_steps, 'dense_layers': self.dense_layers,
                'native_window_calls': dict(self.native_reasons), 'full_cover_path': 'native dense',
                'residual_transition_resets': self.transitions,
                'verification': self.verification, 'verification_seconds': self.verification_seconds,
                'compile': self.sol.kernel.since(self.sol_before) if self.sol_before else {},
                **(self.sol.report() if self.sol else {'sparse_executed': False, 'sparse_kernel_launches': 0})}
            if self.isolate_padding:
                result['padding_method'] = 'exclude_padding_keys_before_sol; native_block_mask_in_dense_steps_layers'
        return result


def summarize_attention(records):
    result = {**records[0], 'by_rank': records}
    if result['kernel'] == 'sol':
        stats = [item['sol'] for item in records]
        result['sol'] = {**result['sol'],
                         'sparse_executed': any(s['sparse_executed'] for s in stats),
                         'sparse_kernel_launches_all_ranks': sum(s['sparse_kernel_launches'] for s in stats),
                         'compile_misses_all_ranks': sum(s['compile'].get('compile_misses', 0) for s in stats),
                         'preprocess_signatures_all_ranks': sum(s['compile'].get('preprocess_signatures', 0) for s in stats),
                         'compile_seconds_max_rank': max(s['compile'].get('compile_seconds', 0.) for s in stats),
                         'preprocess_cold_seconds_max_rank': max(s['compile'].get('preprocess_cold_seconds', 0.) for s in stats)}
    return result
