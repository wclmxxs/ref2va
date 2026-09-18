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
        for attn in hybrids:
            attn._ref2va_attention = self

    def select(self, settings):
        from src.models.linear_attention import scan
        self.kernel, self.isolate_padding = settings.attention_kernel, settings.isolate_padding
        self.linear_chunk = scan.STATS_CHUNK_FRAMES = settings.linear_stats_chunk_frames
        if self.kernel == 'decomposed':
            from src.models.softmax_attention.decomposed import varlen_kernel
            varlen_kernel()  # Fail explicitly if FA4 varlen is not installed.

    def signature(self):
        return self.kernel, self.isolate_padding, self.linear_chunk

    def report(self):
        return {'kernel': self.kernel, 'linear_stats_chunk_frames': self.linear_chunk, 'isolate_padding': self.isolate_padding,
                'padding_method': 'block_mask_key_exclusion' if self.isolate_padding else 'unmasked',
                'score_mod': False}


def summarize_attention(records):
    return {**records[0], 'by_rank': records}
