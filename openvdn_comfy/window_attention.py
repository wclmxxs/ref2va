"""Copy optimizations for the native decomposed mask; no sparsity change."""
from contextlib import nullcontext
from functools import lru_cache


@lru_cache(maxsize=128)
def query_slice(video_start, video_end, num_frames, tokens_per_frame, bounds, anchor_frames):
    # The pinned native plan groups only adjacent frames and preserves Q order.
    # Cache this metadata; do not allocate a Python list on every block/NFE.
    anchor_rows = anchor_frames in ('rows', 'both')
    start = video_start + (tokens_per_frame if anchor_rows else 0)
    stop = video_end - (tokens_per_frame if anchor_rows else 0)
    return start, max(start, stop)


def window_softmax_fast(query, key, value, layout, bounds, scale, anchor_frames='none', profiler=None):
    import torch
    from src.models.softmax_attention import decomposed as native
    stage = profiler.stage if profiler else lambda name: nullcontext()
    varlen = native.varlen_kernel()
    with stage('softmax_plan'):
        plan = native._plan(layout, bounds, anchor_frames, query.device)
    with stage('softmax_key_contiguous'):
        key = key.contiguous()
    with stage('softmax_value_contiguous'):
        value = value.contiguous()
    with stage('softmax_output_allocate'):
        out = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    if len(plan.dense_q):
        with stage('softmax_dense_q_gather'):
            qd = torch.index_select(query, 0, plan.dense_q)
        with stage('softmax_dense_attention'), native.sdpa_kernel(native._dense_backends()):
            od = native.scaled_dot_product_attention(qd.transpose(0,1).unsqueeze(0),
                key.transpose(0,1).unsqueeze(0), value.transpose(0,1).unsqueeze(0), scale=scale)
        with stage('softmax_dense_scatter'):
            out.index_copy_(0, plan.dense_q, od[0].transpose(0,1))
    if plan.has_windows:
        with stage('softmax_window_k_gather'):
            kw = torch.index_select(key, 0, plan.kv_gather)
        with stage('softmax_window_v_gather'):
            vw = torch.index_select(value, 0, plan.kv_gather)
        start, stop = query_slice(layout.video_start, layout.video_end, layout.num_frames,
                                 layout.tokens_per_frame, tuple(bounds), anchor_frames)
        with stage('softmax_window_q_gather'):
            # FA4 requires a contiguous operand even when a sequence view is
            # logically correct (a head shard can still have a strided row).
            qw = query[start:stop].contiguous()
        with stage('softmax_window_attention'):
            ow = varlen(qw, kw, vw, plan.cu_q, plan.cu_k, plan.max_q, plan.max_k, scale)
        with stage('softmax_window_scatter'):
            out[start:stop].copy_(ow)
    return out
