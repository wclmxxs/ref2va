# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang 5e9342d16f03621f8f434baca2bc4bbdfa4800c7.
# Attribution/license: _vendor/sglang_vdn/NOTICE.md and LICENSE.
from __future__ import annotations
import functools
import torch

def _compose_chunk(
    transitions: torch.Tensor, injections: torch.Tensor, reverse: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    # fold each chunk's frames into one affine map S -> S @ M + C, batched over chunks
    order = list(range(transitions.shape[0]))
    if reverse:
        order.reverse()
    chunks, heads = transitions.shape[1], transitions.shape[2]
    dk, dv = transitions.shape[-1], injections.shape[-2]
    folded_t = transitions[order[0]]
    folded_b = injections[order[0]]
    for j in order[1:]:
        step_t = transitions[j].view(chunks * heads, dk, dk)
        folded_b = torch.baddbmm(
            injections[j].view(chunks * heads, dv, dk),
            folded_b.view(chunks * heads, dv, dk),
            step_t,
        ).view(chunks, heads, dv, dk)
        folded_t = torch.bmm(folded_t.view(chunks * heads, dk, dk), step_t).view(
            chunks, heads, dk, dk
        )
    return folded_t, folded_b

@functools.lru_cache(maxsize=64)
def _boundary_frames(
    num_frames: int, chunk: int, frame_offset: int, device: str
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # the gather reads prefix at chunk ends and suffix at chunk starts, on the offset grid
    padded = frame_offset + num_frames
    num_chunks = -(-padded // chunk)
    ends = [
        min((c + 1) * chunk - 1, padded - 1) - frame_offset for c in range(num_chunks)
    ]
    starts = [c * chunk - frame_offset for c in range(num_chunks)]
    dev = torch.device(device)
    ends = [(f, c) for c, f in enumerate(ends) if f >= 0]
    starts = [(f, c) for c, f in enumerate(starts) if f >= 0]
    return (
        num_chunks,
        torch.tensor([f for f, _ in ends], device=dev, dtype=torch.int64),
        torch.tensor([c for _, c in ends], device=dev, dtype=torch.int64),
        torch.tensor([f for f, _ in starts], device=dev, dtype=torch.int64),
        torch.tensor([c for _, c in starts], device=dev, dtype=torch.int64),
    )

def run_boundary_scans(
    transitions: torch.Tensor,
    injections: torch.Tensor,
    text_state: torch.Tensor | None,
    *,
    chunk: int,
    frame_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``run_scans`` restricted to what the chunked gather reads: prefix at each
    chunk's last frame, suffix at each chunk's first frame, zero elsewhere.
    ``frame_offset`` is frame 0's position on the chunk grid (1 when the anchor
    frames were dropped). Same fp32 math, the products re-associated."""
    if chunk <= 1:
        raise ValueError("boundary scan requires chunk > 1")
    num_frames, heads, dv, dk = injections.shape
    num_chunks, ends, end_chunks, starts, start_chunks = _boundary_frames(
        num_frames, chunk, frame_offset, str(injections.device)
    )
    # identity / zero padding fills the leading offset and the partial last chunk
    lead, tail = frame_offset, num_chunks * chunk - frame_offset - num_frames
    eye = torch.eye(dk, device=transitions.device, dtype=transitions.dtype)
    transitions = torch.cat(
        [eye.expand(lead, heads, dk, dk), transitions, eye.expand(tail, heads, dk, dk)]
    )
    injections = torch.cat(
        [
            injections.new_zeros(lead, heads, dv, dk),
            injections,
            injections.new_zeros(tail, heads, dv, dk),
        ]
    )
    # frame-major so each composition step reads contiguous operands
    by_frame_t = (
        transitions.view(num_chunks, chunk, heads, dk, dk).transpose(0, 1).contiguous()
    )
    by_frame_b = (
        injections.view(num_chunks, chunk, heads, dv, dk).transpose(0, 1).contiguous()
    )
    start = (
        torch.zeros(heads, dv, dk, dtype=injections.dtype, device=injections.device)
        if text_state is None
        else text_state.to(injections.dtype)
    )
    # step c: the forward chain on chunk c and the reverse chain on chunk C-1-c
    fwd_t, fwd_b = _compose_chunk(by_frame_t, by_frame_b, reverse=False)
    rev_t, rev_b = _compose_chunk(by_frame_t, by_frame_b, reverse=True)
    chunk_t = torch.stack([fwd_t, rev_t.flip(0)], dim=1)  # [C, 2, H, dk, dk]
    boundary = torch.stack([fwd_b, rev_b.flip(0)], dim=1)  # [C, 2, H, dv, dk]
    flat = boundary.view(num_chunks, 2 * heads, dv, dk)
    state = torch.stack([start, start], dim=0).view(2 * heads, dv, dk)
    for c in range(num_chunks):
        flat[c].baddbmm_(state, chunk_t[c].view(2 * heads, dk, dk))
        state = flat[c]
    prefix = torch.zeros(
        num_frames, heads, dv, dk, dtype=injections.dtype, device=injections.device
    )
    suffix = torch.zeros_like(prefix)
    # step c holds chunk c's forward state and chunk C-1-c's reverse state
    prefix.index_copy_(0, ends, boundary[end_chunks, 0])
    suffix.index_copy_(0, starts, boundary[num_chunks - 1 - start_chunks, 1])
    return prefix, suffix
