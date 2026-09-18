"""Sol diagonal threshold with aligned storage for uneven Ulysses head shards.

Adapted from NVIDIA Sol-H3 preprocess.py at the revision in sol_kernel.py
(Apache-2.0). Same equation; the output head stride is explicit instead of H.
Upstream's 56 heads are aligned, but 6+2/5+3 splits produce 9/10/11 heads.
"""
import triton
import triton.language as tl


@triton.jit
def diagonal_threshold(q_desc, mean, variance, threshold, scale, tq,
                       H: tl.constexpr, NQ: tl.constexpr, HS: tl.constexpr, TAU: tl.constexpr):
    block, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // H, bh % H
    values = q_desc.load([batch, block*64, head, 0]).reshape([64, 128]).to(tl.float32)
    qb = tl.sum(values, axis=0) / tl.minimum(64, tq-block*64).to(tl.float32)
    dim = tl.arange(0, 128)
    mu = tl.load(mean + bh*128 + dim)
    var = tl.load(variance + bh*128 + dim)
    log2_scale = scale * 1.4426950408889634
    center = tl.sum(qb*mu, axis=0) * log2_scale
    spread = tl.sum(qb*qb*var, axis=0) * log2_scale*log2_scale
    value = center + TAU*tl.sqrt(tl.maximum(spread, 0.) + 1e-6)
    tl.store(threshold + (batch*NQ + block)*HS + head, value)
