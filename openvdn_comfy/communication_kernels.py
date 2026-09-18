"""Layout-only Triton kernels. No quantization, reductions, or dropped payloads."""
import triton
import triton.language as tl


@triton.jit
def pack_groups(Q, K, V, G, SHARED, OUT, ROWS: tl.constexpr,
                HEADS: tl.constexpr, DIM: tl.constexpr, PARTS: tl.constexpr,
                SHARED_WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row, target = tl.program_id(0), tl.program_id(2)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    base, extra = HEADS // PARTS, HEADS % PARTS
    local_heads = base + (target < extra)
    first = target * base + tl.minimum(target, extra)
    unit = 3 * DIM + 1
    payload = local_heads * unit
    width = payload + SHARED_WIDTH
    valid = col < width
    head_col = col < payload
    head = first + col // unit
    field = col % unit
    vector = row * HEADS * DIM + head * DIM
    # Match the pinned packer's fp32 conversions/addition order exactly.
    value = tl.zeros((BLOCK,), tl.float32)
    value += tl.load(Q + vector + field, valid & head_col & (field < DIM), other=0).to(tl.float32)
    value += tl.load(K + vector + field - DIM,
                     valid & head_col & (field >= DIM) & (field < 2 * DIM), other=0).to(tl.float32)
    value += tl.load(V + vector + field - 2 * DIM,
                     valid & head_col & (field >= 2 * DIM) & (field < 3 * DIM), other=0).to(tl.float32)
    value += tl.load(G + row * HEADS + head,
                     valid & head_col & (field == 3 * DIM), other=0).to(tl.float32)
    if SHARED_WIDTH:
        value += tl.load(SHARED + row * SHARED_WIDTH + col - payload,
                         valid & ~head_col, other=0).to(tl.float32)
    offset = ROWS * (first * unit + target * SHARED_WIDTH)
    tl.store(OUT + offset + row * width + col, value, valid)


@triton.jit
def unpack_groups(IN, OUT, ROWS: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr,
                  SOFT_PARTS: tl.constexpr, LINEAR_PARTS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < 2 * ROWS * HEADS * DIM
    branch = index // (ROWS * HEADS * DIM)
    row = (index // (HEADS * DIM)) % ROWS
    head = (index // DIM) % HEADS
    dim = index % DIM
    parts = tl.where(branch == 0, SOFT_PARTS, LINEAR_PARTS)
    base, extra = HEADS // parts, HEADS % parts
    large = (base + 1) * extra
    target = tl.where(head < large, head // (base + 1), extra + (head - large) // base)
    first = target * base + tl.minimum(target, extra)
    local_heads = base + (target < extra)
    source = ROWS * (branch * HEADS + first) * DIM + (row * local_heads + head - first) * DIM + dim
    tl.store(OUT + index, tl.load(IN + source, valid, other=0), valid)
