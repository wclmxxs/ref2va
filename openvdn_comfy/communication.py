"""Audited Ulysses pack/unpack replacement, switchable between requests."""
import hashlib
import inspect
import types


def pack_fields(q, k, v, gate, shared, out, heads, parts, linear):
    from .communication_kernels import pack_groups
    rows, _, dim = v.shape
    shared_width = shared.shape[-1] if linear else 0
    max_width = ((heads + parts - 1) // parts) * (3 * dim + 1) + shared_width
    pack_groups[(rows, (max_width + 255) // 256, parts)](
        q, k, v, gate, shared, out, rows, heads, dim, parts, shared_width, 256, num_warps=4)


def unpack_fields(recv, rows, heads, dim, soft_parts, linear_parts):
    from .communication_kernels import unpack_groups
    out = recv.new_empty((2, rows, heads, dim))
    unpack_groups[((out.numel() + 255) // 256,)](
        recv, out, rows, heads, dim, soft_parts, linear_parts, 256, num_warps=4)
    return out[0], out[1]


def rewritten_methods(runtime_type):
    """Keep NCCL ordering, split sizes, pending-work lifetimes and timing unchanged."""
    result = {}
    for name, expected in SOURCE_HASHES.items():
        original = getattr(runtime_type, name)
        source = inspect.getsource(original)
        if hashlib.sha256(source.encode()).hexdigest() != expected:
            raise RuntimeError(f'Ulysses source changed: {name}; audit before enabling fused communication')
        import textwrap
        source = textwrap.dedent(source)
        if name == 'dispatch_fields_to_branches_overlapped':
            begin = source.index('    offset = 0\n')
            end = source.index('    soft_row_width =', begin)
            source = source[:begin] + '''    _pack(softmax_q, softmax_k, value, softmax_gate, linear_shared,
          soft_send, self.num_heads, self.softmax_ranks, False)
''' + source[end:]
            begin = source.index('    offset = 0\n')
            end = source.index('    self.profile_end("branch_pack"', begin)
            source = source[:begin] + '''    _pack(linear_q, linear_k, value, linear_beta, linear_shared,
          linear_send, self.num_heads, self.world_size - self.softmax_ranks, True)
''' + source[end:]
        else:
            begin = source.index('    chunks = []\n')
            end = source.index('    self.profile_end("output_unpack"', begin)
            source = source[:begin] + '''    softmax, linear = _unpack(recv, local_rows, self.num_heads, width,
                               self.softmax_ranks, self.world_size - self.softmax_ranks)
''' + source[end:]
        ns = {**original.__globals__, '_pack': pack_fields, '_unpack': unpack_fields}
        exec(compile(source, f'<ref2va_communication/{name}>', 'exec'), ns)
        result[name] = ns[name]
    return result


class CommunicationRuntime:
    def __init__(self, runtime):
        self.runtime = runtime
        self.native = {name: getattr(runtime, name) for name in SOURCE_HASHES}
        self.fast = {name: types.MethodType(fn, runtime)
                     for name, fn in rewritten_methods(type(runtime)).items()}
        self.active = False
        self.parity = None

    def select(self, enabled):
        self.active = bool(enabled)
        for name, method in (self.fast if self.active else self.native).items():
            setattr(self.runtime, name, method)

    def verify(self, layouts=None, head_dim=96):
        """Small deterministic CUDA check on every rank before enabling the path."""
        import torch
        import torch.distributed as dist
        from src.inference.utils.ulysses_runtime import _pack_branch_target_kernel
        device = self.runtime.device
        gen = torch.Generator(device=device).manual_seed(914 + self.runtime.rank)
        checks = []
        layouts = layouts or (self.runtime.softmax_ranks or min(6, self.runtime.world_size - 1),)
        for parts in layouts:
            rows, heads, dim = 19, 56, head_dim
            q, k, v = [torch.randn(rows, heads, dim, generator=gen, device=device,
                                   dtype=torch.bfloat16) for _ in range(3)]
            gate = torch.randn(rows, heads, 1, generator=gen, device=device, dtype=q.dtype)
            shared = torch.randn(rows, 32, generator=gen, device=device, dtype=q.dtype)
            for linear in (False, True):
                shared_width = 32 if linear else 0
                actual = v.new_empty(rows * (heads * (3 * dim + 1) + parts * shared_width))
                expected = torch.empty_like(actual)
                first = offset = 0
                for target in range(parts):
                    count = heads // parts + (target < heads % parts)
                    width = count * (3 * dim + 1) + shared_width
                    size = rows * width
                    _pack_branch_target_kernel[(rows, (width + 255) // 256)](
                        q, k, v, gate, shared, expected[offset:offset + size], rows,
                        NUM_HEADS=heads, HEAD_DIM=dim, FIRST_HEAD=first, LOCAL_HEADS=count,
                        SHARED_WIDTH=32, LINEAR=linear, BLOCK=256, num_warps=4)
                    offset += size; first += count
                pack_fields(q, k, v, gate, shared, actual, heads, parts, linear)
                checks.append(torch.eq(actual, expected).all())
            splits = [heads // parts + (i < heads % parts) for i in range(parts)]
            linear_parts = 8 - parts
            splits += [heads // linear_parts + (i < heads % linear_parts) for i in range(linear_parts)]
            chunks = [torch.randn(rows, h, dim, generator=gen, device=device, dtype=q.dtype) for h in splits]
            restored = unpack_fields(torch.cat([x.flatten() for x in chunks]), rows, heads, dim, parts, linear_parts)
            checks.extend(torch.eq(a, b).all() for a, b in zip(restored,
                           (torch.cat(chunks[:parts], 1), torch.cat(chunks[parts:], 1))))
        # Compare actual uneven NCCL dispatch and return, not just local packing.
        runtime = self.runtime
        saved = {key: getattr(runtime, key) for key in (
            'sequence_length','splits','local_start','local_end','heads_per_rank','num_heads',
            'softmax_ranks','softmax_head_splits','linear_head_splits')}
        try:
            for parts in layouts:
                runtime.sequence_length = 0
                runtime.softmax_ranks = parts
                runtime.configure(137, 56)  # deliberately uneven sequence shards
                rows = runtime.splits[runtime.rank]
                fields = [torch.randn(rows,56,head_dim,generator=gen,device=device,dtype=torch.bfloat16)
                          for _ in range(5)]
                gates = [torch.randn(rows,56,1,generator=gen,device=device,dtype=torch.bfloat16) for _ in range(2)]
                shared = torch.randn(rows,32,generator=gen,device=device,dtype=torch.bfloat16)
                args = (*fields[:3],gates[0],*fields[3:],gates[1],shared)
                reference, expected_shared, pending = self.native['dispatch_fields_to_branches_overlapped'](*args)
                pending[0].wait()
                actual, actual_shared, pending = self.fast['dispatch_fields_to_branches_overlapped'](*args)
                pending[0].wait()
                checks.append(torch.eq(actual,reference).all())
                if expected_shared is not None:
                    checks.append(torch.eq(actual_shared,expected_shared).all())
                branch = torch.randn(137,runtime.branch_heads,head_dim,generator=gen,device=device,dtype=torch.bfloat16)
                expected = self.native['branches_to_sequence'](branch)
                actual = self.fast['branches_to_sequence'](branch)
                checks.extend(torch.eq(a,b).all() for a,b in zip(actual,expected))
        finally:
            for key,value in saved.items():
                setattr(runtime,key,value)
        ok = torch.stack(checks).all().to(torch.int32)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if not bool(ok):
            raise RuntimeError('Fused communication parity failed')
        self.parity = {'checked': True, 'exact': True, 'layouts': list(layouts), 'all_ranks': True, 'uneven_nccl_transport': True}
        return self.parity


SOURCE_HASHES = {'dispatch_fields_to_branches_overlapped': '1e5e414957bd97e2b9b70b368ed042bce638293df29fc95652b4159bd182040d', 'branches_to_sequence': 'cddbdfddfce15f26abd714c13f077c5a8b1a76f524e132caecc2c3057a3c3a3f'}
