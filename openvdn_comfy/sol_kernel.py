"""Rectangular host adapter for the pinned NVIDIA Sol-Attn Hopper kernel.

SM90's device kernel already derives Q and K extents independently. The
upstream public wrapper/preprocessor assumes square input. This adapter sizes
K summaries from Tk, Q thresholds/output from Tq; it never expands Q to Tk.
No quantized attention or transport and no silent fallback to another backend.
"""
from collections import OrderedDict
import math
import time

BACKEND = 'nvidia_sol_sm90_rect_v1'
REVISION = 'bb60499af0e675095ff67424196d8c18e265f32a'


def validate_inputs(q, k, v, scale, tau, sink_tokens):
    import torch
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape:
        raise ValueError('Sol expects Q=[B,Tq,H,128], K/V=[B,Tk,H,128]')
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:] or q.shape[-1] != 128:
        raise ValueError('Sol batch/head dimensions must match, head dimension must be 128')
    if min(q.shape[:3]) <= 0 or k.shape[1] <= 0:
        raise ValueError('Sol requires nonempty Q/K/V')
    if any(x.dtype != torch.bfloat16 or x.device != q.device for x in (q, k, v)):
        raise ValueError('Sol requires BF16 Q/K/V on the same device')
    if q.device.type != 'cuda' or tuple(torch.cuda.get_device_capability(q.device)) != (9, 0):
        raise RuntimeError('This Sol adapter requires an SM90 GPU (H100/H200)')
    if any(x.stride(-1) != 1 for x in (q, k, v)):
        raise ValueError('Sol requires contiguous head dimensions')
    if type(sink_tokens) is not int or not 0 <= sink_tokens <= k.shape[1]:
        raise ValueError('Invalid Sol exact sink length')
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(tau):
        raise ValueError('Sol scale/tau must be finite; scale must be positive')


def prepare_rectangular(q, k, v, *, scale, tau):
    import torch
    import triton
    from triton.tools.tensor_descriptor import TensorDescriptor
    from ._vendor.sol_attn import preprocess as p
    from .sol_preprocess import diagonal_threshold
    batch, tq, heads, dim = q.shape
    nk, nq = triton.cdiv(k.shape[1], 64), triton.cdiv(tq, 64)
    kc, vc = p._reduce_kv(k, v)
    mean = torch.empty((batch, heads, dim), device=q.device, dtype=torch.float32)
    variance = torch.empty_like(mean)
    # CuTe's SM90 mainloop assumes every non-inner threshold stride is
    # 16-byte aligned. Uneven head splits (9,10,11) need an explicit storage
    # stride; the logical shape remains H, with no additional attention heads.
    head_stride = (heads+3)//4*4
    threshold = torch.empty((batch, nq, head_stride), device=q.device, dtype=torch.float32)[:, :, :heads]
    kd = TensorDescriptor.from_tensor(kc, [1, 64, 1, 128])
    qd = TensorDescriptor.from_tensor(q, [1, 64, 1, 128])
    p._reduce_kc_stats_kernel[(1, batch*heads)](kd, mean, variance, nk, heads, nk, dim, 128, 64)
    diagonal_threshold[(nq, batch*heads)](
        qd, mean, variance, threshold, scale, tq, heads, nq, head_stride, tau, num_warps=4)
    return kc, vc, threshold


class SolKernel:
    def __init__(self, max_shapes=128):
        self.compiled = OrderedDict()
        self.seen_preprocess = set()
        self.max_shapes = max_shapes
        self.compile_misses = 0
        self.compile_seconds = 0.
        self.preprocess_cold_seconds = 0.
        self.preprocess_misses = 0
        self.validated = {}

    @staticmethod
    def dependencies():
        try:
            import cutlass.cute as cute
            import cuda.bindings.driver as cuda
            from ._vendor.sol_attn.sm90 import make_kernel
            from ._vendor.sol_attn.common import to_cute_tensor
        except (ImportError, OSError) as error:
            raise RuntimeError('Sol SM90 dependencies unavailable; run bash deploy.sh install-sol') from error
        return cute, cuda, make_kernel, to_cute_tensor

    def __call__(self, q, k, v, *, scale, tau, sink_tokens=0, all_exact=False):
        import torch
        validate_inputs(q, k, v, scale, tau, sink_tokens)
        cute, cuda, make_kernel, convert = self.dependencies()
        with torch.cuda.device(q.device):
            key = (q.device.index, *(tuple(x.shape) for x in (q, k, v)),
                   *(tuple(x.stride()) for x in (q, k, v)))
            preprocess_key = (key, float(tau))
            cold = preprocess_key not in self.seen_preprocess
            if cold:
                torch.cuda.synchronize(q.device)
            started = time.perf_counter() if cold else None
            kc, vc, threshold = prepare_rectangular(q, k, v, scale=scale, tau=tau)
            if cold:
                torch.cuda.synchronize(q.device)
                self.preprocess_cold_seconds += time.perf_counter()-started
                self.preprocess_misses += 1
                self.seen_preprocess.add(preprocess_key)
            if all_exact:
                threshold.fill_(-float('inf'))
            out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
            lse = torch.empty(q.shape[:3], device=q.device, dtype=torch.float32)
            tensors = [q, k, v, out, kc, vc, threshold, lse]
            args = [convert(x) for x in tensors]
            # Descriptors include strides, batch and local head count. Physical
            # K length controls the SM90 tail recipe; Q length controls its grid.
            stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
            sink_end = (sink_tokens+63)//64
            if sink_end >= 65536:
                raise ValueError('Sol sink exceeds the SM90 packed sink range')
            sink_range = sink_end << 16
            compiled = self.compiled.get(key)
            if compiled is None:
                started = time.perf_counter()
                operator = make_kernel(k.shape[1], 1)
                # The square wrapper infers full Q tiles from K length. For a
                # rectangular final Q tile use the kernel's existing guarded path.
                if q.shape[1] % 64:
                    operator.sol_attn_assume_lane_group_route_reduce = False
                from ._vendor.sol_attn.sm90._compat.cute_dsl_utils import converter_compatibility
                with converter_compatibility():
                    compiled = cute.compile(operator, *args, float(scale), sink_range,
                                            stream=stream, options='--enable-tvm-ffi')
                self.compile_seconds += time.perf_counter()-started
                self.compile_misses += 1
                self.compiled[key] = compiled
                while len(self.compiled) > self.max_shapes:
                    self.compiled.popitem(last=False)
            else:
                self.compiled.move_to_end(key)
            compiled(*args, float(scale), sink_range, stream=stream)
            return out

    def snapshot(self):
        return dict(compile_misses=self.compile_misses, compile_seconds=self.compile_seconds,
                    preprocess_signatures=self.preprocess_misses,
                    preprocess_cold_seconds=self.preprocess_cold_seconds)

    def since(self, before):
        return {name: value-before[name] for name, value in self.snapshot().items()}

    def verify(self, device):
        """Once/device, including real rectangular tails and sparse correction.

        This is an arithmetic check, not a quality-equivalence claim. It runs on
        first opt-in, never during the default native startup warmup.
        """
        import torch
        key = str(device)
        if key in self.validated:
            return self.validated[key]
        self.dependencies()
        from .sol_reference import sol_reference
        generator = torch.Generator(device=device).manual_seed(7719)
        results = []
        for tq, tk, heads, sink in ((73, 197, 2, 65), (131, 65, 1, 0), (64, 128, 1, 128), (73, 4096, 1, 0)):
            q = torch.randn((2, tq, heads, 128), generator=generator, device=device, dtype=torch.bfloat16)
            k = torch.randn((2, tk, heads, 128), generator=generator, device=device, dtype=torch.bfloat16)
            v = torch.randn(k.shape, generator=generator, device=device, dtype=torch.bfloat16)
            for exact in (True, False):
                out = self(q, k, v, scale=128**-.5, tau=1., sink_tokens=sink, all_exact=exact)
                ref = sol_reference(q, k, v, scale=128**-.5, tau=1., sink_tokens=sink, all_exact=exact)
                if not torch.isfinite(out).all():
                    raise RuntimeError('Sol SM90 arithmetic check produced nonfinite output')
                error = (out.float()-ref).abs()
                relative = float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(ref).clamp_min(1e-8))
                maximum = float(error.max())
                # BF16 centroid/value-sum and probability products introduce
                # rounding. Check rectangular routing and tails before generation.
                if relative > .025 or maximum > .08:
                    raise RuntimeError(f'Sol SM90 arithmetic check failed: relative={relative}, max={maximum}')
                results.append(dict(tq=tq, tk=tk, heads=heads, sink_tokens=sink,
                                    all_exact=exact, relative_l2=relative, max_abs=maximum))
        self.validated[key] = {'passed': True, 'cases': results}
        return self.validated[key]
