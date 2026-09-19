"""Request-scoped full-information linear attention acceleration.

The CUDA factorization and chunk reassociation preserve the delta equations, but
not bitwise FP32 reduction order. Native factors/scans remain available separately.
"""
from contextlib import contextmanager, nullcontext
from functools import lru_cache
import inspect
import types

from .delta_kernel import DeltaKernel
from .fine_profile import source, instrument, rule


@lru_cache(maxsize=128)
def boundary_compatible(frames, chunk, offset, bounds):
    if chunk <= 1 or offset not in (0, 1):
        return False
    count = (frames+offset+chunk-1)//chunk
    ends = {min((c+1)*chunk-1,frames+offset-1)-offset for c in range(count)}
    starts = {c*chunk-offset for c in range(count)}
    return all((not 0 <= lo-1 < frames or lo-1 in ends) and
               (not 0 <= hi+1 < frames or hi+1 in starts) for lo,hi in bounds)


class LinearAcceleration:
    def __init__(self, hybrids, linear_kv, runtime, scan):
        from .boundary_scan import run_boundary_scans
        self.runtime, self.linear_kv, self.native_scans = runtime, linear_kv, scan._run_scans_inference
        self.kernel, self.profiler = DeltaKernel(), None
        self.active = False
        self.fused_delta = self.boundary_scan = False
        self.calls = {}
        self.scan_verification = {'checked': False}
        self.methods = {}
        self.text_methods = {}
        for attn in hybrids:
            attn.linear_attention._ref2va_linear_engine = self
            # Never attach the parent nn.Module to its child (would create a
            # registered-module cycle). These checkpoint properties are static.
            attn.linear_attention._ref2va_scan_geometry = (attn.chunk, int(attn.anchor_frames == 'both'))
        for branch, native, sampled in linear_kv.bindings:
            fn = branch._text_chunk_state.__func__
            if fn not in self.text_methods:
                text = source(fn, '_text_chunk_state')
                old = 'backend.factor_apply(ones, A, B)'
                if text.count(old) != 1:
                    raise RuntimeError('Pinned text delta boundary changed')
                text = text.replace(old, '_ref2va_linear_factor(backend, ones, A, B)')
                self.text_methods[fn] = instrument(fn, text, {}, {'_ref2va_linear_factor': self.factor})
            for fn in (native.__func__, sampled.__func__):
                if fn in self.methods:
                    continue
                text = source(fn, '_readout_inference') if fn is native.__func__ else inspect.getsource(fn)
                old = 'backend, alpha, A, B, text_state=text_state)'
                if text.count(old) != 1:
                    raise RuntimeError('Linear acceleration readout boundary changed')
                text = text.replace(old, '''backend, alpha, A, B, text_state=text_state, bounds=bounds,
            chunk=self._ref2va_scan_geometry[0],
            frame_offset=self._ref2va_scan_geometry[1])''')
                clone = instrument(fn, text, {}, {'_run_scans_inference': self.scans})
                clone._ref2va_accelerated = True
                self.methods[fn] = clone
        self.boundaries = run_boundary_scans
        self.profiled_boundaries = instrument(run_boundary_scans, inspect.getsource(run_boundary_scans), {
            'assign:fwd_t,fwd_b': rule('linear_chunk_compose_forward'),
            'assign:rev_t,rev_b': rule('linear_chunk_compose_reverse'),
            'for:range(num_chunks)': rule('linear_chunk_scan'),
        }, {'_ref2va_fine': self})

    def stage(self, name):
        return self.profiler.stage(name) if self.runtime.profile_enabled and self.profiler else nullcontext()

    def factor(self, backend, alpha, A, B):
        if self.fused_delta and type(backend).__name__ in ('VdnDelta', 'VdnScaledDelta'):
            with self.stage('linear_fused_delta'):
                scale = backend.inv_tokens if type(backend).__name__ == 'VdnScaledDelta' else 1.
                a = A.float() * scale if scale != 1 else A.float()
                b = B.float() * scale**.5 if scale != 1 else B.float()
                result = self.kernel(a, b, alpha.float())
                self.calls['fused_delta'] = self.calls.get('fused_delta',0)+1
                return result[0].to(A.dtype), result[1].to(B.dtype)
        if self.runtime.profile_enabled and self.profiler:
            return self.profiler.factor_apply(backend, alpha, A, B)
        return backend.factor_apply(alpha, A, B)

    def scans(self, backend, alpha, A, B, text_state=None, *, bounds, chunk, frame_offset):
        import torch
        use_boundary = self.boundary_scan and boundary_compatible(A.shape[0], chunk, frame_offset, tuple(bounds))
        with torch.autocast(device_type=A.device.type, enabled=False):
            if use_boundary:
                with self.stage('linear_factor_apply'):
                    transition, injection = self.factor(backend, alpha, A, B)
                fn = self.profiled_boundaries if self.runtime.profile_enabled else self.boundaries
                self.calls['boundary_scan'] = self.calls.get('boundary_scan',0)+1
                return fn(transition, injection, text_state, chunk=chunk, frame_offset=frame_offset)
            # Reuse the native sequential scan, only substitute the factor operation.
            proxy = types.SimpleNamespace(factor_apply=lambda *args: self.factor(backend, *args))
            self.calls['native_scan'] = self.calls.get('native_scan',0)+1
            if self.runtime.profile_enabled and self.profiler:
                return self.profiler.native_scans_with_factor(proxy, alpha, A, B, text_state=text_state)
            return self.native_scans(proxy, alpha, A, B, text_state=text_state)

    def select(self, settings, device):
        if self.active:
            raise RuntimeError('Cannot change linear acceleration during inference')
        self.fused_delta, self.boundary_scan = settings.fused_delta, settings.boundary_scan
        if self.fused_delta:
            self.kernel.verify(device)
        if self.boundary_scan and not self.scan_verification['checked']:
            self.verify_scans(device)

    def verify_scans(self, device):
        import torch
        generator = torch.Generator(device=device).manual_seed(921)
        frames, heads, dim = 13, 2, 8
        transition = torch.eye(dim, device=device)[None, None].expand(frames, heads, dim, dim).clone()*.8
        transition += torch.randn(transition.shape, device=device, generator=generator)*.01
        injection = torch.randn(transition.shape, device=device, generator=generator)*.1
        text = torch.randn(heads, dim, dim, device=device, generator=generator)
        proxy = types.SimpleNamespace(factor_apply=lambda *args: (transition, injection))
        reference = self.native_scans(proxy, None, transition, injection, text_state=text)
        from .boundary_scan import _boundary_frames
        errors = []
        for offset in (0, 1):
            result = self.boundaries(transition, injection, text, chunk=5, frame_offset=offset)
            _, ends, _, starts, _ = _boundary_frames(frames, 5, offset, str(device))
            for actual, expected, ids in zip(result, reference, (ends, starts)):
                a, b = actual.index_select(0, ids), expected.index_select(0, ids)
                error = float((a-b).norm()/b.norm().clamp_min(1e-20))
                if not bool(torch.isfinite(a).all()) or error > 2e-5:
                    raise RuntimeError(f'Boundary scan arithmetic check failed: relative={error}')
                errors.append(error)
        self.scan_verification = {'checked': True, 'passed': True, 'relative_errors': errors}

    @contextmanager
    def request(self):
        if self.active:
            raise RuntimeError('Linear acceleration requests cannot overlap')
        self.active, self.calls = True, {}
        saved = []
        try:
            if self.fused_delta or self.boundary_scan:
                for branch, _, _ in self.linear_kv.bindings:
                    original = branch._readout_inference
                    saved.append((branch, '_readout_inference', original))
                    branch._readout_inference = types.MethodType(self.methods[original.__func__], branch)
                    if self.fused_delta:
                        original = branch._text_chunk_state
                        saved.append((branch, '_text_chunk_state', original))
                        branch._text_chunk_state = types.MethodType(self.text_methods[original.__func__], branch)
            yield self
        finally:
            for branch, attr, original in reversed(saved):
                setattr(branch, attr, original)
            self.active = False

    def report(self):
        return {'rank': self.runtime.rank, 'fused_delta': self.fused_delta, 'boundary_scan': self.boundary_scan,
                'calls':dict(self.calls), 'kernel_verification':self.kernel.verification,
                'scan_verification':self.scan_verification,
                'retains_all_frames_and_tokens':True, 'bitwise_equivalent':False}
