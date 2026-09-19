"""Request-only profiling of pinned eager boundaries; no changes to tensor math.

CUDA events measure compute-stream spans, not pure kernel busy time. Optional
Kineto collects actual device events on one representative rank of each branch.
Unprofiled requests keep the original methods and compiler callables untouched.
"""
import ast
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import inspect
import linecache
import sys
import textwrap
import time
import types

SOURCE_HASHES = {
    'window_softmax_decomposed': '1e7c577ced86ec313a229caf94850faededd4993d6948796a0ec7943afbe6325',
    '_readout_inference': 'bced583d3b29059e2b82a6c462f94a1439dcc9a58f9c088410e5bc1e89de734f',
    '_text_chunk_state': '62b5ebf34af88a66bf97c2cc99a901bf6d6ecc05655508f51c38427c62f0c5f2',
    '_feature_one': '891e0115f1c639099edfcb6949cfc2c95a4eebbc01b9cbad6dec2436c411ba7f',
    'prepare_linear_features_inference': '26a2800e37e85f0db57a4f9986022bcf9ad5595cc0878fd8874157ad1b27c4ba',
    '_run_scans_inference': '98b0fc2e2276374b89b4c60f9d22448aeb8689297a905cac4dcc3dbb5188f945',
    'SanaDelta.factor_apply': '9e0a3c5e220eaea2715f46cc18f930420a98afffa74b8acf274941cbb1d2c5d4',
    'VdnDelta.factor_apply': 'f25df5fb4c9d2743a0660ef162bebc09ddff0418c56363ca020a1667c7b280ce',
    'VdnScaledDelta.factor_apply': 'c6eb84a8c21d611a1fcd2ec657322053ca43585048a65cc38ef80b7e1091183c',
}


def source(function, contract):
    text = inspect.getsource(function)
    if hashlib.sha256(text.encode()).hexdigest() != SOURCE_HASHES[contract]:
        raise RuntimeError(f'Pinned fine-profile source changed: {contract}')
    return textwrap.dedent(text)


def target_name(node):
    if isinstance(node, ast.Tuple):
        return ','.join(target_name(x) for x in node.elts)
    return ast.unparse(node)


def call_name(node):
    if isinstance(node, ast.Call):
        return call_name(node.func)
    return ast.unparse(node)


def instrument(function, text, rules, extra):
    """Wrap specified statements, retaining their order, operands and contexts.

    Keys match assignment targets, loop iterators or return/with calls. Every
    audited boundary must occur exactly the expected number of times.
    """
    counts = defaultdict(int)

    class Wrap(ast.NodeTransformer):
        def visit(self, node):
            node = super().visit(node)
            candidates = []
            if isinstance(node, ast.Assign):
                key = 'assign:' + target_name(node.targets[0])
                candidates = [key + '@' + call_name(node.value), key]
            elif isinstance(node, ast.For):
                candidates = ['for:' + ast.unparse(node.iter)]
            elif isinstance(node, ast.Return):
                candidates = ['return:' + call_name(node.value)]
            elif isinstance(node, ast.With):
                candidates = ['with:' + call_name(node.items[0].context_expr)]
            elif isinstance(node, ast.If):
                candidates = ['if:' + ast.unparse(node.test)]
            key = next((k for k in candidates if k in rules), None)
            if key is None:
                return node
            expression, _ = rules[key]
            counts[key] += 1
            context = ast.parse(f'_ref2va_fine.stage({expression})', mode='eval').body
            return ast.copy_location(ast.With(items=[ast.withitem(context_expr=context)], body=[node]), node)

    tree = Wrap().visit(ast.parse(text))
    if dict(counts) != {k:v[1] for k,v in rules.items()}:
        raise RuntimeError(f'Unexpected fine-profile boundaries: {function.__name__}: {dict(counts)}')
    text = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    path = f'<ref2va-fine/{function.__qualname__}-{hashlib.sha256(text.encode()).hexdigest()[:12]}.py>'
    linecache.cache[path] = (len(text), None, text.splitlines(keepends=True), path)
    namespace = {**function.__globals__, **extra}
    exec(compile(text, path, 'exec'), namespace)
    return namespace[function.__name__]


def rule(name, count=1):
    return repr(name), count


SOFTMAX_RULES = {
    'assign:plan': rule('softmax_plan'),
    'if:not key.is_contiguous()': rule('softmax_key_contiguous'),
    'if:not value.is_contiguous()': rule('softmax_value_contiguous'),
    'assign:out': rule('softmax_output_allocate'),
    'assign:qd': rule('softmax_dense_q_gather'),
    'with:sdpa_kernel': rule('softmax_dense_attention'),
    'assign:out[plan.dense_q]': rule('softmax_dense_scatter'),
    'assign:kw': rule('softmax_window_k_gather'),
    'assign:vw': rule('softmax_window_v_gather'),
    'assign:_ref2va_qw': rule('softmax_window_q_gather'),
    'assign:_ref2va_ow': rule('softmax_window_attention'),
    'assign:out[plan.win_q]': rule('softmax_window_scatter'),
}
READOUT_RULES = {
    'assign:query_by_frame,key,value': rule('linear_features'),
    'assign:A,B': rule('linear_statistics_total'),
    'assign:alpha': rule('linear_alpha'),
    'assign:text_state': rule('linear_text_state'),
    'assign:prefix_states,suffix_states': rule('linear_scans'),
    'assign:linear_state': rule('linear_state_gather'),
    'assign:readout': rule('linear_query_readout'),
    'return:linear_epilogue': rule('linear_norm_gate'),
}
TEXT_RULES = {
    'assign:A,B': rule('linear_text_statistics'),
    'assign:_,injection': rule('linear_text_factor_apply'),
}
FEATURE_RULES = {
    'assign:x,w_tm': ("'linear_' + proj + '_spatial_conv'", 1),
    'assign:out@temporal_conv_activate': ("'linear_' + proj + '_temporal_activate'", 1),
    'assign:out@_compiled': ("'linear_' + proj + '_activate'", 2),
}
SCAN_RULES = {
    'assign:transitions,injections': rule('linear_factor_apply'),
    'for:range(num_frames)': rule('linear_scan_forward'),
    'for:range(num_frames - 1, -1, -1)': rule('linear_scan_reverse'),
}


def prepare_softmax(function, profiler):
    text = source(function, 'window_softmax_decomposed')
    old = '''        out[plan.win_q] = varlen(query[plan.win_q], kw, vw, plan.cu_q, plan.cu_k,
                                 plan.max_q, plan.max_k, scale)'''
    new = '''        _ref2va_qw = query[plan.win_q]
        _ref2va_ow = varlen(_ref2va_qw, kw, vw, plan.cu_q, plan.cu_k,
                            plan.max_q, plan.max_k, scale)
        out[plan.win_q] = _ref2va_ow
        del _ref2va_qw, _ref2va_ow'''
    if text.count(old) != 1:
        raise RuntimeError('Unexpected decomposed window boundary')
    return instrument(function, text.replace(old, new), SOFTMAX_RULES, {'_ref2va_fine': profiler})


def prepare_scans(function, profiler):
    text = source(function, '_run_scans_inference')
    old = 'backend.factor_apply(alpha, A_raw, B_raw)'
    if text.count(old) != 1:
        raise RuntimeError('Unexpected delta factor boundary')
    text = text.replace(old, '_ref2va_fine.factor_apply(backend, alpha, A_raw, B_raw)')
    return instrument(function, text, SCAN_RULES, {'_ref2va_fine': profiler})


def union_us(intervals):
    end = None
    total = 0.
    for first, last in sorted(intervals):
        total += max(0., last - max(first, end if end is not None else first))
        end = max(last, end if end is not None else last)
    return total


def kernel_summary(events, device_index, limit=40):
    """Actual CUDA activity durations, without counting linked CPU events twice."""
    device, cpu, intervals = {}, {}, []
    for event in events:
        if str(event.device_type).endswith('CUDA') and event.device_index == device_index:
            elapsed = event.time_range.elapsed_us()
            row = device.setdefault(event.name, {'name': event.name, 'calls': 0, 'device_ms': 0.})
            row['calls'] += 1
            row['device_ms'] += elapsed / 1000
            intervals.append((event.time_range.start, event.time_range.end))
        elif str(event.device_type).endswith('CPU') and not event.is_user_annotation:
            row = cpu.setdefault(event.name, {'name': event.name, 'calls': 0, 'self_cpu_ms': 0.})
            row['calls'] += 1
            row['self_cpu_ms'] += event.self_cpu_time_total / 1000
    kernels = sorted(device.values(), key=lambda x:x['device_ms'], reverse=True)
    return {'cuda_events_available': bool(intervals), 'device_index': device_index,
            'device_busy_union_ms': union_us(intervals)/1000 if intervals else None,
            'sum_device_event_ms': sum(r['device_ms'] for r in kernels) if intervals else None,
            'nccl_device_event_ms': sum(r['device_ms'] for r in kernels if 'nccl' in r['name'].lower()) if intervals else None,
            'device_event_count': len(intervals), 'top_kernels': kernels[:limit],
            'top_cpu_operators': sorted(cpu.values(), key=lambda x:x['self_cpu_ms'], reverse=True)[:limit],
            'notes': ['Device events include kernels, memcpy and memset; overlapping streams are not additive latency.',
                      'CPU self time includes launch/synchronization overhead, not just CPU arithmetic.',
                      'Tracing changes scheduling; compare latency using separate profile=false requests.']}


class FineProfiler:
    def __init__(self, runtime, linear_kv, decomposed, features, scan, delta, acceleration=None):
        self.runtime, self.linear_kv, self.decomposed = runtime, linear_kv, decomposed
        runtime._ref2va_fine = self
        self.softmax = prepare_softmax(decomposed.window_softmax_decomposed, self)
        self.scans = prepare_scans(scan._run_scans_inference, self)
        self.native_scans_with_factor = instrument(scan._run_scans_inference,
            source(scan._run_scans_inference, '_run_scans_inference'), SCAN_RULES, {'_ref2va_fine': self})
        self.features = instrument(features.prepare_linear_features_inference,
            source(features.prepare_linear_features_inference, 'prepare_linear_features_inference'),
            FEATURE_RULES, {'_ref2va_fine': self})
        self.factors, self.readouts, self.feature_methods, self.text_methods = {}, {}, {}, {}
        for cls in (delta.SanaDelta, delta.VdnDelta, delta.VdnScaledDelta):
            fn = cls.factor_apply
            rules = {'assign:transition': rule('linear_factor_transition'),
                     'assign:injection': rule('linear_factor_injection')}
            if cls is not delta.SanaDelta:
                rules.update({'assign:chol': rule('linear_cholesky'),
                              'assign:inv': rule('linear_inverse')})
            if cls is delta.VdnDelta:
                rules['assign:linv'] = rule('linear_triangular_solve')
            self.factors[cls] = instrument(fn, source(fn, cls.__name__+'.factor_apply'), rules, {'_ref2va_fine': self})
        for branch, native, sampled in linear_kv.bindings:
            # The native contract is audited; the sampled function is generated
            # locally from that same checked source by LinearKVRuntime.
            native_text = source(native.__func__, '_readout_inference')
            for fn, text in ((native.__func__, native_text),
                             (sampled.__func__, inspect.getsource(sampled.__func__))):
                if fn not in self.readouts:
                    self.readouts[fn] = instrument(fn, text, READOUT_RULES,
                        {'_ref2va_fine': self, '_run_scans_inference': self.scans})
            fn = branch._feature_one.__func__
            if fn not in self.feature_methods:
                self.feature_methods[fn] = instrument(fn, source(fn, '_feature_one'), {},
                    {'prepare_linear_features_inference': self.features})
            fn = branch._text_chunk_state.__func__
            if fn not in self.text_methods:
                text = source(fn, '_text_chunk_state')
                old = 'backend.factor_apply(ones, A, B)'
                if text.count(old) != 1:
                    raise RuntimeError('Pinned text factor boundary changed')
                text = text.replace(old, '_ref2va_fine.factor_apply(backend, ones, A, B)')
                self.text_methods[fn] = instrument(fn, text, TEXT_RULES, {'_ref2va_fine': self})
        if acceleration is not None:
            acceleration.profiler = self
            for fn in acceleration.methods.values():
                self.readouts[fn] = instrument(fn, inspect.getsource(fn), READOUT_RULES,
                    {'_ref2va_fine': self})  # retain the accelerated scan binding
            for fn in acceleration.text_methods.values():
                self.text_methods[fn] = instrument(fn, inspect.getsource(fn), TEXT_RULES, {'_ref2va_fine': self})
        self.active = False
        self.trace_active = False
        self.host = {}
        self.trace = {'requested': False, 'collected': False}

    @contextmanager
    def stage(self, name):
        start = self.runtime.profile_start()
        began = time.perf_counter()
        try:
            if self.trace_active:
                import torch
                with torch.profiler.record_function('ref2va::' + name):
                    yield
            else:
                yield
        finally:
            ms = (time.perf_counter() - began) * 1000
            record = self.host.setdefault(name, {'calls': 0, 'host_ms': 0.})
            record['calls'] += 1
            record['host_ms'] += ms
            self.runtime.profile_end(name, start)

    def factor_apply(self, backend, *args):
        return self.factors[type(backend)](backend, *args)

    @contextmanager
    def request(self, kernels=False):
        if self.active:
            raise RuntimeError('Fine profiling cannot overlap requests')
        self.host = {}
        self.trace = {'requested': bool(kernels), 'collected': False}
        if not self.runtime.profile_enabled:
            yield self
            return
        self.active = True
        bindings = []
        profiler = None
        old_softmax = self.decomposed.window_softmax_decomposed
        try:
            # All requests in a resident process are serialized. Only this module
            # entry and these model instances are patched, never torch operators.
            self.decomposed.window_softmax_decomposed = self.softmax
            for branch, _, _ in self.linear_kv.bindings:
                for attr, mapping in (('_readout_inference', self.readouts), ('_feature_one', self.feature_methods),
                                      ('_text_chunk_state', self.text_methods)):
                    original = getattr(branch, attr)
                    bindings.append((branch, attr, original))
                    setattr(branch, attr, types.MethodType(mapping[original.__func__], branch))
            ranks = sorted({0, self.runtime.softmax_ranks} if self.runtime.softmax_ranks else {0})
            self.trace['selected_ranks'] = ranks
            if kernels and self.runtime.rank in ranks:
                import torch
                activities = torch.profiler.supported_activities()
                if torch.profiler.ProfilerActivity.CUDA not in activities:
                    self.trace['error'] = 'CUDA profiling is unavailable in this PyTorch/CUPTI installation'
                else:
                    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=False, with_stack=False)
                    # Do not hide initialization/inference failures or retry a render.
                    profiler.__enter__()
                    self.trace_active = True
            try:
                yield self
            finally:
                self.trace_active = False
                if profiler is not None:
                    began = time.perf_counter()
                    error = sys.exc_info()
                    try:
                        profiler.__exit__(*error)
                        self.trace.update(kernel_summary(profiler.events(), self.runtime.device.index), collected=True,
                                          finalize_seconds=time.perf_counter()-began)
                    except Exception as tracing_error:
                        if error[0] is None:
                            raise
                        self.trace['error'] = str(tracing_error)  # preserve the actual inference exception
        finally:
            self.decomposed.window_softmax_decomposed = old_softmax
            for obj, attr, original in reversed(bindings):
                setattr(obj, attr, original)
            self.active = self.trace_active = False

    def report(self):
        return {'version': 1, 'host_scopes': self.host, 'kernel_trace': self.trace,
                'method': 'CUDA event spans plus unsynchronized host wall time; optional Kineto device activities',
                'scope': 'denoise only; scopes are inclusive and nested; no per-stage synchronize',
                'parents': {'linear_features': ['linear_q_activate', 'linear_k_spatial_conv', 'linear_v_spatial_conv',
                    'linear_k_temporal_activate', 'linear_v_temporal_activate'],
                    'linear_scans': ['linear_factor_apply', 'linear_scan_forward', 'linear_scan_reverse',
                                    'linear_chunk_compose_forward', 'linear_chunk_compose_reverse', 'linear_chunk_scan'],
                    'linear_factor_apply': ['linear_cholesky', 'linear_triangular_solve', 'linear_inverse',
                                           'linear_factor_transition', 'linear_factor_injection', 'linear_fused_delta'],
                    'linear_statistics_total': ['linear_kv_select', 'linear_frame_statistics', 'linear_kv_rescale']},
                'notes': ['Dense/window attention spans include host launch gaps; use kernel_trace for device activities.',
                          'Factor subscopes include both video scans and the optional text-state factorization.',
                          'Spatial-conv scopes include tensor views/weight preparation surrounding conv2d.',
                          'Do not add parent/child scopes, CPU/GPU times, or ranks into latency.']}
