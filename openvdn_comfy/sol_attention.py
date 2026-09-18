"""VDN-window adapter. Never changes the trained linear branch or its gate."""
from collections import OrderedDict

from .sol_plan import DevicePlan, WindowPlan


def dense_attention(q, k, v, scale):
    from torch.nn.attention import sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention
    from src.models.softmax_attention.decomposed import _dense_backends
    # No math fallback: it would materialize a very large dense score matrix.
    with sdpa_kernel(_dense_backends()):
        return scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                            v.transpose(0, 1)[None], scale=scale)[0].transpose(0, 1)


class WindowSol:
    def __init__(self, kernel=None, dense=dense_attention, staging_bytes=128*1024**2):
        from .sol_kernel import SolKernel
        self.kernel = kernel if kernel is not None else SolKernel()
        self.dense = dense
        self.staging_bytes = staging_bytes
        self.plans = OrderedDict()
        self.reset()

    def reset(self):
        self.calls = self.launches = self.dense_rows = self.window_rows = self.exact_sink_tokens = 0
        self.plan_misses = 0

    def __call__(self, q, k, v, layout, bounds, scale, *, anchors, tau, valid_prefix=None):
        import torch
        if q.ndim != 3 or q.shape != k.shape or k.shape != v.shape:
            raise ValueError('Window Sol expects equal sequence/head Q/K/V tensors')
        key = (layout.seq_len, layout.video_start, layout.video_end, layout.num_frames,
               layout.tokens_per_frame, tuple(bounds), anchors, valid_prefix, str(q.device))
        if key not in self.plans:
            self.plans[key] = DevicePlan(WindowPlan(layout, bounds, anchors, valid_prefix), q.device)
            self.plan_misses += 1
            while len(self.plans) > 32:
                self.plans.popitem(last=False)
        self.plans.move_to_end(key)
        plan = self.plans[key]
        out = torch.empty_like(q)
        if len(plan.dense_queries):
            qi, ki = plan.dense_queries, plan.dense_keys
            out[qi] = self.dense(q[qi], k[ki], v[ki], scale)
            self.dense_rows += qi.numel()
        for (tq, tk, sink), query_indices, key_indices in plan.batches:
            # Bound simultaneous gather buffers; never pad Q out to K length.
            per_window = (2*tq+2*tk) * q.shape[1] * q.shape[2] * q.element_size()
            batch_size = max(1, self.staging_bytes // per_window)
            for start in range(0, len(query_indices), batch_size):
                qi, ki = query_indices[start:start+batch_size], key_indices[start:start+batch_size]
                out[qi] = self.kernel(q[qi], k[ki], v[ki], scale=scale, tau=tau, sink_tokens=sink)
                self.launches += 1
                self.window_rows += qi.numel()
                self.exact_sink_tokens += sink * len(qi)
        self.calls += 1
        return out

    def report(self):
        return dict(window_attention_calls=self.calls, sparse_kernel_launches=self.launches,
                    sparse_executed=self.launches > 0, dense_query_rows=self.dense_rows,
                    window_query_rows=self.window_rows, kernel_query_rows=self.window_rows,
                    protected_key_tokens=self.exact_sink_tokens, plan_misses=self.plan_misses,
                    scope='local video softmax only; globals/anchors exact; VDN linear branch unchanged')
