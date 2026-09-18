"""Disjoint VDN query groups with the original restricted K/V domain.

Only the local video leg is approximated. Globals/anchors occupy the leading
exact sink in each group; dense query rows retain their original full key set.
The plan is CPU-only metadata, independent of CUDA kernels and weights.
"""
from collections import OrderedDict
from dataclasses import dataclass


def merge(ranges):
    result = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if result and result[-1][1] >= start:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return tuple(result)


def length(ranges):
    return sum(b-a for a, b in ranges)


@dataclass(frozen=True)
class WindowGroup:
    queries: tuple
    keys: tuple
    sink_tokens: int


class WindowPlan:
    def __init__(self, layout, bounds, anchors, valid_prefix=None):
        S, start, end = layout.seq_len, layout.video_start, layout.video_end
        frames, tpf = layout.num_frames, layout.tokens_per_frame
        if anchors not in ('none', 'rows', 'columns', 'both'):
            raise ValueError('Unknown anchor mode')
        if not (0 <= start < end <= S and frames > 0 and tpf > 0
                and end-start == frames*tpf and len(bounds) == frames):
            raise ValueError('Invalid VDN sequence geometry')
        prefix = start if valid_prefix is None else valid_prefix
        if not 0 <= prefix <= start:
            raise ValueError('Invalid padding boundary')
        def frame(f):
            return (start+f*tpf, start+(f+1)*tpf)
        global_queries = merge(((0, start), (end, S)))
        global_keys = merge(((0, prefix), (end, S)))
        anchor_set = {0, frames-1}
        dense_rows = anchor_set if anchors in ('rows', 'both') else set()
        dense_columns = anchor_set if anchors in ('columns', 'both') else set()
        # Include padding queries in the output, but exclude padding keys when
        # isolation is requested. Real rows can never read the excluded gap.
        self.dense_queries = merge((*global_queries, *(frame(f) for f in dense_rows)))
        self.dense_keys = merge((*global_keys, (start, end)))
        sink = merge((*global_keys, *(frame(f) for f in dense_columns)))
        groups = []
        for f in range(frames):
            if f in dense_rows:
                continue
            lo, hi = bounds[f]
            # OpenVDN supplies inclusive, UNCLAMPED frame/chunk bounds.
            # Negative lo and hi >= frames are normal at either video edge;
            # the native mask only evaluates keys in the real frame domain.
            if max(lo, 0) > min(hi, frames-1):
                raise ValueError(f'VDN window {f} [{lo}, {hi}] has no frames in [0, {frames-1}]')
            if groups and groups[-1][-1]+1 == f and bounds[groups[-1][-1]] == bounds[f]:
                groups[-1].append(f)
            else:
                groups.append([f])
        self.batches = OrderedDict()
        for fs in groups:
            lo, hi = bounds[fs[0]]
            local = merge(frame(f) for f in range(max(lo, 0), min(hi+1, frames))
                          if f not in dense_columns)
            # Keep protected K/V first, even when anchors are nonadjacent in the
            # original sequence. This is a permutation, never a second softmax.
            group = WindowGroup(merge(frame(f) for f in fs), (*sink, *local), length(sink))
            key = (length(group.queries), length(group.keys), group.sink_tokens)
            self.batches.setdefault(key, []).append(group)
        self.query_rows = length(self.dense_queries) + sum(
            length(g.queries) for batch in self.batches.values() for g in batch)
        if self.query_rows != S:
            raise ValueError('VDN query partition is incomplete')


class DevicePlan:
    def __init__(self, plan, device):
        import torch
        def indices(ranges):
            values = [torch.arange(a, b, device=device) for a, b in ranges]
            return torch.cat(values) if values else torch.empty(0, device=device, dtype=torch.long)
        self.dense_queries = indices(plan.dense_queries)
        self.dense_keys = indices(plan.dense_keys)
        self.batches = [(shape, torch.stack([indices(g.queries) for g in batch]),
                        torch.stack([indices(g.keys) for g in batch]))
                       for shape, batch in plan.batches.items()]
