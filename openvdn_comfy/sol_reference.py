"""Slow independent Sol math oracle; only used by tests/first-opt-in checks."""
import math


def sol_reference(q, k, v, *, scale, tau, sink_tokens=0, all_exact=False):
    import torch
    # Match BF16 centroids and block value sums, but use FP32 attention math.
    kc = torch.stack([x.float().mean(1).to(k.dtype).float()
                      for x in k.split(64, dim=1)], dim=1)
    vc = torch.stack([x.float().sum(1).to(v.dtype).float()
                      for x in v.split(64, dim=1)], dim=1)
    mean = kc.mean(1)
    variance = (kc.square().mean(1) - mean.square()).clamp_min(0)
    result = []
    key_blocks = torch.arange(kc.shape[1], device=q.device)
    for query_block, query in enumerate(q.split(64, dim=1)):
        qb = query.float().mean(1)
        threshold = (qb * mean).sum(-1) * scale / math.log(2)
        threshold += tau * ((qb.square() * variance).sum(-1) * (scale / math.log(2))**2 + 1e-6).sqrt()
        centroid_logits = torch.einsum('bqhd,bkhd->bhqk', query.float(), kc) * scale
        exact = centroid_logits.mean(2) / math.log(2) > threshold[..., None]
        # The pinned Sol selector always keeps q_block +/- 1 exact, including
        # below-threshold blocks. These are the packed window block indices.
        # See common/selector.py::sol_attn_route_is_exact; sinks are additional.
        exact |= (key_blocks-query_block).abs() <= 1
        exact[..., :(sink_tokens+63)//64] = True
        if all_exact:
            exact.fill_(True)
        logits, values = [], []
        # Nonselected blocks repeat a centroid score and the rounded mean V.
        # This equals exp(score)*block_length and exp(score)*V_sum in Sol.
        for block, start in enumerate(range(0, k.shape[1], 64)):
            keys, vals = k[:, start:start+64].float(), v[:, start:start+64].float()
            true_logits = torch.einsum('bqhd,bkhd->bhqk', query.float(), keys) * scale
            keep = exact[..., block, None, None]
            logits.append(torch.where(keep, true_logits, centroid_logits[..., block, None]))
            values.append(torch.where(keep.transpose(1, 2), vals,
                                      vc[:, block, None] / keys.shape[1]))
        weights = torch.cat(logits, dim=-1).softmax(-1)
        result.append(torch.einsum('bhqk,bkhd->bqhd', weights, torch.cat(values, dim=1)))
    return torch.cat(result, dim=1)
