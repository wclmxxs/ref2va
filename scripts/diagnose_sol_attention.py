"""Model-free Sol diagnostics. Save synthetic inputs for offline reproduction.

Run after the resident worker has stopped. This uses one GPU and no NCCL.
It does not change kernel settings or the validation tolerance.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def error_metrics(actual, expected):
    import torch
    actual, expected = actual.float(), expected.float()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return {'passed': False, 'finite': False}
    delta = actual-expected
    relative = float(delta.norm()/expected.norm().clamp_min(1e-8))
    maximum = float(delta.abs().max())
    result = dict(passed=relative <= .025 and maximum <= .08, finite=True,
                  relative_l2=relative, max_abs=maximum)
    if actual.ndim == 4:
        per_head = delta.square().sum((0, 1, 3)).sqrt()/expected.square().sum((0, 1, 3)).sqrt().clamp_min(1e-8)
        result['relative_l2_by_head'] = per_head.tolist()
        result['relative_l2_by_query_block'] = [
            float(d.norm()/r.norm().clamp_min(1e-8))
            for d, r in zip(delta.split(64, 1), expected.split(64, 1))]
        result['worst_index_bthd'] = list(torch.unravel_index(delta.abs().argmax(), delta.shape))
        result['worst_index_bthd'] = [int(i) for i in result['worst_index_bthd']]
    return result


def reference_preprocess(q, k, v, scale, tau):
    import torch
    kc = torch.stack([x.float().mean(1).to(k.dtype) for x in k.split(64, 1)], 1)
    vc = torch.stack([x.float().sum(1).to(v.dtype) for x in v.split(64, 1)], 1)
    mean = kc.float().mean(1)
    variance = (kc.float().square().mean(1)-mean.square()).clamp_min(0)
    thresholds = []
    for query in q.split(64, 1):
        qb = query.float().mean(1)
        center = (qb*mean).sum(-1)*scale/math.log(2)
        spread = (qb.square()*variance).sum(-1)*(scale/math.log(2))**2
        thresholds.append(center+tau*(spread+1e-6).sqrt())
    return kc, vc, torch.stack(thresholds, 1)


def route_details(q, kc, threshold, scale, sink):
    import torch
    margins, masks = [], []
    blocks = torch.arange(kc.shape[1], device=q.device)
    for i, query in enumerate(q.split(64, 1)):
        score = torch.einsum('bqhd,bkhd->bhqk', query.float(), kc.float()).mean(2)*scale/math.log(2)
        margin = score-threshold[:, i, :, None]
        exact = (margin > 0) | ((blocks-i).abs() <= 1) | (blocks < (sink+63)//64)
        margins.append(margin)
        masks.append(exact)
    # These are independently computed routes, not a readback of the CUDA mask.
    return torch.stack(margins, 1), torch.stack(masks, 1)


def diagnose_case(kernel, q, k, v, folder, heads):
    import torch
    from openvdn_comfy.sol_kernel import prepare_rectangular
    from openvdn_comfy.sol_reference import sol_reference
    kwargs = dict(scale=128**-.5, tau=1., sink_tokens=65)
    actual = kernel(q, k, v, **kwargs)
    expected = sol_reference(q, k, v, **kwargs)
    before = kernel.snapshot()
    again = kernel(q, k, v, **kwargs)
    torch.cuda.synchronize(q.device)
    hot = kernel.since(before)
    exact = kernel(q, k, v, **kwargs, all_exact=True)
    dense = sol_reference(q, k, v, **kwargs, all_exact=True)
    result = dict(heads=heads, sparse=error_metrics(actual, expected),
                  all_exact=error_metrics(exact, dense),
                  repeat_exact=bool(torch.equal(actual, again)), hot_compile_misses=hot['compile_misses'])
    result['passed'] = (result['sparse']['passed'] and result['all_exact']['passed']
                        and result['repeat_exact'] and not hot['compile_misses'])
    if not result['passed']:
        kc, vc, threshold = prepare_rectangular(q, k, v, scale=kwargs['scale'], tau=kwargs['tau'])
        ref_kc, ref_vc, ref_threshold = reference_preprocess(q, k, v, kwargs['scale'], kwargs['tau'])
        result['preprocess'] = {name: error_metrics(a, b) for name, a, b in (
            ('kc', kc, ref_kc), ('vc', vc, ref_vc), ('threshold', threshold, ref_threshold))}
        margin, routes = route_details(q, kc, threshold, kwargs['scale'], kwargs['sink_tokens'])
        ref_margin, ref_routes = route_details(q, ref_kc, ref_threshold, kwargs['scale'], kwargs['sink_tokens'])
        result['reference_route_differences_after_preprocess'] = int((routes != ref_routes).sum())
        result['minimum_reference_threshold_margin'] = float(ref_margin.abs().min())
        # Copy only synthetic test data, never model weights or user media.
        tensors = dict(q=q, k=k, v=v, sparse=actual, repeated=again, all_exact=exact,
                       expected=expected, dense=dense, kc=kc, vc=vc, threshold=threshold,
                       ref_kc=ref_kc, ref_vc=ref_vc, ref_threshold=ref_threshold,
                       margin=margin, reference_routes=routes, ref_margin=ref_margin,
                       ref_routes=ref_routes)
        tensors = {name: tensor.detach().cpu().clone() for name, tensor in tensors.items()}
        cpu_expected = sol_reference(tensors['q'], tensors['k'], tensors['v'], **kwargs)
        result['sparse_vs_cpu_reference'] = error_metrics(tensors['sparse'], cpu_expected)
        result['gpu_vs_cpu_reference'] = error_metrics(tensors['expected'], cpu_expected)
        tensors['cpu_expected'] = cpu_expected
        target = folder/f'heads-{heads}.pt'
        torch.save(dict(tensors=tensors, parameters=kwargs, result=result), target)
        result['sample_file'] = target.name
    save_json(folder/f'heads-{heads}.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'output/sol-diagnostics')
    args = parser.parse_args()
    import torch
    from openvdn_comfy.sol_kernel import SolKernel
    torch.cuda.set_device(args.device)
    device = torch.device('cuda', args.device)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    folder = args.output_dir/run_id
    report = dict(run_id=run_id, status='running', passed=False, cases=[],
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  torch_version=torch.__version__, device=torch.cuda.get_device_name(device),
                  matmul_precision=torch.get_float32_matmul_precision(),
                  allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    def persist():
        save_json(folder/'report.json', report)
        save_json(args.output_dir/'latest.json', dict(run_id=run_id, status=report['status'],
                                                    report=f'{run_id}/report.json'))
    persist()
    print(f'Sol diagnostic artifacts: {folder}', flush=True)
    try:
        kernel = SolKernel()
        report['verification'] = kernel.verify(device)
        persist()
        # Preserve the validator's RNG sequence to reproduce its heads=12 case.
        generator = torch.Generator(device=device).manual_seed(33)
        for heads in (9, 10, 11, 12, 14):
            q = torch.randn((2, 129, heads, 128), device=device, dtype=torch.bfloat16, generator=generator)
            k, v = [torch.randn((2, 517, heads, 128), device=device, dtype=torch.bfloat16,
                                generator=generator) for _ in range(2)]
            case = diagnose_case(kernel, q, k, v, folder, heads)
            report['cases'].append(case)
            persist()
            print(json.dumps(case, allow_nan=False), flush=True)
        report['passed'] = all(case['passed'] for case in report['cases'])
        report['status'] = 'passed' if report['passed'] else 'failed'
    except Exception as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        persist()
    print(f"Sol diagnostics {report['status']}: {folder/'report.json'}", flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
