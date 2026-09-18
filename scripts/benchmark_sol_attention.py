"""Resumable native/Sol hot comparison, preserving request conditioning and cache settings."""
import argparse
import json
from pathlib import Path
import statistics

try:
    from benchmark_attention_pipeline import sample
    from benchmark_optimizations import Client, save
except ImportError:
    from scripts.benchmark_attention_pipeline import sample
    from scripts.benchmark_optimizations import Client, save


def variants(layouts, taus, dense_steps, dense_layers):
    result = {}
    for ranks in layouts:
        result[f'native-{ranks}+{8-ranks}'] = dict(attention_kernel='native', softmax_ranks=ranks)
        for tau in taus:
            result[f'sol-{ranks}+{8-ranks}-tau{tau:g}'] = dict(attention_kernel='sol', softmax_ranks=ranks,
                sol_tau=tau, sol_dense_steps=dense_steps, sol_dense_layers=dense_layers)
    return result


def validate_result(job, request):
    up = job['metrics']['upstream']
    actual = up['optimizations']['requested']
    keys = ('attention_kernel', 'sol_tau', 'sol_dense_steps', 'sol_dense_layers')
    if any(actual[k] != request[k] for k in keys if k in request):
        raise RuntimeError('Server did not apply requested attention settings')
    if up['parallel']['softmax_ranks'] != request['softmax_ranks']:
        raise RuntimeError('Server did not apply requested rank split')
    sol = up['optimizations']['attention'].get('sol', {})
    if request['attention_kernel'] == 'sol' and not sol.get('sparse_executed'):
        raise RuntimeError('Sol was selected but no sparse kernel executed; do not label this a Sol benchmark')
    return sol


def summarize(records):
    result = []
    for name in dict.fromkeys(r['variant'] for r in records):
        hot = [r for r in records if r['variant'] == name and r['pass'] > 0]
        if not hot:
            continue
        valid = all(r['graph_reused'] for r in hot)
        median = lambda k: statistics.median(r[k] for r in hot) if valid else None
        result.append(dict(variant=name, valid_hot=valid, hot_runs=len(hot),
                           denoise_seconds=median('denoise_seconds'), output_seconds=median('output_seconds'),
                           processing_seconds=median('processing_seconds'),
                           cached_steps=[r['cached_steps'] for r in hot], videos=[r['video_url'] for r in hot]))
    return result


def run(client, body, root, options, repeat):
    health = client.http('/openvdn/health')
    if not health.get('ready') or health.get('metrics_schema_version', 0) < 8:
        raise RuntimeError('Deploy schema 8 first')
    before = root/'health-before.json'
    if before.exists() and json.loads(before.read_text())['instance'] != health['instance']:
        raise RuntimeError('Worker instance changed; use another result directory')
    save(before, health)
    records = []
    for index in range(repeat+1):
        for name, changes in options.items():
            request = {**body, **changes, 'profile': False}
            print('RUN', name, index, flush=True)
            job = sample(client, request, root/name/f'pass-{index}', health['instance'])
            sol = validate_result(job, request)
            t, up = job['metrics']['timings'], job['metrics']['upstream']
            records.append(dict(variant=name, **{'pass': index}, job_id=job['job_id'],
                denoise_seconds=t['denoise_seconds'], output_seconds=t['decode_and_encode_seconds'],
                processing_seconds=t['processing_wall_seconds'], graph_reused=up['compilation']['runtime_graph_reused'],
                cached_steps=up['cache_dit']['cached_steps'], sol=sol, video_url=client.base+job['video_url']))
            save(root/'results.json', records)
            save(root/'summary.json', summarize(records))
    return summarize(records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', required=True)
    parser.add_argument('--request-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--layouts', nargs='+', type=int, choices=(6, 5, 4), default=[6])
    parser.add_argument('--taus', nargs='+', type=float, default=[1.])
    parser.add_argument('--dense-steps', type=int, choices=range(8), default=1)
    parser.add_argument('--dense-layers', type=int, choices=range(50), default=2)
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 1 or any(not 0 <= tau <= 4 for tau in args.taus):
        parser.error('repeat must be positive; tau must be in [0,4]')
    run(Client(args.server), json.loads(args.request_file.read_text()), args.output_dir,
        variants(args.layouts, args.taus, args.dense_steps, args.dense_layers), args.repeat)
