"""Same-request VAE-only ablation. Cold compilation is never counted as hot."""
import argparse
import copy
import json
from pathlib import Path
import statistics

try:
    from benchmark_optimizations import Client, save
    from benchmark_sglang_acceleration import sample
except ImportError:
    from scripts.benchmark_optimizations import Client, save
    from scripts.benchmark_sglang_acceleration import sample

VARIANTS = {'native': {'vae_tile_batch_size': 1, 'vae_compile': False},
            'batch4': {'vae_tile_batch_size': 4, 'vae_compile': False},
            'batch4_compiled': {'vae_tile_batch_size': 4, 'vae_compile': True}}


def hot_status(task):
    vae = task['video_vae_decode']
    return (task['compilation']['runtime_graph_reused'] and vae['compilation']['runtime_graph_reused']
            and all(not r['verification']['new_checks'] for r in vae['tile_decoder_by_rank']))


def summarize(records):
    rows = []
    for variant in VARIANTS:
        hot = [r for r in records if r['variant'] == variant and r['kind'] == 'hot']
        if not hot:
            continue
        valid = all(r['hot'] for r in hot)
        rows.append({'variant': variant, 'runs': len(hot), 'all_hot': valid,
                     **{k: statistics.median(r[k] for r in hot) if valid else None for k in
                        ('denoise_seconds', 'video_vae_decode_seconds', 'worker_wall_seconds')},
                     'video_urls': [r['video_url'] for r in hot]})
    return rows


def benchmark(client, body, root, repeat=3):
    if body.get('seed') is None or repeat < 1:
        raise ValueError('Use a fixed seed and repeat >= 1')
    health = client.http('/openvdn/health')
    if not health.get('ready') or health.get('metrics_schema_version', 0) < 13:
        raise RuntimeError('Deploy schema 13 and wait for ready first')
    initial = root / 'health-before.json'
    if initial.exists():
        if json.loads(initial.read_text())['instance'] != health['instance']:
            raise RuntimeError('Worker changed; use a new results directory')
    else:
        save(initial, health)
    base = copy.deepcopy(body)
    base['optimization'] = {**health['request_options']['optimizations'],
                            'softmax_ranks': health['profile']['softmax_ranks'],
                            **(body.get('optimization') or {}), 'profile': False, 'profile_kernels': False}
    records = []

    def run(variant, kind, index):
        request = copy.deepcopy(base)
        request['optimization'].update(VARIANTS[variant])
        label = f'{variant}-{kind}-{index}'
        print('RUN', label, flush=True)
        task = sample(client, request, root / label, health['instance'])
        actual = task['optimizations']['requested']
        if any(actual[k] != v for k, v in VARIANTS[variant].items()):
            raise RuntimeError('Server did not apply requested VAE controls')
        for rank in task['video_vae_decode']['tile_decoder_by_rank']:
            if (rank['tile_batch_size'] != VARIANTS[variant]['vae_tile_batch_size'] or
                    rank['compile_enabled'] != VARIANTS[variant]['vae_compile']):
                raise RuntimeError('VAE controls were requested but not effective')
        record = {'variant': variant, 'kind': kind, 'index': index, 'hot': hot_status(task),
                  'task_id': task['id'], 'video_url': task['content']['url'],
                  'cached_steps': task['cache_dit']['cached_steps'],
                  **{k: task['timings'][k] for k in
                     ('denoise_seconds', 'video_vae_decode_seconds', 'worker_wall_seconds')}}
        records.append(record)
        save(root / 'results.json', records)
        save(root / 'summary.json', summarize(records))
        print(json.dumps(record), flush=True)

    for variant in VARIANTS:
        run(variant, 'warmup', 0)
    for i in range(1, repeat + 1):
        for variant in list(VARIANTS) if i % 2 else list(VARIANTS)[::-1]:
            run(variant, 'hot', i)
    after = client.http('/openvdn/health')
    save(root / 'health-after.json', after)
    if after.get('instance') != health['instance']:
        raise RuntimeError('Worker changed during benchmark')
    return summarize(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True)
    parser.add_argument('--request-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args()
    rows = benchmark(Client(args.server), json.loads(args.request_file.read_text()), args.output_dir, args.repeat)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
