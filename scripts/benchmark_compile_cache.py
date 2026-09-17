"""Allow first-use compilation, then check a second hot pass of every original request.

python3 scripts/benchmark_compile_cache.py --server http://HOST:8188 \
    --requests-dir /path/to/original-cases --output-dir work/cache-benchmark

Input layout: case-name/request.json (the public REST body). No prompt rewriting,
reference substitution or setting changes. Only sequential renders are submitted.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_optimizations import Client, save


def summarize_job(name, pass_index, job, server):
    metrics = job['metrics']
    upstream = metrics['upstream']
    if metrics.get('metrics_schema_version', 0) < 6:
        raise RuntimeError('Server needs schema 6; deploy before benchmarking')
    actual, plan = upstream['actual_geometry'], upstream['render_plan']
    if (not actual.get('validated') or any(actual[key] != plan[key]
            for key in ('generation_width', 'generation_height'))):
        raise RuntimeError('Actual sampler canvas differs from the request')
    compilation, timing = upstream['compilation'], metrics['timings']
    return {'case': name, 'pass': pass_index, 'job_id': job['job_id'],
            'duration': job['request']['duration'], 'ratio': job['request']['ratio'],
            'resolution': job['request']['resolution'],
            'reference_short_edge': job['request'].get('reference_short_edge', 768),
            'graph_reused': compilation['runtime_graph_reused'],
            'compiled_new_graph': compilation['compiled_new_graph'],
            'compile_seconds': compilation['dynamo_compile_seconds'],
            'disk_hits': compilation['disk_graph_cache_hits'],
            'geometry_id': compilation['geometry_id'],
            'prefix_capacity': compilation['token_bucket']['prefix_capacity'],
            'padding_tokens': compilation['token_bucket']['padding_tokens'],
            'conditioning_seconds': timing['conditioning_seconds'],
            'denoise_wall_seconds': timing['denoise_wall_seconds'],
            'hot_denoise_seconds': timing['hot_denoise_seconds'],
            'video_vae_seconds': timing.get('video_vae_decode_seconds'),
            'audio_vae_seconds': timing.get('audio_vae_decode_seconds'),
            'generation_wall_seconds': timing['generation_wall_seconds'],
            'cache_dit_hits': upstream['cache_dit']['cache_hits'],
            'video_url': server.rstrip('/') + job['video_url']}


def benchmark(client, files, output, passes=2, allow_first_compile=True):
    health = client.http('/openvdn/health')
    if not health.get('ready') or health.get('metrics_schema_version', 0) < 6:
        raise RuntimeError('Deploy schema 6 and wait until startup warmup finishes')
    instance = health['instance']
    save(output / 'health-before.json', health)
    bodies = [(path.parent.name, json.loads(path.read_text())) for path in files]
    rows = []
    for pass_index in range(1, passes + 1):
        for name, body in bodies:
            job = client.run(body, output / f'pass-{pass_index}' / name, instance)
            rows.append(summarize_job(name, pass_index, job, client.base))
            save(output / 'results.json', rows)
            print(f"{name} pass={pass_index} reused={rows[-1]['graph_reused']} "
                  f"denoise={rows[-1]['denoise_wall_seconds']:.3f}s "
                  f"compile={rows[-1]['compile_seconds']:.3f}s", flush=True)
    after = client.http('/openvdn/health')
    save(output / 'health-after.json', after)
    if after.get('instance') != instance:
        raise RuntimeError('Server restarted during the benchmark')
    with (output / 'timings.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checked = [row for row in rows if not (allow_first_compile and row['pass'] == 1)]
    misses = [row for row in checked if not row['graph_reused']]
    report = {'instance': instance, 'cases': len(bodies), 'passes': passes,
              'checked_runs': len(checked), 'graph_hits': len(checked) - len(misses),
              'all_graphs_reused': not misses, 'misses': misses,
              'scope': 'actual denoise runs; compiler time is not subtracted from latency'}
    save(output / 'summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True)
    parser.add_argument('--requests-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--passes', type=int, default=2)
    parser.add_argument('--allow-first-compile', action=argparse.BooleanOptionalAction, default=True,
                        help='Allow cold first requests (default); --no-allow-first-compile checks both passes')
    args = parser.parse_args()
    files = sorted(args.requests_dir.glob('*/request.json'))
    if not files or args.passes < 2:
        parser.error('Need case-name/request.json files and at least two passes')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use an empty output directory to preserve previous measurements')
    report = benchmark(Client(args.server), files, args.output_dir, args.passes, args.allow_first_compile)
    print(json.dumps(report, indent=2))
    if not report['all_graphs_reused']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
