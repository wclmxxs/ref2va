"""Sequential same-case layout and DBCache comparison through the public REST API.

python3 scripts/benchmark_optimizations.py --server http://HOST:8188 \
  --request-file case.json --output-dir work/optimization-benchmark
No SSH, remote shell, restarts, external jobs, or new Python dependencies required.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


class Client:
    def __init__(self, base, timeout=1800):
        self.base, self.timeout = base.rstrip('/'), timeout

    def http(self, path, body=None):
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        request = Request(self.base + path, data=data, headers={'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as error:
            raise RuntimeError(f'HTTP {error.code}: {error.read().decode()[:4000]}') from error

    def run(self, body, folder, instance):
        health = self.http('/openvdn/health')
        if health.get('instance') != instance or not health.get('ready'):
            raise RuntimeError('Server restarted or is not ready; do not mix benchmark instances')
        save(folder / 'request.json', body)
        submitted = self.http('/openvdn/jobs', body)
        save(folder / 'submit.json', submitted)
        started = time.monotonic()
        while True:
            job = self.http(submitted['status_url'])
            save(folder / 'status.json', job)
            if job['status'] == 'succeeded':
                return job
            if job['status'] in ('failed', 'interrupted'):
                raise RuntimeError(f"{submitted['job_id']}: {job.get('error')}")
            if time.monotonic() - started > self.timeout:
                raise TimeoutError(f"Job is still on the server: {submitted['job_id']}; no duplicate submitted")
            time.sleep(2)


def summarize(jobs):
    rows = []
    variants = list(dict.fromkeys(j['variant'] for j in jobs if j['kind'] == 'hot'))
    for variant in variants:
        hot = [j for j in jobs if j['variant'] == variant and j['kind'] == 'hot']
        median = lambda key: statistics.median(j[key] for j in hot)
        rows.append({'variant': variant, 'softmax_ranks': hot[0]['softmax_ranks'],
                     'cache_dit': hot[0]['cache_dit'], 'threshold': hot[0]['threshold'],
                     'runs': len(hot), 'denoise_seconds': median('denoise_seconds'),
                     'gpu_seconds': median('gpu_seconds'), 'processing_seconds': median('processing_seconds'),
                     'cache_hits': [j['cache_hits'] for j in hot],
                     'compiled_hot_runs': sum(j['compiled_new_graph'] for j in hot),
                     'video_urls': [j['video_url'] for j in hot]})
    return rows


def write_report(root, jobs):
    rows = summarize(jobs)
    save(root / 'summary.json', rows)
    lines = ['# OpenVDN layout / DBCache benchmark', '',
             'Same input, seed and geometry. One warmup per variant, then median of hot runs.',
             'Profile samples are diagnostic only and excluded from speed comparisons.',
             'GPU = denoising + video VAE + audio VAE. Approximation quality requires video review.', '',
             '| Variant | Hot runs | Denoise s | GPU s | Processing s | Cache hits | New-graph hot runs |',
             '|---|---:|---:|---:|---:|---|---:|']
    for row in rows:
        lines.append(f"| {row['variant']} | {row['runs']} | {row['denoise_seconds']:.3f} | "
                     f"{row['gpu_seconds']:.3f} | {row['processing_seconds']:.3f} | "
                     f"{row['cache_hits']} | {row['compiled_hot_runs']} |")
    lines += ['', '## Samples', '']
    for row in rows:
        lines.append(f"- {row['variant']}: " + ', '.join(f'[video {i+1}]({url})' for i,url in enumerate(row['video_urls'])))
    lines += ['', '## Profiles', '',
              'Per-rank input / block / attention / FFN / final gather / output head timings,',
              'plus nested branch packing, exposed dispatch waits and return communication,',
              'are saved in each profile/status.json under metrics.upstream.parallel_profile.',
              'These compute-stream spans include waits and enqueue gaps; they are not isolated NCCL kernel times.']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    return rows


def benchmark(client, body, root, layouts=(6,5,4), thresholds=(.04,.08,.12), repeat=3):
    health = client.http('/openvdn/health')
    if not health.get('ready') or health.get('metrics_schema_version',0) < 5:
        raise RuntimeError('Deploy schema 5 and wait for ready before benchmarking')
    if not health.get('request_options',{}).get('profile'):
        raise RuntimeError('Per-request profiling unavailable')
    if (root / 'results.json').exists():
        raise ValueError('Output directory already has a run; use a new directory to avoid overwriting evidence')
    root.mkdir(parents=True, exist_ok=True)
    save(root / 'health-before.json',health)
    save(root / 'original-request.json',body)
    jobs = []

    def run(variant, kind, request, index=0):
        name = f'{variant}-{kind}' + (f'-{index}' if index else '')
        print(f'Running {name}', flush=True)
        job = client.run(request, root / name, health['instance'])
        metrics, upstream = job['metrics'],job['metrics']['upstream']
        timing = metrics['timings']
        record = {'variant':variant,'kind':kind,'job_id':job['job_id'],
                  'softmax_ranks':request['softmax_ranks'],'cache_dit':request['cache_dit'],
                  'threshold':request.get('cache_dit_threshold',.08),
                  'denoise_seconds':timing['denoise_seconds'],
                  'gpu_seconds':sum(timing[k] for k in ('denoise_seconds','video_vae_decode_seconds','audio_vae_decode_seconds')),
                  'processing_seconds':timing['processing_wall_seconds'],
                  'cache_hits':upstream['cache_dit']['cache_hits'],
                  'compiled_new_graph':upstream['compilation']['compiled_new_graph'],
                  'video_url':client.base + job['video_url']}
        jobs.append(record)
        save(root / 'results.json',jobs)
        write_report(root,jobs)
        print(f"  denoise={record['denoise_seconds']:.3f}s, hits={record['cache_hits']}",flush=True)

    for layout in layouts:
        variant = f'layout-{layout}-cache-off'
        request = {**body,'softmax_ranks':layout,'profile':False,'cache_dit':False}
        run(variant,'warmup',request)
        for i in range(repeat):
            run(variant,'hot',request,i+1)
        run(variant,'profile',{**request,'profile':True})
    # Benchmark caching on the measured fastest uncached layout, keeping a direct
    # uncached comparator. This choice is NOT applied to the service defaults.
    rows = summarize(jobs)
    clean = [r for r in rows if not r['compiled_hot_runs']]
    if not clean:
        raise RuntimeError('All layouts compiled during hot runs; rerun after shapes warm before choosing a layout')
    best = min(clean,key=lambda r:r['denoise_seconds'])['softmax_ranks']
    for threshold in thresholds:
        variant = f'layout-{best}-cache-{threshold:g}'
        request = {**body,'softmax_ranks':best,'profile':False,'cache_dit':True,'cache_dit_threshold':threshold}
        run(variant,'warmup',request)
        for i in range(repeat):
            run(variant,'hot',request,i+1)
    after = client.http('/openvdn/health')
    save(root / 'health-after.json',after)
    if after.get('instance') != health['instance']:
        raise RuntimeError('Server instance changed; benchmark cannot be compared as a single session')
    return write_report(root,jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True)
    parser.add_argument('--request-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--layouts', type=int, nargs='+', default=[6,5,4])
    parser.add_argument('--thresholds', type=float, nargs='*', default=[.04,.08,.12])
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 1 or not all(0 <= x <= 7 for x in args.layouts):
        parser.error('repeat must be >= 1; layouts must be in [0,7]')
    if not all(0 <= t <= 1 for t in args.thresholds):
        parser.error('thresholds must be in [0,1]')
    benchmark(Client(args.server), json.loads(args.request_file.read_text()), args.output_dir,
              args.layouts,args.thresholds,args.repeat)
    print(args.output_dir / 'report.md')


if __name__ == '__main__':
    main()
