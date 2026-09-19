"""Resumable same-request ablations, hot timings and separate fine/kernel profiles.

Accepts the 8b200-style business request JSON. Keeps its prompt, references, seed,
geometry and cache configuration. No SSH or extra Python dependencies required.
"""
import argparse
import copy
import json
from pathlib import Path
import statistics
import time

try:
    from benchmark_optimizations import Client, save
except ImportError:
    from scripts.benchmark_optimizations import Client, save

PREFIX = '/ic/capcut/edit_gateway/v2'
BASE = dict(fused_delta=False, boundary_scan=False, fast_softmax=False, dual_stream=False)
COMBINED = {**BASE, 'fused_delta': True, 'boundary_scan': True, 'fast_softmax': True}
VARIANTS = {
    'native': BASE,
    'fused_delta': {**BASE, 'fused_delta': True},
    'boundary_scan': {**BASE, 'boundary_scan': True},
    'fast_softmax': {**BASE, 'fast_softmax': True},
    'combined': COMBINED,
    'ulysses_serial': {**COMBINED, 'softmax_ranks': 0},
    'ulysses_dual': {**COMBINED, 'softmax_ranks': 0, 'dual_stream': True},
}


def sample(client, request, folder, instance):
    health = client.http('/openvdn/health')
    if health.get('instance') != instance or not health.get('ready'):
        raise RuntimeError('Worker changed/unavailable; do not mix benchmark instances')
    intent, receipt = folder/'request.json', folder/'submit.json'
    if receipt.exists():
        if json.loads(intent.read_text()) != request:
            raise ValueError(f'Request changed: {folder}')
        submission = json.loads(receipt.read_text())
    else:
        if intent.exists():
            raise RuntimeError(f'Uncertain POST: inspect server task/queue before retrying {folder}')
        save(intent, request)
        submission = client.http(PREFIX+'/video_generation', request)
        save(receipt, submission)
    task_id = submission['task_id']
    started = time.monotonic()
    while True:
        status = client.http(PREFIX+'/query/video_generation', {'model':request['model'], 'task_id':task_id})
        save(folder/'status.json', status)
        task = status['task']
        if task['status'] == 'succeeded':
            return task
        if task['status'] in ('failed', 'interrupted', 'cancelled'):
            raise RuntimeError(f'{task_id}: {task.get("error", task["status"])}')
        if time.monotonic()-started > client.timeout:
            raise TimeoutError(f'{task_id} still running; resume this output directory without resubmitting')
        time.sleep(2)


def summary(records):
    rows = []
    for variant in dict.fromkeys(r['variant'] for r in records):
        hot = [r for r in records if r['variant']==variant and r['kind']=='hot']
        if not hot:
            continue
        valid = all(r['graph_reused'] and not r['compiled_new_graph'] for r in hot)
        rows.append({'variant':variant, 'runs':len(hot), 'all_hot':valid,
            **{key:statistics.median(r[key] for r in hot) if valid else None
               for key in ('denoise_seconds','gpu_worker_seconds','worker_wall_seconds')},
            'cached_steps':[r['cached_steps'] for r in hot], 'video_urls':[r['video_url'] for r in hot]})
    return rows


def benchmark(client, body, root, variants, repeat=3, kernels=False):
    if body.get('seed') is None:
        raise ValueError('Set a fixed seed in the benchmark request')
    health = client.http('/openvdn/health')
    if not health.get('ready') or health.get('metrics_schema_version', 0) < 12:
        raise RuntimeError('Deploy schema 12 and wait for ready first')
    initial = root/'health-before.json'
    if initial.exists():
        if json.loads(initial.read_text())['instance'] != health['instance']:
            raise RuntimeError('Worker changed; use a new results directory')
    else:
        save(initial, health)
    records = []

    def run(variant, kind, index=0):
        request = copy.deepcopy(body)
        # Freeze resident defaults in the evidence and change only this ablation.
        opt = {**health['request_options']['optimizations'],
               'softmax_ranks':health['profile']['softmax_ranks'],
               **(request.get('optimization') or {}), **VARIANTS[variant],
               'profile':kind in ('profile','kernels'), 'profile_kernels':kind=='kernels'}
        request['optimization'] = opt
        label = f'{variant}-{kind}-{index}'
        print('RUN', label, flush=True)
        task = sample(client, request, root/label, health['instance'])
        actual = task['optimizations']['requested']
        if any(actual[key] != value for key,value in VARIANTS[variant].items() if key!='softmax_ranks'):
            raise RuntimeError('Server did not apply requested optimization controls')
        timing, compilation = task['timings'], task['compilation']
        record = {'variant':variant, 'kind':kind, 'index':index, 'task_id':task['id'],
            **{key:timing[key] for key in ('denoise_seconds','gpu_worker_seconds','worker_wall_seconds')},
            'graph_reused':compilation['runtime_graph_reused'],
            'compiled_new_graph':compilation['compiled_new_graph'],
            'cached_steps':task['cache_dit']['cached_steps'], 'video_url':task['content']['url']}
        records.append(record)
        save(root/'results.json', records);save(root/'summary.json',summary(records))
        if kind in ('profile','kernels'):
            save(root/label/'profiling.json',task['profiling'])
        print(json.dumps(record),flush=True)

    for variant in variants:
        run(variant,'warmup')
    for iteration in range(1, repeat+1):
        for variant in variants if iteration%2 else variants[::-1]:
            run(variant,'hot',iteration)
    for variant in variants:
        run(variant,'profile')
    if kernels:
        for variant in ('native','combined','ulysses_dual'):
            if variant in variants:
                run(variant,'kernels')
    after = client.http('/openvdn/health')
    save(root/'health-after.json',after)
    if after['instance'] != health['instance']:
        raise RuntimeError('Worker restarted; do not combine these measurements')
    return summary(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',required=True)
    parser.add_argument('--request-file',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    parser.add_argument('--repeat',type=int,default=3)
    parser.add_argument('--kernels',action='store_true',help='Separate expensive Kineto traces after hot measurements')
    args=parser.parse_args()
    if args.repeat<1:parser.error('--repeat must be positive')
    benchmark(Client(args.server),json.loads(args.request_file.read_text()),args.output_dir,args.variants,args.repeat,args.kernels)


if __name__=='__main__':main()
