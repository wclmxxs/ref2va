"""Resumable same-case ablation of schema-7 optimizations. No server restart."""
import argparse
import json
from pathlib import Path
import statistics
import time

try:
    from benchmark_optimizations import Client, save
except ImportError:
    from scripts.benchmark_optimizations import Client, save

BASELINE = dict(fast_communication=False, attention_kernel='native', isolate_padding=False,
                streaming_output=False, cleanup_policy='always', linear_stats_chunk_frames=16, linear_kv_keep_ratio=1.0,
                profile_kernels=False, fused_delta=False, boundary_scan=False, fast_softmax=False, dual_stream=False,
                vae_tile_batch_size=1, vae_compile=False)
COMBINED = {**BASELINE, 'fast_communication':True, 'streaming_output':True, 'cleanup_policy':'adaptive'}
VARIANTS = {
    'baseline': BASELINE,
    'communication': {**BASELINE,'fast_communication':True},
    'streaming': {**BASELINE,'streaming_output':True},
    'cleanup': {**BASELINE,'cleanup_policy':'adaptive'},
    'combined': COMBINED,
    'decomposed': {**COMBINED,'attention_kernel':'decomposed'},
    'linear32': {**COMBINED,'linear_stats_chunk_frames':32},
    'isolated': {**COMBINED,'isolate_padding':True},
}


def sample(client, request, folder, instance):
    intent, submit = folder/'request.json', folder/'submit.json'
    health = client.http('/openvdn/health')
    if health.get('instance') != instance or not health.get('ready'):
        raise RuntimeError('Worker changed or unavailable; do not mix benchmark instances')
    if submit.exists():
        if json.loads(intent.read_text()) != request:
            raise ValueError(f'Request changed for {folder}')
        submission = json.loads(submit.read_text())
    else:
        if intent.exists():
            raise RuntimeError(f'Uncertain POST: inspect the server queue before retrying {folder}')
        save(intent,request)
        submission = client.http('/openvdn/jobs', request)
        save(submit,submission)
    # A polling timeout never cancels or duplicates the already accepted job.
    started = time.monotonic()
    while True:
        job = client.http(submission['status_url'])
        save(folder/'status.json',job)
        if job['status']=='succeeded':
            return job
        if job['status'] in ('failed','interrupted'):
            raise RuntimeError(job.get('error',job['status']))
        if time.monotonic()-started > client.timeout:
            raise TimeoutError(f"Still running: {submission['job_id']}; resume using this output directory")
        time.sleep(2)


def summarize(records):
    rows=[]
    for variant in dict.fromkeys(r['variant'] for r in records):
        hot=[r for r in records if r['variant']==variant and r['pass']>0]
        if not hot:
            continue
        reused=all(r['graph_reused'] for r in hot)
        median=lambda key: statistics.median(r[key] for r in hot)
        rows.append(dict(variant=variant,runs=len(hot),all_hot_graphs_reused=reused,
                         denoise_seconds=median('denoise_seconds') if reused else None,
                         output_seconds=median('output_seconds') if reused else None,
                         processing_seconds=median('processing_seconds') if reused else None,
                         cached_steps=[r['cached_steps'] for r in hot], videos=[r['video_url'] for r in hot]))
    return rows


def run(client, body, root, variants, repeat):
    if body.get('softmax_backend','flex') != 'flex':
        raise ValueError('Use the current flex resident profile for a same-bucket comparison')
    health=client.http('/openvdn/health')
    if not health.get('ready') or health.get('metrics_schema_version',0)<13:
        raise RuntimeError('Deploy schema 13 first (VAE optimizations are explicitly disabled in this ablation)')
    before=root/'health-before.json'
    if before.exists() and json.loads(before.read_text())['instance']!=health['instance']:
        raise RuntimeError('Worker instance changed; start a separate output directory')
    save(before,health)
    records=[]
    # One warm pass per variant, then interleave hot repetitions to reduce drift.
    for index in range(repeat+1):
        for variant in variants:
            request={**body,**VARIANTS[variant],'profile':False}
            print('RUN',variant,index,flush=True)
            job=sample(client,request,root/variant/f'pass-{index}',health['instance'])
            metrics=job['metrics'];up=metrics['upstream'];t=metrics['timings']
            actual=up['optimizations']['requested']
            if any(actual[k]!=v for k,v in VARIANTS[variant].items()):
                raise RuntimeError('Server did not apply requested optimization controls')
            record={'variant':variant,'pass':index,'job_id':job['job_id'],
                         'denoise_seconds':t['denoise_seconds'],
                         'output_seconds':t['decode_and_encode_seconds'],
                         'processing_seconds':t['processing_wall_seconds'],
                         'graph_reused':up['compilation']['runtime_graph_reused'],
                         'cached_steps':up['cache_dit']['cached_steps'],
                         'video_url':client.base+job['video_url'],
                         'cleanup':up['optimizations']['cleanup']}
            records.append(record)
            save(root/'results.json',records)
            save(root/'summary.json',summarize(records))
    rows=summarize(records)
    lines=['# Attention / communication / output comparison','',
           'Hot-run medians only. Compile misses invalidate a variant. Timings overlap; do not sum component columns.',
           'Kernel/padding experiments need separate visual review. Parameters and full responses are saved beside this report.','',
           '| Variant | Hot runs | DiT s | Output wall s | Processing s | Cache steps |',
           '|---|---:|---:|---:|---:|---|']
    for row in rows:
        fmt=lambda key: f'{row[key]:.3f}' if row[key] is not None else 'INVALID: compiled'
        lines.append(f"| {row['variant']} | {row['runs']} | {fmt('denoise_seconds')} | {fmt('output_seconds')} | {fmt('processing_seconds')} | {row['cached_steps']} |")
    lines+=['','Videos:']
    lines += [f"- {row['variant']}: "+', '.join(f'[run {i+1}]({url})' for i,url in enumerate(row['videos'])) for row in rows]
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--server',required=True)
    parser.add_argument('--request-file',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--repeat',type=int,default=3)
    parser.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS))
    args=parser.parse_args()
    if args.repeat<1:
        parser.error('--repeat must be positive')
    run(Client(args.server),json.loads(args.request_file.read_text()),args.output_dir,args.variants,args.repeat)
