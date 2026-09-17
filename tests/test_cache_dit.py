from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openvdn_comfy.api import normalize_request, prompt_graph
from openvdn_comfy.backend import validate_profile, PROFILE_FIELDS
from openvdn_comfy.cache_dit import CacheConfig, DBCache, FIELDS
from openvdn_comfy.config import Settings
from openvdn_comfy.dit_runtime import summarize_cache, summarize_profiles
from openvdn_comfy.nodes import OpenVDNH200Generate, OpenVDNH200Request


def runtime(rank=0, world_size=1):
    return types.SimpleNamespace(rank=rank, world_size=world_size, local_start=0, local_end=4,
                                 profile_start=lambda: None, profile_end=lambda *a: None)


def config(**kw):
    return CacheConfig(enabled=True, fn_blocks=1, bn_blocks=1, warmup_steps=2, **kw)


def advance(cache, x, blocks=None):
    blocks = blocks or [lambda x: x + 1, lambda x: x * 2, lambda x: x + 3, lambda x: x - 4]
    return cache.run(blocks, x, (), lambda block, value, args: block(value))


def groups(cache):
    # Reference rows 0, target video rows 1:3, audio row 3; zero text rows.
    cache.configure_groups(torch.zeros(1, 4, 2), torch.arange(3), torch.tensor([3]), 1, 3)


def test_disabled_and_zero_threshold_are_exact_and_do_not_collect(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('disabled cache touched approximation state')
    cache = DBCache(runtime())
    monkeypatch.setattr(cache, 'decision', forbidden)
    monkeypatch.setattr(torch.Tensor, 'clone', forbidden)
    for cfg in [replace(config(), enabled=False),
                replace(config(), threshold=0), replace(config(), max_cached_steps=0)]:
        with cache.request(cfg, total_blocks=4):
            for i in range(8):
                x = torch.full((1, 4, 2), float(i))
                assert torch.equal(advance(cache, x), (x + 1) * 2 + 3 - 4)
            assert cache.report()['cache_hits'] == 0
            assert cache.report()['executed_blocks'] == 32
            assert cache.previous is cache.residual is None


def test_reuse_formula_guards_suffix_and_request_isolation():
    cache = DBCache(runtime())
    for request in range(2):
        with cache.request(config(), total_blocks=4):
            groups(cache)
            outputs = []
            for i in range(8):
                outputs.append(advance(cache, torch.full((1, 4, 2), float(i + request * 10))))
            report = cache.report()
            assert report['cached_steps'] == [3, 5]
            assert report['skipped_blocks'] == 4
            assert report['executed_blocks'] == 28
            assert report['decisions'][0]['reason'] == 'warmup'
            assert report['decisions'][3]['reason'] == 'max_consecutive'
            assert report['decisions'][5]['reason'] == 'max_cached_steps'
            assert report['decisions'][7]['reason'] == 'last_steps'
            # Step 3: prefix=3, last middle residual from step 2 = 5;
            # 3+5-4=4. Suffix is still executed, no stale full model output.
            assert outputs[2][0,0,0].item() == 4 + request * 20
        assert cache.previous is cache.residual is cache.groups is None


def test_mutating_blocks_do_not_corrupt_boundaries_or_residuals():
    cache = DBCache(runtime())
    blocks = [lambda x: x.add_(1), lambda x: x.mul_(2), lambda x: x.add_(3), lambda x: x.sub_(4)]
    with cache.request(config(), total_blocks=4):
        groups(cache)
        for i in range(3):
            value = advance(cache, torch.full((1,4,2), float(i)), blocks)
        assert value[0,0,0].item() == 4
        assert torch.equal(cache.previous, torch.ones_like(cache.previous))
        assert torch.equal(cache.residual, torch.full_like(cache.residual, 5))


def test_audio_gate_nonfinite_and_missing_cache_fail_closed():
    cache = DBCache(runtime())
    with cache.request(config(), total_blocks=4):
        groups(cache)
        cache.previous = torch.ones(1,4,2)
        cache.residual = torch.ones(1,4,2)
        cache.residual_finite = torch.tensor(True)
        current = torch.ones(1,4,2)
        current[:,3] = 1.2
        reuse, scores = cache.decision(current)
        assert not reuse and scores['audio'] > .19
        assert scores['video'] == 0
        current[:,3] = float('nan')
        reuse, scores = cache.decision(current)
        assert not reuse and scores['audio'] is None
        json.dumps(scores, allow_nan=False)
        cache.residual_finite = torch.tensor(False)
        assert not cache.decision(torch.ones(1,4,2))[0]
        cache.previous = None
        assert not cache.decision(torch.ones(1,4,2))[0]


def test_cache_released_on_error_and_nested_scope_rejected():
    cache = DBCache(runtime())
    with pytest.raises(RuntimeError, match='boom'):
        with cache.request(config(), total_blocks=4):
            groups(cache)
            advance(cache, torch.ones(1,4,2))
            with pytest.raises(RuntimeError, match='concurrent'):
                with cache.request(config(), total_blocks=4):
                    pass
            raise RuntimeError('boom')
    assert not cache.active_request and cache.previous is cache.residual is cache.groups is None


@pytest.mark.parametrize('kw', [{'cache_dit': 'true'}, {'cache_dit_threshold': True},
    {'cache_dit_threshold': float('nan')}, {'cache_dit_threshold': float('inf')},
    {'cache_dit_threshold': -1}, {'cache_dit_threshold': 1.1}, {'cache_dit_fn_blocks': 0},
    {'cache_dit_fn_blocks': 42, 'cache_dit_bn_blocks': 8}, {'cache_dit_bn_blocks': -1},
    {'cache_dit_warmup_steps': 0}, {'cache_dit_warmup_steps': 8},
    {'cache_dit_max_consecutive': 0}, {'cache_dit_max_cached_steps': 8}, {'cache_dit_last_steps': 8}])
def test_invalid_parameters(kw):
    with pytest.raises(ValueError):
        Settings(**kw).validate()


def test_options_reach_api_graph_ui_and_no_longer_require_restart():
    body = {'prompt': 'test', 'reference_image_url': 'https://example.com/a.png',
            'profile': True, 'softmax_ranks': 4, 'cache_dit': True, 'cache_dit_threshold': .04,
            'cache_dit_fn_blocks': 12, 'cache_dit_bn_blocks': 4, 'cache_dit_last_steps': 2}
    normalized, settings = normalize_request(body)
    graph = prompt_graph(normalized)['1']['inputs']
    for name in FIELDS:
        assert graph[name] == getattr(settings, name)
        for cls in (OpenVDNH200Generate, OpenVDNH200Request):
            assert name in cls.INPUT_TYPES()['optional']
    state = {'ready':True, 'profile':{name: getattr(Settings(),name) for name in PROFILE_FIELDS}}
    validate_profile(settings, state)
    validate_profile(Settings(profile=False,softmax_ranks=0),state)
    with pytest.raises(ValueError):
        validate_profile(Settings(fp8=False),state)


def _distributed(rank, world_size, init, output):
    dist.init_process_group('gloo', init_method=init, rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=25))
    try:
        cache = DBCache(runtime(rank,world_size))
        with cache.request(config(), total_blocks=4):
            # Unequal sizes: rank 0 owns 3 video rows, rank 1 owns 1 audio row.
            count = 3 if rank == 0 else 1
            cache.groups = [torch.tensor([],dtype=torch.long) for _ in range(4)]
            cache.groups[2 if rank == 0 else 3] = torch.arange(count)
            cache.previous = torch.ones(1,count,2)
            cache.residual = torch.ones_like(cache.previous)
            cache.residual_finite = torch.tensor(True)
            outcomes = []
            for mode in ('high_audio','small_change','missing_rank','bad_residual'):
                current = torch.ones_like(cache.previous) if cache.previous is not None else torch.ones(1,count,2)
                if mode == 'high_audio' and rank == 1:
                    current += .2
                if mode == 'small_change':
                    current += .01
                if mode == 'missing_rank' and rank == 1:
                    cache.previous = None
                if mode == 'bad_residual':
                    cache.previous = torch.ones_like(current)
                    cache.residual_finite = torch.tensor(rank == 0)
                outcomes.append(cache.decision(current))
        # Also execute a complete 8-step sequence whose toy blocks each perform
        # a real collective. Skips must be identical or the group cannot finish.
        def block(value):
            sync = torch.ones(1)
            dist.all_reduce(sync)
            return value + sync.item()
        with cache.request(config(), total_blocks=4):
            groups(cache)
            for i in range(8):
                advance(cache, torch.full((1,4,2), float(i+rank)), [block]*4)
            report = cache.report()
            assert report['cached_steps'] == [3,5] and report['executed_blocks'] == 28
            outcomes.append(report['cached_steps'])
        Path(output,f'{rank}.json').write_text(json.dumps(outcomes,allow_nan=False))
    finally:
        dist.destroy_process_group()


def test_real_two_rank_gloo_agrees_when_one_shard_rejects(tmp_path):
    # Actual collectives in subprocesses, with a timeout: no mocks that can hide
    # rank-dependent branching/deadlocks. CUDA/NCCL still needs the H200 run.
    init = (tmp_path/'group').as_uri()
    procs = [subprocess.Popen([sys.executable,__file__,'--rank',str(rank),init,str(tmp_path)]) for rank in range(2)]
    try:
        for p in procs:
            assert p.wait(timeout=40) == 0
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill();p.wait()
    a,b = [json.loads((tmp_path/f'{rank}.json').read_text()) for rank in range(2)]
    assert a == b
    assert [entry[0] for entry in a[:4]] == [False,True,False,False]
    assert a[4] == [3,5]


def test_profile_summary_exposes_nested_scopes_without_invented_speedup():
    records = [{'rank':i,'role':'softmax' if i < 6 else 'linear','profile_enabled':True,
                'ms_per_nfe': {'blocks':100.,'attention':70.,'ffn':20.,
                               'softmax_compute' if i < 6 else 'linear_compute':40. if i < 6 else 25.,
                               'output_dispatch':10. if i < 6 else 22.}}
               for i in range(8)]
    summary = summarize_profiles(records)
    assert summary['max_ms_per_nfe']['blocks'] == 100
    assert summary['branches']['linear']['max_return_dispatch_ms_per_nfe'] == 22
    assert not summary['pure_nccl_kernel_time']
    with pytest.raises(RuntimeError,match='diverged'):
        summarize_cache([{'config':{},'cached_steps':[3]}, {'config':{},'cached_steps':[4]}])


if __name__ == '__main__':
    _distributed(int(sys.argv[2]),2,sys.argv[3],sys.argv[4])


def test_benchmark_pairs_uncached_layouts_with_same_case_cache_variants(tmp_path):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'scripts/benchmark_optimizations.py'
    spec = importlib.util.spec_from_file_location('benchmark_under_test',path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    requests = []
    class Client:
        base = 'http://test'
        def http(self,path):
            return {'ready':True,'instance':'stable','metrics_schema_version':5,'request_options':{'profile':True}}
        def run(self,body,folder,instance):
            requests.append(dict(body))
            assert instance == 'stable' and body['seed'] == 42 and body['prompt'] == 'unchanged'
            timing = {'denoise_seconds': body['softmax_ranks'] - int(body['cache_dit']),
                      'video_vae_decode_seconds':1.,'audio_vae_decode_seconds':.1,'processing_wall_seconds':10.}
            return {'job_id':str(len(requests)),'video_url':'/view?filename=test.mp4',
                    'metrics':{'timings':timing,'upstream':{'cache_dit':{'cache_hits':int(body['cache_dit'])},
                                                        'compilation':{'compiled_new_graph':False}}}}
    rows = module.benchmark(Client(),{'prompt':'unchanged','seed':42},tmp_path,[6,4],[.04,.08],repeat=2)
    assert len(rows) == 4
    assert all(row['runs'] == 2 for row in rows)
    cached = [r for r in requests if r['cache_dit']]
    assert len(cached) == 6 and all(r['softmax_ranks'] == 4 and not r['profile'] for r in cached)
    assert sum(r['profile'] for r in requests) == 2
    assert (tmp_path/'report.md').is_file()
    with pytest.raises(ValueError, match='already has a run'):
        module.benchmark(Client(),{},tmp_path)
