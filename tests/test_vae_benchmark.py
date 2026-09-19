import copy
import json
from types import SimpleNamespace

from scripts.benchmark_vae import benchmark, hot_status, summarize


def test_vae_only_ablation_preserves_case_and_resumes_without_duplicate_posts(tmp_path):
    health = {'ready': True, 'instance': 'fixed', 'metrics_schema_version': 13,
              'profile': {'softmax_ranks': 2}, 'request_options': {'optimizations': {'fused_delta': True}}}
    posted = {}
    def http(path, body=None):
        if path == '/openvdn/health':
            return copy.deepcopy(health)
        if path.endswith('/query/video_generation'):
            request = posted[body['task_id']]
            opt = request['optimization']
            return {'task': {'id': body['task_id'], 'status': 'succeeded',
                            'optimizations': {'requested': opt}, 'content': {'url': 'http://test/video.mp4'},
                            'compilation': {'runtime_graph_reused': True},
                            'cache_dit': {'cached_steps': [4, 6]},
                            'timings': {'denoise_seconds': 10., 'video_vae_decode_seconds': 3., 'worker_wall_seconds': 14.},
                            'video_vae_decode': {'compilation': {'runtime_graph_reused': True},
                                                 'tile_decoder_by_rank': [{'tile_batch_size': opt['vae_tile_batch_size'],
                                                                          'compile_enabled': opt['vae_compile'],
                                                                          'verification': {'new_checks': []}}]}}}
        key = str(len(posted) + 1)
        posted[key] = body
        return {'task_id': key}
    client = SimpleNamespace(http=http, timeout=1)
    body = {'model': 'MiniMax-H3', 'seed': 42, 'duration': 10, 'ratio': '9:16', 'resolution': '768P',
            'content': [{'type': 'text', 'text': 'same prompt'}],
            'optimization': {'softmax_ranks': 0, 'dual_stream': True, 'cache_dit': {'enabled': True, 'rdt': .25}}}
    original = copy.deepcopy(body)
    rows = benchmark(client, body, tmp_path, repeat=2)
    assert body == original and len(posted) == 9
    assert all(r['all_hot'] and r['runs'] == 2 for r in rows)
    for request in posted.values():
        assert request['content'] == body['content'] and request['seed'] == 42
        assert all(request['optimization'][key] == value for key, value in body['optimization'].items())
    assert benchmark(client, body, tmp_path, repeat=2) == rows
    assert len(posted) == 9
    task = json.loads((tmp_path/'batch4_compiled-hot-1/status.json').read_text())['task']
    task['video_vae_decode']['tile_decoder_by_rank'][0]['verification']['new_checks'] = [{'passed': True}]
    assert not hot_status(task)
    task['video_vae_decode']['tile_decoder_by_rank'][0]['verification']['new_checks'] = []
    task['video_vae_decode']['compilation']['runtime_graph_reused'] = False
    assert not hot_status(task)
    invalid = {'kind': 'hot', 'variant': 'batch4_compiled', 'hot': False, 'video_url': 'x'}
    assert summarize([invalid])[0]['video_vae_decode_seconds'] is None
