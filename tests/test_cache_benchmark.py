import json

import pytest

from scripts.benchmark_compile_cache import benchmark


def test_thirteen_original_requests_are_unchanged_and_disk_hits_are_not_hot_hits(tmp_path):
    files = []
    for index in range(13):
        path = tmp_path / 'requests' / f'case-{index:02}' / 'request.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'prompt': f'Original template {index}', 'duration': 10,
                                    'ratio': '9:16', 'resolution': 768, 'reference_short_edge': 512,
                                    'cache_dit': True, 'cache_dit_threshold': .25,
                                    'reference_image_urls': [f'https://example.com/{index}.png']}))
        files.append(path)
    class Client:
        base = 'http://example.test'
        calls = []
        def http(self, path):
            return {'ready': True, 'metrics_schema_version': 6, 'instance': 'fixed'}
        def run(self, body, folder, instance):
            self.calls.append(body)
            hit = len(self.calls) != 4
            return {'request': body, 'job_id': str(len(self.calls)), 'video_url': '/view?file=result.mp4',
                'metrics': {'metrics_schema_version': 6, 'timings': {
                    'conditioning_seconds': 0, 'denoise_wall_seconds': 11 if hit else 25,
                    'hot_denoise_seconds': 11 if hit else None, 'generation_wall_seconds': 16 if hit else 30},
                    'upstream': {'actual_geometry': {'validated': True, 'generation_width': 768, 'generation_height': 1376},
                        'render_plan': {'generation_width': 768, 'generation_height': 1376},
                        'cache_dit': {'cache_hits': 2}, 'compilation': {'runtime_graph_reused': hit,
                            'compiled_new_graph': not hit, 'dynamo_compile_seconds': 0 if hit else 14,
                            'disk_graph_cache_hits': 8, 'geometry_id': 'bucket',
                            'token_bucket': {'prefix_capacity': 2048, 'padding_tokens': 512}}}}}
    client = Client()
    report = benchmark(client, files, tmp_path / 'output')
    assert len(client.calls) == 26
    original = [json.loads(path.read_text()) for path in files]
    assert client.calls[:13] == client.calls[13:] == original
    assert report['cases'] == 13 and report['graph_hits'] == 25
    assert not report['all_graphs_reused']
    assert report['misses'][0]['case'] == 'case-03'
    assert (tmp_path / 'output/timings.csv').is_file()


def test_old_server_rejected_before_submitting_jobs(tmp_path):
    class Old:
        def http(self, path):
            return {'ready': True, 'metrics_schema_version': 5}
    with pytest.raises(RuntimeError, match='schema 6'):
        benchmark(Old(), [], tmp_path)
