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
    report = benchmark(client, files, tmp_path / 'output', allow_first_compile=False)
    assert len(client.calls) == 26
    original = [json.loads(path.read_text()) for path in files]
    assert client.calls[:13] == client.calls[13:] == original
    assert report['cases'] == 13 and report['graph_hits'] == 25
    assert not report['all_graphs_reused']
    assert report['misses'][0]['case'] == 'case-03'
    assert (tmp_path / 'output/timings.csv').is_file()
    client.calls = []
    lazy = benchmark(client, files, tmp_path / 'lazy-output')
    assert lazy['all_graphs_reused'] and lazy['checked_runs'] == lazy['graph_hits'] == 13
    assert client.calls[:13] == client.calls[13:] == original


def test_old_server_rejected_before_submitting_jobs(tmp_path):
    class Old:
        def http(self, path):
            return {'ready': True, 'metrics_schema_version': 5}
    with pytest.raises(RuntimeError, match='schema 6'):
        benchmark(Old(), [], tmp_path)


def test_pipeline_benchmark_preserves_content_and_excludes_cold_or_compiling_runs():
    from scripts.benchmark_attention_pipeline import VARIANTS, summarize
    from openvdn_comfy.optimization_options import FIELDS
    assert all(set(v)==set(FIELDS) for v in VARIANTS.values())
    rows=[{'variant':'baseline','pass':0,'graph_reused':False,'denoise_seconds':999,
           'output_seconds':2,'processing_seconds':1001,'cached_steps':[4,6],'video_url':'warm'},
          {'variant':'baseline','pass':1,'graph_reused':True,'denoise_seconds':11,
           'output_seconds':4,'processing_seconds':16,'cached_steps':[4,6],'video_url':'hot'},
          {'variant':'combined','pass':1,'graph_reused':False,'denoise_seconds':50,
           'output_seconds':3,'processing_seconds':54,'cached_steps':[4,6],'video_url':'compiled'}]
    summary=summarize(rows)
    assert summary[0]['denoise_seconds']==11 and summary[0]['videos']==['hot']
    assert summary[1]['denoise_seconds'] is None


def test_pipeline_benchmark_resumes_accepted_job_without_duplicate_post(tmp_path):
    import json
    from scripts.benchmark_attention_pipeline import sample
    request={'prompt':'unchanged'}
    (tmp_path/'request.json').write_text(json.dumps(request))
    (tmp_path/'submit.json').write_text(json.dumps({'status_url':'/openvdn/jobs/existing'}))
    class Client:
        timeout=5
        def http(self,path,body=None):
            assert body is None, 'Resume must not POST'
            return {'ready':True,'instance':'same'} if path.endswith('health') else {'status':'succeeded'}
    assert sample(Client(),request,tmp_path,'same')['status']=='succeeded'
    (tmp_path/'submit.json').unlink()
    import pytest
    with pytest.raises(RuntimeError,match='Uncertain POST'):
        sample(Client(),request,tmp_path,'same')
