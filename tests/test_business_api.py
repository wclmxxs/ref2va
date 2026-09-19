import asyncio
import base64
import io
import json
from pathlib import Path
import socket
import sys
import time
import types
import uuid

from aiohttp import web
from aiohttp.abc import AbstractResolver
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image
import pytest

from openvdn_comfy import api, business_api, business_contract as contract, business_media as media, jobs, nodes
from openvdn_comfy.config import Settings


def inline_image(size=(48, 80), color='red', fmt='PNG', orientation=None):
    data = io.BytesIO()
    image = Image.new('RGB', size, color)
    kwargs = {}
    if orientation:
        exif = Image.Exif(); exif[274] = orientation
        kwargs['exif'] = exif
    image.save(data, format=fmt, **kwargs)
    return base64.b64encode(data.getvalue()).decode()


def body(**changes):
    return {'model': 'MiniMax-H3', 'content': [
        {'type': 'text', 'text': 'Use <Picture 1> as the character reference.'},
        {'type': 'image_url', 'role': 'reference_image', 'image_url': {'base64': inline_image()}}],
        'resolution': '768P', 'duration': 10, 'ratio': '9:16', 'seed': 42, **changes}


@pytest.fixture
def environment(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, 'RUNTIME', tmp_path/'runtime')
    monkeypatch.setattr(media, 'RUNTIME', tmp_path/'runtime')
    monkeypatch.delenv('PUBLIC_BASE_URL', raising=False)
    monkeypatch.delenv('REF2VA_SYNC_TIMEOUT_SECONDS', raising=False)
    monkeypatch.delenv('BUSINESS_MODEL', raising=False)
    settings = Settings()
    state = {'ready': True, 'instance': 'worker-1',
             'profile': {k: getattr(settings, k) for k in (*api.PROFILE_FIELDS, 'softmax_ranks', 'profile')},
             'request_options': {'optimizations': {'fast_communication': True},
                                 'cache_dit': {k: getattr(settings, k) for k in api.CACHE_FIELDS}}}
    monkeypatch.setattr(api, 'health', lambda: state)
    monkeypatch.setitem(sys.modules, 'folder_paths', types.SimpleNamespace(get_output_directory=lambda: str(tmp_path/'output')))

    class Queue:
        def __init__(self):
            self.pending = []
            self.on_put = None
        def put(self, item):
            self.pending.append(item)
            if self.on_put:
                self.on_put(item[1])
        def get_history(self, job_id):
            return {}
        def get_current_queue_volatile(self):
            return [], self.pending

    async def validate(job_id, graph, target):
        assert graph == {'1': {'class_type': 'OpenVDNH200BusinessRequest', 'inputs': {'job_id': job_id}}}
        return True, None, ['1'], {}

    monkeypatch.setitem(sys.modules, 'execution', types.SimpleNamespace(validate_prompt=validate))
    server = types.SimpleNamespace(routes=web.RouteTableDef(), number=0, prompt_queue=Queue())
    api.register_routes(server)

    def client():
        app = web.Application()
        app.add_routes(server.routes)
        return TestClient(TestServer(app))

    def finish(job_id, status='succeeded'):
        output = tmp_path/'output'/'openvdn'/f'{job_id}.mp4'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b'0123456789abcdef')
        jobs.update_job(job_id, status=status, phase='complete', error='test failure' if status == 'failed' else None,
                        metrics={'output': str(output), 'timings': {'denoise_seconds': 11.2, 'worker_wall_seconds': 14.6},
                                 'upstream': {'compilation': {'runtime_graph_reused': True},
                                              'cache_dit': {'cached_steps': [4, 6]}}})
    return types.SimpleNamespace(client=client, server=server, state=state, finish=finish, root=tmp_path)


def test_complete_optimization_mapping_and_request_defaults(environment):
    state = environment.state
    original = json.dumps(state, sort_keys=True)
    optimization = {'cache_dit': {'enabled': True, 'warmup': 3, 'rdt': .25,
        'max_continuous_cached_steps': 1, 'fn_blocks': 8, 'bn_blocks': 8, 'max_cached_steps': 2, 'last_steps': 1},
        'attention_kernel': 'native', 'isolate_padding': False, 'linear_stats_chunk_frames': 16, 'linear_kv_keep_ratio': .5,
        'softmax_ranks': 6, 'fast_communication': True, 'streaming_output': True,
        'cleanup_policy': 'adaptive', 'profile': False, 'profile_kernels': False,
        'fused_delta': True, 'boundary_scan': True, 'fast_softmax': True, 'dual_stream': False}
    request, settings, sources = contract.normalize_request(body(optimization=optimization, reference_short_edge=512), state)
    assert request['seed'] == 42 and request['reference_short_edge'] == 512
    for new, old in contract.CACHE_MAPPING.items():
        assert getattr(settings, old) == optimization['cache_dit'][new]
    for key in contract.OPTIMIZATIONS:
        assert getattr(settings, key) == optimization[key]
    assert json.dumps(state, sort_keys=True) == original
    assert not contract.normalize_request(body(), state)[1].cache_dit
    state['request_options']['cache_dit'].update(cache_dit=True, cache_dit_threshold=.15)
    _, inherited, _ = contract.normalize_request(body(optimization={'cache_dit': {'rdt': None}}), state)
    assert inherited.cache_dit and inherited.cache_dit_threshold == .15


def test_documented_example_covers_every_business_parameter(environment):
    path = Path(__file__).resolve().parents[1]/'examples/business-request.json'
    example = json.loads(path.read_text())
    assert set(example) == set(contract.FIELDS)
    assert set(example['optimization']) == {*contract.OPTIMIZATIONS, 'cache_dit'}
    assert set(example['optimization']['cache_dit']) == set(contract.CACHE_MAPPING)
    request, settings, sources = contract.normalize_request(example, environment.state)
    assert settings.cache_dit and settings.cache_dit_threshold == .25
    assert request['num_inference_steps'] == 8 and sources[0]['kind'] == 'url'


def test_linear_kv_ratio_is_request_local_and_rejected_before_queueing(environment):
    async def exercise():
        async with environment.client() as client:
            for ratio in (True, 0, .75, '0.5', float('nan')):
                response = await client.post(contract.PREFIX+'/video_generation',
                                             json=body(optimization={'linear_kv_keep_ratio': ratio}))
                assert response.status == 400, await response.text()
            assert not environment.server.prompt_queue.pending
            for optimization, expected in (({'linear_kv_keep_ratio': .5}, .5),
                                           ({'linear_kv_keep_ratio': .25}, .25), ({}, 1.)):
                response = await client.post(contract.PREFIX+'/video_generation', json=body(optimization=optimization))
                assert response.status == 200, await response.text()
                task_id = (await response.json())['task_id']
                internal = contract.job_id(task_id)
                assert jobs.read_job(internal)['settings']['linear_kv_keep_ratio'] == expected
                jobs.update_job(internal, status='succeeded', metrics={
                    'timings': {'linear_frame_statistics_seconds': .2},
                    'upstream': {'optimizations': {'linear_kv': {'requested_keep_ratio': expected}}}})
                response = await client.post(contract.PREFIX+'/query/video_generation',
                                             json={'model': 'MiniMax-H3', 'task_id': task_id})
                assert response.status == 200, await response.text()
                task = (await response.json())['task']
                assert task['optimizations']['linear_kv']['requested_keep_ratio'] == expected
                assert task['timings']['linear_frame_statistics_seconds'] == .2
    asyncio.run(exercise())


def test_large_valid_inline_body_exceeds_aiohttp_default_one_mib(environment):
    stream = io.BytesIO()
    Image.new('RGB', (640, 768), 'red').save(stream, format='PNG', compress_level=0)
    data = body();data['content'][1]['image_url'] = {'base64': base64.b64encode(stream.getvalue()).decode()}
    assert len(json.dumps(data)) > 1024*1024
    async def exercise():
        async with environment.client() as client:
            response = await client.post(contract.PREFIX+'/video_generation', json=data)
            assert response.status == 200, await response.text()
    asyncio.run(exercise())


def test_url_preparation_redirects_and_media_errors_before_queueing(environment, monkeypatch):
    # A test-only DNS resolver routes a public-looking fixture host to our local HTTP fixture.
    class FixtureResolver(AbstractResolver):
        async def resolve(self, host, port=0, family=socket.AF_INET):
            return [{'hostname': host, 'host': '127.0.0.1', 'port': port,
                     'family': socket.AF_INET, 'proto': 0, 'flags': 0}]
        async def close(self):
            pass
    monkeypatch.setattr(media, 'PublicResolver', FixtureResolver)
    async def exercise():
        image_bytes = base64.b64decode(inline_image((80, 48), 'blue'))
        app = web.Application()
        async def image(request):
            return web.Response(body=image_bytes, content_type='image/png')
        async def redirect(request):
            raise web.HTTPFound('/image')
        async def private(request):
            raise web.HTTPFound('http://127.0.0.1/private')
        async def damaged(request):
            return web.Response(body=b'broken png', content_type='image/png')
        async def oversized(request):
            return web.Response(headers={'Content-Length': str(20*1024*1024+1)})
        for path, handler in [('/image', image), ('/redirect', redirect), ('/private', private),
                              ('/damaged', damaged), ('/oversized', oversized)]:
            app.router.add_get(path, handler)
        async with TestServer(app) as source, environment.client() as client:
            for path, expected in [('/redirect', 200), ('/private', 400), ('/damaged', 400), ('/oversized', 413)]:
                data = body(ratio='adaptive')
                url = f'http://media.example.test:{source.port}{path}'
                data['content'][1]['image_url'] = {'url': url, 'base64': '  '}
                response = await client.post(contract.PREFIX+'/video_generation', json=data)
                assert response.status == expected, await response.text()
                if expected == 200:
                    internal = contract.job_id((await response.json())['task_id'])
                    record = jobs.read_job(internal)
                    assert record['settings']['ratio'] == '5:3'
                    assert record['request']['reference_images'][0]['url'] == url
            assert len(environment.server.prompt_queue.pending) == 1
    asyncio.run(exercise())


@pytest.mark.parametrize('changes', [
    {'num_inference_steps': 6}, {'num_inference_steps': True}, {'seed': -1}, {'seed': True},
    {'resolution': 768}, {'resolution': '769P'}, {'resolution': '2160P'},
    {'duration': True}, {'duration': 16}, {'duration': None}, {'ratio': '1:9'},
    {'reference_short_edge': 513}, {'content': []}, {'model': ''}, {'unknown': 1},
    {'optimization': {'sol_attn': {'enabled': True}}},
    {'optimization': {'cache_dit': {'rdt': float('nan')}}},
    {'optimization': {'cache_dit': {'warmup': 0}}},
    {'optimization': {'cache_dit': {'fn_blocks': 49, 'bn_blocks': 1}}},
    {'optimization': {'cache_dit': {'max_continuous_cached_steps': 0}}},
    {'optimization': {'attention_kernel': 'decomposed', 'isolate_padding': True}},
    {'optimization': {'fast_communication': 1}}, {'optimization': {'profile': 'true'}}])
def test_invalid_parameters_fail_before_queueing(environment, changes):
    with pytest.raises(ValueError):
        contract.normalize_request(body(**changes), environment.state)
    assert not environment.server.prompt_queue.pending


def test_http_submission_query_order_inline_privacy_and_range_download(environment, monkeypatch):
    async def exercise():
        async with environment.client() as client:
            data = body(ratio='adaptive', reference_short_edge=512,
                        optimization={'cache_dit': {'enabled': True, 'rdt': .25}})
            data['content'][1]['image_url']['url'] = 'http://127.0.0.1/ignored-when-inline'
            data['content'].append({'type': 'image_url', 'role': 'reference_image',
                                    'image_url': {'base64': inline_image((80, 48), 'blue')}})
            response = await client.post(contract.PREFIX+'/video_generation', json=data)
            assert response.status == 200, await response.text()
            result = await response.json()
            assert list(result) == ['task_id']
            task_id = result['task_id']; job_id = contract.job_id(task_id)
            record = jobs.read_job(job_id)
            assert record['settings']['ratio'] == '3:5'
            assert record['render_plan']['width'] == 768 and record['render_plan']['height'] == 1280
            assert record['settings']['cache_dit_threshold'] == .25
            assert [Image.open(p).size for p in record['resolved_references']] == [(48, 80), (80, 48)]
            assert Image.open(record['resolved_references'][0]).getpixel((0, 0)) == (255, 0, 0)
            serialized = json.dumps(record)
            assert data['content'][1]['image_url']['base64'] not in serialized
            assert 'ignored-when-inline' not in serialized
            assert all('sha256' in m and 'bytes' in m for m in record['request']['reference_images'])
            assert environment.server.prompt_queue.pending[0][2]['1']['inputs'] == {'job_id': job_id}
            query = {'model': 'alias', 'task_id': task_id}
            task = (await (await client.post(contract.PREFIX+'/query/video_generation', json=query)).json())['task']
            assert task['status'] == 'queued' and task['seed'] == 42 and task['inference_time_s'] is None
            assert 'content' not in task
            content_path = contract.PREFIX+f'/video_generation/{task_id}/content'
            assert (await client.get(content_path)).status == 409
            environment.finish(job_id)
            monkeypatch.setenv('PUBLIC_BASE_URL', 'https://video.example.com/')
            task = (await (await client.post(contract.PREFIX+'/query/video_generation', json=query)).json())['task']
            assert task['inference_time_s'] == 14.6
            assert task['timings']['denoise_seconds'] == 11.2 and task['compilation']['runtime_graph_reused']
            assert task['content']['url'] == 'https://video.example.com'+content_path
            video = await client.get(content_path, headers={'Range': 'bytes=2-7'})
            assert video.status == 206 and await video.read() == b'234567'
            assert video.headers['Content-Type'] == 'video/mp4'
            outside = environment.root/'outside.mp4';outside.write_bytes(b'private')
            jobs.update_job(job_id, metrics={'output': str(outside)})
            assert (await client.get(content_path)).status == 404
    asyncio.run(exercise())


def test_http_errors_unready_restart_and_body_limit(environment):
    async def exercise():
        async with environment.client() as client:
            invalid = [body(num_inference_steps=4), body(optimization={'cache_dit': {'rdt': -1}})]
            for value in ('not base64', 'data:image/png;base64,invalid', 'file:///etc/passwd'):
                data = body();data['content'][1]['image_url'] = {'base64': value, 'url': 'https://example.com/no-fallback'}
                invalid.append(data)
            data = body();data['content'][1]['role'] = 'first_frame';invalid.append(data)
            data = body();data['content'][1]['image_url'] = {'url': 'http://127.0.0.1/private', 'base64': ''};invalid.append(data)
            for data in invalid:
                response = await client.post(contract.PREFIX+'/video_generation', json=data)
                assert response.status == 400, await response.text()
                assert (await response.json())['error']['type'] == 'invalid_request_error'
            assert not environment.server.prompt_queue.pending
            assert (await client.post(contract.PREFIX+'/query/video_generation', data=b'x'*131073)).status == 413
            for value, status in [('bad-id', 400), ('video_'+uuid.uuid4().hex, 404)]:
                response = await client.post(contract.PREFIX+'/query/video_generation', json={'model': 'MiniMax-H3', 'task_id': value})
                assert response.status == status
            task_id = (await (await client.post(contract.PREFIX+'/video_generation', json=body())).json())['task_id']
            environment.server.prompt_queue.pending.clear()
            task = (await (await client.post(contract.PREFIX+'/query/video_generation', json={'model': 'MiniMax-H3', 'task_id': task_id})).json())['task']
            assert task['status'] == 'cancelled'
            environment.state['ready'] = False
            response = await client.post(contract.PREFIX+'/video_generation', json=body())
            assert response.status == 503 and (await response.json())['error']['http_code'] == 503
    asyncio.run(exercise())


def test_sync_success_failure_and_timeout_retains_existing_task(environment, monkeypatch):
    async def exercise():
        async with environment.client() as client:
            environment.server.prompt_queue.on_put = environment.finish
            for endpoint in ('/sync_infer', contract.PREFIX+'/sync_infer'):
                response = await client.post(endpoint, json=body())
                assert response.status == 200
                assert (await response.json())['task']['status'] == 'succeeded'
            environment.server.prompt_queue.on_put = lambda job_id: environment.finish(job_id, 'failed')
            response = await client.post('/sync_infer', json=body())
            assert response.status == 500 and (await response.json())['task']['error']['message'] == 'test failure'
            environment.server.prompt_queue.on_put = None
            monkeypatch.setenv('REF2VA_SYNC_TIMEOUT_SECONDS', '.01')
            response = await client.post('/sync_infer', json=body())
            assert response.status == 504
            result = await response.json()
            response = await client.post(contract.PREFIX+'/query/video_generation', json={'model': 'MiniMax-H3', 'task_id': result['task_id']})
            assert (await response.json())['task']['status'] == 'queued'
            assert len(environment.server.prompt_queue.pending) == 4
    asyncio.run(exercise())


def test_seed_resolution_and_concurrent_submissions_have_distinct_ids(environment, monkeypatch):
    seeds = iter([123456, 987654])
    monkeypatch.setattr(contract.secrets, 'randbits', lambda bits: next(seeds))
    async def exercise():
        async with environment.client() as client:
            missing = body();del missing['seed']
            responses = await asyncio.gather(*(client.post(contract.PREFIX+'/video_generation', json=b)
                                               for b in (missing, body(seed=None), body(seed=42))))
            ids = [(await r.json())['task_id'] for r in responses]
            assert len(set(ids)) == 3
            actual = sorted(jobs.read_job(contract.job_id(value))['business']['seed'] for value in ids)
            assert actual == [42, 123456, 987654]
            assert sorted(q[0] for q in environment.server.prompt_queue.pending) == [0, 1, 2]
    asyncio.run(exercise())


def test_prepared_node_reaches_existing_worker_and_separates_input_time_from_queue(environment, monkeypatch):
    current = types.SimpleNamespace(prompt_id=None)
    monkeypatch.setitem(sys.modules, 'comfy_execution.utils', types.SimpleNamespace(get_executing_context=lambda: current))
    mm = types.SimpleNamespace(throw_exception_if_processing_interrupted=lambda: None)
    monkeypatch.setitem(sys.modules, 'comfy', types.SimpleNamespace(model_management=mm))
    monkeypatch.setitem(sys.modules, 'comfy.model_management', mm)
    monkeypatch.setitem(sys.modules, 'comfy_api.input_impl', types.SimpleNamespace(VideoFromFile=lambda path: path))
    captured = []
    def generate(**kwargs):
        captured.append(kwargs)
        assert kwargs['settings'].reference_short_edge == 512
        assert kwargs['settings'].cache_dit and kwargs['settings'].cache_dit_threshold == .25
        assert len(kwargs['refs']) == 1 and Path(kwargs['refs'][0]).is_file()
        kwargs['progress']('denoising')
        output = kwargs['output'];output.parent.mkdir(parents=True, exist_ok=True);output.write_bytes(b'mp4')
        log = environment.root/'logs';log.mkdir()
        value = {'metrics_schema_version': 9, 'output': str(output), 'log_directory': str(log),
                'timings': {'denoise_seconds': 11., 'worker_wall_seconds': 14.}, 'upstream': {}}
        assert kwargs['defer_output'] is True
        return types.SimpleNamespace(finish=lambda: value)
    monkeypatch.setattr(nodes, 'generate', generate)
    async def exercise():
        async with environment.client() as client:
            response = await client.post(contract.PREFIX+'/video_generation', json=body(reference_short_edge=512,
                optimization={'cache_dit': {'enabled': True, 'rdt': .25}}))
            internal = contract.job_id((await response.json())['task_id'])
            current.prompt_id = internal
            jobs.update_job(internal, created_at=time.time()-10, queued_at=time.time()-1, reference_prepare_seconds=9.)
            result = await nodes.OpenVDNH200BusinessRequest().generate(internal)
            for _ in range(100):
                record = jobs.read_job(internal)
                if record['status'] != 'running':
                    break
                await asyncio.sleep(.01)
            assert record['status'] == 'succeeded'
            assert len(captured) == 1
            timings = record['metrics']['timings']
            assert .9 < timings['api_queue_seconds'] < 2
            assert timings['input_prepare_seconds'] == 9 and timings['processing_wall_seconds'] >= 9
            assert timings['api_wall_seconds'] >= 10
            assert result['result'][0] == internal
            with pytest.raises(ValueError, match='already executed'):
                await nodes.OpenVDNH200BusinessRequest().generate(internal)
            current.prompt_id = str(uuid.uuid4())
            with pytest.raises(ValueError, match='original queued task'):
                await nodes.OpenVDNH200BusinessRequest().generate(internal)
    asyncio.run(exercise())


def test_inline_format_limit_and_exif_adaptive_orientation(environment, monkeypatch):
    data = base64.b64decode(inline_image(fmt='GIF'))
    with pytest.raises(ValueError, match='JPEG, PNG or WebP'):
        media.inspect_image(data)
    monkeypatch.setattr(media, 'MAX_PIXELS', 100)
    with pytest.raises(media.ImageTooLarge):
        media.inspect_image(base64.b64decode(inline_image()))
    monkeypatch.setattr(media, 'MAX_PIXELS', 40_000_000)
    monkeypatch.setattr(media, 'MAX_BASE64_CHARS', 8)
    with pytest.raises(media.ImageTooLarge):
        media.decode_inline('A'*12)
    monkeypatch.setattr(media, 'MAX_BASE64_CHARS', 4*((20*1024*1024+2)//3))
    async def exercise():
        value = inline_image((80, 48), fmt='JPEG', orientation=6)
        data = body(ratio=None);data['content'][1]['image_url'] = {'url': 'data:image/jpeg;base64,'+value}
        req, settings, sources = contract.normalize_request(data, environment.state)
        paths, metadata = await media.prepare_sources(sources)
        assert Image.open(paths[0]).size == (48, 80)
        assert contract.resolve_geometry(req, settings, metadata).ratio == '3:5'
    asyncio.run(exercise())


@pytest.mark.parametrize('encoder_fails', [False, True])
def test_api_starts_next_gpu_job_while_previous_cpu_output_is_pending(environment, monkeypatch, encoder_fails):
    import threading
    current = types.SimpleNamespace(prompt_id=None)
    monkeypatch.setitem(sys.modules, 'comfy_execution.utils', types.SimpleNamespace(get_executing_context=lambda: current))
    mm = types.SimpleNamespace(throw_exception_if_processing_interrupted=lambda: None)
    monkeypatch.setitem(sys.modules, 'comfy', types.SimpleNamespace(model_management=mm))
    monkeypatch.setitem(sys.modules, 'comfy.model_management', mm)
    monkeypatch.setitem(sys.modules, 'comfy_api.input_impl', types.SimpleNamespace(VideoFromFile=lambda path: path))
    release, waiting = threading.Event(), threading.Event()
    gpu_calls = []
    def generate(**kwargs):
        index = len(gpu_calls)
        gpu_calls.append(kwargs)
        def finish():
            if index == 0:
                waiting.set()
                assert release.wait(5)
                if encoder_fails:
                    raise RuntimeError('CPU output failed: encoder test')
            output = kwargs['output']; output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(b'mp4')
            log = environment.root/f'logs-{index}'; log.mkdir()
            return {'output': str(output), 'log_directory': str(log), 'timings': {'worker_wall_seconds': 1.}, 'upstream': {}}
        return types.SimpleNamespace(finish=finish)
    monkeypatch.setattr(nodes, 'generate', generate)
    async def exercise():
        async with environment.client() as client:
            ids = [(await (await client.post(contract.PREFIX+'/video_generation', json=body())).json())['task_id']
                   for _ in range(2)]
            internal = [contract.job_id(t) for t in ids]
            try:
                for job in internal:
                    current.prompt_id = job
                    result = await asyncio.wait_for(nodes.OpenVDNH200BusinessRequest().generate(job), timeout=2)
                    assert result['result'] == (job,)
                    environment.server.prompt_queue.pending[:] = [p for p in environment.server.prompt_queue.pending if p[1] != job]
                assert len(gpu_calls) == 2 and waiting.wait(1) and not release.is_set()
                response = await client.post(contract.PREFIX+'/query/video_generation', json={'model': 'MiniMax-H3', 'task_id': ids[0]})
                task = (await response.json())['task']
                assert task['status'] == 'running' and task['phase'] == 'encoding_output' and 'content' not in task
                assert (await client.get(contract.PREFIX+f'/video_generation/{ids[0]}/content')).status == 409
            finally:
                release.set()
            for _ in range(100):
                records = [jobs.read_job(job) for job in internal]
                if all(r['status'] in ('succeeded', 'failed') for r in records): break
                await asyncio.sleep(.01)
            assert records[0]['status'] == ('failed' if encoder_fails else 'succeeded')
            assert records[1]['status'] == 'succeeded' and environment.state['ready']
    asyncio.run(exercise())
