from concurrent.futures import Future
from dataclasses import asdict
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import types

import psutil
import pytest

from openvdn_comfy import backend, gpu_check, runner, jobs, api
from openvdn_comfy.config import Settings, atomic_json
from openvdn_comfy.hardware import Hardware, runtime_directory, install_workflows
from openvdn_comfy.output_completion import OutputCompletions
from openvdn_comfy.supervision import WorkerWatchdog, Policy, fail_pending

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location('test_' + name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('gpu,size,backend_name,soft', [('h200',8,'flex',0),('h200',4,'flex',0),
                                                       ('b200',8,'decomposed',0),('b200',4,'decomposed',0),
                                                       ('b300',8,'decomposed',0),('b300',4,'decomposed',0)])
def test_all_hardware_profiles_and_rank_limits(monkeypatch, gpu, size, backend_name, soft):
    monkeypatch.setenv('REF2VA_GPU_TYPE',gpu)
    monkeypatch.setenv('REF2VA_GPUS',str(size))
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.delenv('REF2VA_SOFTMAX_RANKS',raising=False)
    monkeypatch.delenv('REF2VA_SOFTMAX_BACKEND',raising=False)
    h=Hardware.from_env()
    assert len(h.visible_devices()) == size
    assert (Settings().softmax_backend,Settings().softmax_ranks)==(backend_name,soft)
    assert backend.startup_settings().softmax_ranks==soft
    assert Settings(softmax_ranks=size-1).validate()
    with pytest.raises(ValueError): Settings(softmax_ranks=size).validate()
    from openvdn_comfy.hardware import GPU_SPECS
    capability, minimum = GPU_SPECS[gpu]
    h.validate_device(gpu.upper(), (minimum+5)*1024**3, capability)
    with pytest.raises(RuntimeError): h.validate_device('RTX 5090', 32*1024**3, (12,0))


def test_fleet_splits_selected_devices_ports_and_namespaces():
    fleet=script('fleet')
    a,b=fleet.plans('b200',4,9000,'7,6,5,4,3,2,1,0')
    assert a['CUDA_VISIBLE_DEVICES']=='7,6,5,4' and b['CUDA_VISIBLE_DEVICES']=='3,2,1,0'
    assert a['REF2VA_PORT']=='9000' and b['REF2VA_PORT']=='9001'
    assert a['REF2VA_INSTANCE']=='worker-0' and b['REF2VA_INSTANCE']=='worker-1'
    assert fleet.plans('h200',8,8188)[0]['REF2VA_INSTANCE']==''
    for args in [('b100',4,8188),('h200',2,8188),('h200',4,65535)]:
        with pytest.raises(ValueError): fleet.plans(*args)
    with pytest.raises(ValueError): fleet.plans('h200',4,8188,'0,1,2,3')


def test_startup_custom_defaults_and_dual_stream_dependencies(monkeypatch):
    monkeypatch.setenv('REF2VA_SOFTMAX_RANKS', '3')
    monkeypatch.delenv('REF2VA_DUAL_STREAM', raising=False)
    monkeypatch.setenv('REF2VA_CACHE_DIT', '0')
    monkeypatch.setenv('REF2VA_CACHE_DIT_THRESHOLD', '.15')
    monkeypatch.setenv('REF2VA_CACHE_DIT_MAX_CACHED_STEPS', '1')
    settings = backend.startup_settings()
    assert not settings.dual_stream and not settings.cache_dit
    assert settings.cache_dit_threshold == .15 and settings.cache_dit_max_cached_steps == 1
    monkeypatch.setenv('REF2VA_DUAL_STREAM', '1')
    with pytest.raises(ValueError, match='dual_stream requires'):
        backend.startup_settings()
    monkeypatch.delenv('REF2VA_DUAL_STREAM')
    monkeypatch.setenv('REF2VA_SOFTMAX_RANKS', '0')
    monkeypatch.setenv('REF2VA_INFERENCE_KERNELS', '0')
    assert not backend.startup_settings().dual_stream


def test_render_cli_fast_defaults_and_explicit_baseline(monkeypatch):
    render = script('render')
    requests = []
    def capture(**kwargs):
        requests.append(kwargs['settings'].validate())
        return {}
    monkeypatch.setattr(render, 'generate', capture)
    monkeypatch.setattr(sys, 'argv', ['render', '--prompt', 'test'])
    render.main()
    assert requests[-1].cache_dit and requests[-1].cache_dit_threshold == .25
    assert requests[-1].dual_stream and requests[-1].vae_tile_batch_size == 4 and requests[-1].vae_compile
    monkeypatch.setattr(sys, 'argv', ['render', '--prompt', 'test', '--softmax-ranks', '3', '--no-cache-dit',
                                     '--vae-tile-batch-size', '1', '--no-vae-compile'])
    render.main()
    assert not requests[-1].dual_stream and not requests[-1].cache_dit
    assert requests[-1].vae_tile_batch_size == 1 and not requests[-1].vae_compile


def test_instance_runtime_and_cache_isolation(monkeypatch,tmp_path):
    paths=[]
    for i in range(2):
        monkeypatch.setenv('REF2VA_INSTANCE',f'worker-{i}')
        paths.append(runtime_directory(tmp_path))
    assert paths[0]!=paths[1]
    monkeypatch.setenv('REF2VA_GPU_TYPE','b200')
    monkeypatch.setenv('REF2VA_GPUS','4')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','4,5,6,7')
    for key in ('NCCL_NVLS_ENABLE','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'):
        monkeypatch.delenv(key,raising=False)
    env=runner.worker_environment()
    assert env['CUDA_VISIBLE_DEVICES']=='4,5,6,7'
    assert 'b200-4' in env['TORCHINDUCTOR_CACHE_DIR']
    assert 'NCCL_NVLS_ENABLE' not in env
    monkeypatch.setenv('REF2VA_GPU_TYPE','h200')
    assert runner.worker_environment()['NCCL_NVLS_ENABLE']=='0'
    monkeypatch.setenv('REF2VA_INSTANCE','../escape')
    with pytest.raises(ValueError): runtime_directory(tmp_path)


def test_four_gpu_cleanup_selects_only_its_physical_cards(monkeypatch):
    monkeypatch.setenv('REF2VA_GPUS','4')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','4,5,6,7')
    output='\n'.join(f'{i}, GPU-{i}, 180000' for i in range(8))
    monkeypatch.setattr(gpu_check.subprocess,'check_output',lambda *a,**k:output)
    assert [x[0] for x in gpu_check.selected_gpus()]==['4','5','6','7']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','4,GPU-4,6,7')
    with pytest.raises(ValueError,match='duplicate physical'): gpu_check.selected_gpus()


def test_four_rank_watchdog_ignores_other_instances(monkeypatch,tmp_path):
    monkeypatch.setenv('REF2VA_GPUS','4')
    now=[0]
    w=WorkerWatchdog(tmp_path,'one',Policy(idle_timeout=3),lambda:now[0])
    process=types.SimpleNamespace(poll=lambda:None)
    state={'instance':'one','status':'ready','ready':True}
    for second in range(10):
        now[0]=second
        for rank in range(4): atomic_json(tmp_path/f'heartbeats/one-{rank}.json',{'sequence':second})
        atomic_json(tmp_path/'errors/other-0.json',{'traceback':'OOM'})
        assert w.check(process,state) is None


def metric():
    return {'upstream':{'timings':{'gpu_worker_seconds':.2,'denoise_seconds':.1}},
            'conditioning_cache_hit':True,'encode_seconds':0.}


def test_cpu_outputs_bounded_failure_does_not_cancel_next_gpu(tmp_path):
    completion=OutputCompletions(tmp_path,'worker',capacity=2)
    release=threading.Event()
    def finish():
        assert release.wait(3)
        return {'h264_encode_seconds':.3},{'video_codec':'libx264'}
    pending=types.SimpleNamespace(finish=finish,decode_started=time.perf_counter())
    for token in ('a','b'):
        completion.reserve()
        completion.submit({'token':token,'output':str(tmp_path/f'{token}.mp4')},metric(),pending)
    assert all((tmp_path/f'gpu_results/{t}.json').exists() for t in ('a','b'))
    assert not (tmp_path/'results/a.json').exists()
    acquired=threading.Event()
    thread=threading.Thread(target=lambda:(completion.reserve(),acquired.set()),daemon=True)
    thread.start()
    assert not acquired.wait(.05)
    release.set();assert acquired.wait(2)
    completion.slots.release();completion.pool.shutdown(wait=True);thread.join()
    assert backend.read_json(tmp_path/'results/a.json')['ok']
    broken=OutputCompletions(tmp_path,'worker')
    broken.reserve()
    def fail(): raise RuntimeError('encoder failed')
    broken.submit({'token':'c','output':str(tmp_path/'c.mp4')},metric(),types.SimpleNamespace(finish=fail))
    broken.pool.shutdown(wait=True)
    assert 'encoder failed' in backend.read_json(tmp_path/'results/c.json')['error']
    assert not (tmp_path/'cancel.json').exists()


def test_gpu_lock_released_before_waiting_cpu_output(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'RUNTIME',tmp_path/'runtime')
    monkeypatch.setattr(runner,'WORKER_PYTHON',Path(sys.executable))
    received=[]
    def worker(request,**kw):
        received.append(request)
        return {'_output_ticket':{'token':str(len(received)),'instance':'worker'}}
    pending=runner.generate(prompt='a',output=tmp_path/'a.mp4',worker_call=worker,defer_output=True)
    second=runner.generate(prompt='b',output=tmp_path/'b.mp4',worker_call=worker,defer_output=True)
    assert len(received)==2
    with (tmp_path/'runtime/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert isinstance(pending,runner.PendingGeneration) and isinstance(second,runner.PendingGeneration)


def test_output_ticket_does_not_block_watchdog_or_mark_task_complete(tmp_path,monkeypatch):
    monkeypatch.setattr(jobs,'RUNTIME',tmp_path)
    import uuid
    job=str(uuid.uuid4())
    jobs.create_job(job,{},Settings().render_plan())
    jobs.update_job(job,status='running',phase='encoding_output',output_owner={'pid':os.getpid(),'created':psutil.Process().create_time()})
    server=types.SimpleNamespace(prompt_queue=None)
    assert api.current_job(server,job)['status']=='running'
    w=WorkerWatchdog(tmp_path,'worker',Policy(request_timeout=1),clock=lambda:0)
    atomic_json(tmp_path/'command.json',{'instance':'worker','token':'a'})
    atomic_json(tmp_path/'gpu_results/a.json',{'instance':'worker','pending_output':True})
    process=types.SimpleNamespace(poll=lambda:None)
    assert w.check(process,{'ready':True,'instance':'worker','status':'ready'}) is None
    w.clock=lambda:2
    assert w.check(process,{'ready':True,'instance':'worker','status':'ready'}) is None
    fail_pending(tmp_path,'worker','GPU crashed during another task')
    assert backend.read_json(tmp_path/'results/a.json')['ok'] is False


def test_fleet_validates_both_before_stop_and_preserves_manifest_controls(monkeypatch,tmp_path):
    fleet=script('fleet')
    monkeypatch.setattr(fleet,'MANIFEST',tmp_path/'fleet.json')
    for name in ('REF2VA_GPUS','REF2VA_GPU_TYPE','CUDA_VISIBLE_DEVICES','PUBLIC_BASE_URL'):
        monkeypatch.delenv(name,raising=False)
    calls=[]
    def invoke(action,plan,args=(),capture=False):
        calls.append((action,dict(plan),args))
        return types.SimpleNamespace(stdout='{"running":true,"ready":true}')
    monkeypatch.setattr(fleet,'invoke',invoke)
    monkeypatch.setattr(sys,'argv',['fleet','restart','--gpu-type','b200','--gpus','4','--port','9000'])
    fleet.main()
    assert [x[0] for x in calls]==['validate','validate','stop','stop','stop','up','up','status','status']
    assert [x[1]['CUDA_VISIBLE_DEVICES'] for x in calls[5:7]]==['0,1,2,3','4,5,6,7']
    assert [x[1]['REF2VA_PORT'] for x in calls[5:7]]==['9000','9001']
    assert len(json.loads(fleet.MANIFEST.read_text())['instances'])==2
    calls.clear()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3')
    monkeypatch.setattr(sys,'argv',['fleet','status'])
    fleet.main()
    assert [x[1]['REF2VA_PORT'] for x in calls]==['9000','9001']
    calls.clear()
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    monkeypatch.setattr(sys,'argv',['fleet','up'])
    fleet.main()
    assert [x[0] for x in calls] == ['validate','stop','stop','stop','up','status']


def test_fleet_rejects_invalid_second_profile_before_stopping(monkeypatch,tmp_path):
    fleet=script('fleet')
    monkeypatch.setattr(fleet,'MANIFEST',tmp_path/'fleet.json')
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.delenv('PUBLIC_BASE_URL',raising=False)
    calls=[]
    def invoke(action,plan,*args,**kwargs):
        calls.append(action)
        if plan['REF2VA_INSTANCE']=='worker-1': raise ValueError('invalid child configuration')
    monkeypatch.setattr(fleet,'invoke',invoke)
    monkeypatch.setattr(sys,'argv',['fleet','restart','--gpus','4'])
    with pytest.raises(ValueError,match='invalid child'): fleet.main()
    assert calls==['validate','validate']
    assert not fleet.MANIFEST.exists()


def test_cancel_cpu_wait_never_writes_cancel_for_next_gpu(tmp_path,monkeypatch):
    monkeypatch.setattr(backend,'BACKEND',tmp_path)
    atomic_json(tmp_path/'command.json',{'instance':'worker','token':'next'})
    def interrupt(): raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        backend.wait_output({'instance':'worker','token':'previous'},interrupt=interrupt)
    assert not (tmp_path/'cancel.json').exists()


@pytest.mark.parametrize('gpu,size', [('h200',4),('b200',4),('b200',8),('b300',4),('b300',8)])
def test_workflow_starters_match_hardware_without_overwriting_saved_edits(tmp_path,monkeypatch,gpu,size):
    monkeypatch.setenv('REF2VA_GPU_TYPE',gpu)
    monkeypatch.setenv('REF2VA_GPUS',str(size))
    settings=Settings()
    install_workflows(ROOT,tmp_path,settings)
    for name,offset in [('openvdn_url_request',10),('openvdn_ref2va_like',8)]:
        path=tmp_path/'default/workflows'/f'{name}_{gpu}_{size}_fast_v1.json'
        data=json.loads(path.read_text())
        node=next(n for n in data['nodes'] if n['type'] in ('OpenVDNH200Generate','OpenVDNH200Request'))
        assert node['widgets_values'][offset:offset+2]==[settings.softmax_backend,settings.softmax_ranks]
        from openvdn_comfy.nodes import cache_inputs, optimization_inputs
        assert node['widgets_values'][offset+4:]==[getattr(settings,k) for k in (*cache_inputs(),*optimization_inputs())]
        path.write_text('{"user edited":true}')
        install_workflows(ROOT,tmp_path,settings)
        assert json.loads(path.read_text())=={'user edited':True}
