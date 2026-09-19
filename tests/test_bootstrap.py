"""Startup contract: offline reuse, boot ownership, all-API readiness and failure cleanup."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psutil
import pytest

from openvdn_comfy import backend, hardware
from openvdn_comfy.config import atomic_json
from openvdn_comfy.host_identity import host_identity
from openvdn_comfy import network

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location('test_launch_' + name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('kind', ['h200', 'b200', 'b300'])
def test_hardware_autodetection_on_cloned_machine(monkeypatch, kind):
    rows = '\n'.join(f'{i}, GPU-new-{i}, NVIDIA {kind.upper()} SXM' for i in range(8))
    monkeypatch.setattr(hardware.subprocess, 'check_output', lambda *a, **k: rows)
    actual, devices = hardware.detect_hardware()
    assert actual == kind and devices == ','.join(f'GPU-new-{i}' for i in range(8))
    assert hardware.detect_hardware(kind, '7,6,5,4,3,2,1,0')[1].split(',')[0] == 'GPU-new-7'
    with pytest.raises(ValueError, match='absent'): hardware.detect_hardware('auto', 'GPU-old-0')
    with pytest.raises(ValueError, match='distinct'): hardware.detect_hardware('auto', '0,1,2,3,GPU-new-0,5,6,7')
    with pytest.raises(ValueError, match='detected'): hardware.detect_hardware('b200' if kind != 'b200' else 'h200')


def test_b300_check_keeps_strict_architecture_and_memory_validation():
    hardware.Hardware('b300', 4).validate_device('NVIDIA B300 SXM6 AC', int(267.7*1024**3), (10,3))
    with pytest.raises(RuntimeError): hardware.Hardware('b200', 4).validate_device('NVIDIA B300', 268*1024**3, (10,3))
    with pytest.raises(RuntimeError): hardware.Hardware('b300', 4).validate_device('NVIDIA B300', 268*1024**3, (10,0))


@pytest.mark.parametrize('repair_env,repair_models', [(False,False),(True,False),(False,True),(True,True)])
def test_unified_bootstrap_only_installs_missing_parts_and_rechecks(monkeypatch,tmp_path,repair_env,repair_models):
    boot=script('bootstrap')
    monkeypatch.setattr(boot,'ROOT',tmp_path)
    monkeypatch.setattr(boot,'detect_hardware',lambda *a:('b300',','.join(f'GPU-{i}' for i in range(8))))
    env_checks=iter([['missing'] if repair_env else [], []])
    model_checks=iter([['missing'] if repair_models else [], []])
    monkeypatch.setattr(boot,'ensure_toolchain',lambda *a:'/prepared/bin/nvcc')
    monkeypatch.setattr(boot,'environment_errors',lambda root:next(env_checks))
    monkeypatch.setattr(boot,'model_errors',lambda root:next(model_checks))
    calls=[]
    monkeypatch.setattr(boot,'stop_before_repair',lambda env:calls.append(('stop', env)))
    monkeypatch.setattr(boot,'run',lambda command,**kw:calls.append((command,kw['env'])))
    monkeypatch.setattr(sys,'argv',['bootstrap','--gpus','4'])
    boot.main()
    stages=[call[0] for call in calls]
    assert sum(isinstance(c,list) and c[-1]=='install' for c in stages)==int(repair_env)
    assert sum(isinstance(c,list) and c[-1]=='download' for c in stages)==int(repair_models)
    assert stages[-1][2]=='ensure'
    assert stages[-2][-1]==str(tmp_path/'scripts/prepare_kernels.py')
    assert calls[-1][1]['REF2VA_NVCC']=='/prepared/bin/nvcc'
    assert all(env['REF2VA_GPU_TYPE']=='b300' and env['REF2VA_GPUS']=='4' for _,env in calls)
    assert stages.count('stop')==int(repair_env)+int(repair_models)


def test_failed_install_verification_never_starts_service(monkeypatch,tmp_path):
    boot=script('bootstrap')
    monkeypatch.setattr(boot,'ROOT',tmp_path)
    monkeypatch.setattr(boot,'detect_hardware',lambda *a:('b300','0,1,2,3,4,5,6,7'))
    monkeypatch.setattr(boot,'environment_errors',lambda root:['still incompatible'])
    calls=[]
    monkeypatch.setattr(boot,'stop_before_repair',lambda env:None)
    monkeypatch.setattr(boot,'run',lambda command,**kw:calls.append(command))
    monkeypatch.setattr(sys,'argv',['bootstrap','--gpus','4'])
    with pytest.raises(RuntimeError,match='after install'): boot.main()
    assert len(calls)==1 and calls[0][-1]=='install'


def test_fleet_reuses_identical_live_code_but_restarts_on_clone_or_code_change(monkeypatch,tmp_path):
    fleet=script('fleet')
    monkeypatch.setattr(fleet,'MANIFEST',tmp_path/'fleet.json')
    monkeypatch.setattr(fleet,'check_ports',lambda _:None)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1,2,3,4,5,6,7')
    monkeypatch.delenv('PUBLIC_BASE_URL',raising=False)
    signature=['one']; host=[{'boot_id':'first','machine_id':'first'}]
    monkeypatch.setattr(fleet,'fingerprint',lambda *a:signature[0])
    monkeypatch.setattr(fleet,'host_identity',lambda:host[0])
    calls=[]
    monkeypatch.setattr(fleet,'invoke',lambda action,*a,**k:(calls.append(action) or types.SimpleNamespace(stdout='{"running":true,"ready":true}')))
    monkeypatch.setattr(sys,'argv',['fleet','ensure','--gpu-type','b300','--gpus','4'])
    for iteration in range(4):
        calls.clear()
        if iteration==2: host[0]={'boot_id':'clone','machine_id':'second'}
        if iteration==3: signature[0]='updated-code'
        fleet.main()
        assert calls.count('stop')==(0 if iteration==1 else 3)
        assert calls[-2:]==['status','status']


@pytest.mark.parametrize('failure',['exit','recovering','timeout','interrupt'])
def test_failed_or_cancelled_startup_reaps_both_instances(monkeypatch,tmp_path,failure):
    fleet=script('fleet')
    monkeypatch.setattr(fleet,'MANIFEST',tmp_path/'fleet.json')
    monkeypatch.setattr(fleet,'diagnostics',lambda _:None)
    monkeypatch.setattr(fleet,'check_ports',lambda _:None)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.delenv('PUBLIC_BASE_URL',raising=False)
    calls=[]; now=[0]
    monkeypatch.setattr(fleet.time,'monotonic',lambda:now[0])
    def sleep(_):
        if failure=='interrupt': raise KeyboardInterrupt
        now[0]+=2
    monkeypatch.setattr(fleet.time,'sleep',sleep)
    def invoke(action,plan,*a,**k):
        calls.append((action,plan['REF2VA_INSTANCE']))
        state={'running':True,'ready':plan['REF2VA_INSTANCE']=='worker-0'}
        if action=='status' and plan['REF2VA_INSTANCE']=='worker-1':
            if failure=='exit': state['running']=False
            if failure=='recovering': state['supervision']={'status':'recovering','last_error':'CUDA failure'}
        return types.SimpleNamespace(stdout=json.dumps(state))
    monkeypatch.setattr(fleet,'invoke',invoke)
    monkeypatch.setattr(sys,'argv',['fleet','ensure','--gpu-type','b300','--gpus','4','--wait-timeout','1'])
    error=KeyboardInterrupt if failure=='interrupt' else TimeoutError if failure=='timeout' else RuntimeError
    with pytest.raises(error): fleet.main()
    assert calls[-2:]==[('stop','worker-0'),('stop','worker-1')]


def test_service_requires_real_api_with_matching_worker_and_ignores_http_proxy(monkeypatch,tmp_path):
    service=script('service')
    monkeypatch.setattr(service,'BACKEND',tmp_path)
    owner={'pid':os.getpid(),'created':psutil.Process().create_time(),'host':host_identity()}
    atomic_json(tmp_path/'ui.json',owner)
    payload={'ready':True,'instance':'old'}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path=='/openvdn/health'
            self.send_response(200);self.end_headers();self.wfile.write(json.dumps(payload).encode())
        def log_message(self,*args): pass
    http=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
    monkeypatch.setenv('REF2VA_PORT',str(http.server_port));monkeypatch.setenv('REF2VA_LISTEN','0.0.0.0')
    monkeypatch.setenv('HTTP_PROXY','http://does-not-exist.invalid:9')
    monkeypatch.setenv('http_proxy','http://does-not-exist.invalid:9')
    try:
        assert not service.api_ready({'ready':True,'instance':'current'})
        payload['instance']='current'
        assert service.api_ready({'ready':True,'instance':'current'})
        payload['ready']=False
        assert not service.api_ready({'ready':True,'instance':'current'})
    finally:
        http.shutdown();http.server_close();thread.join()
    assert not service.api_ready({'ready':True,'instance':'current'})


def test_listener_configuration_and_ipv6_health_url_format(monkeypatch):
    monkeypatch.delenv('REF2VA_LISTEN', raising=False)
    assert network.listen_value() == '0.0.0.0,::'
    assert network.health_urls(8189) == ['http://127.0.0.1:8189/openvdn/health',
                                        'http://[::1]:8189/openvdn/health']
    monkeypatch.setenv('REF2VA_LISTEN', '127.0.0.1')
    assert network.listen_value() == '127.0.0.1'
    assert network.listen_value('0.0.0.0, [::], ::') == '0.0.0.0,::'
    assert network.health_urls(8188, '[2001:db8::123]') == ['http://[2001:db8::123]:8188/openvdn/health']
    for value in ('', ',', 'http://[::1]', '[::1]:8188', '0.0.0.0,', '::,invalid'):
        with pytest.raises(ValueError, match='addresses without ports'):
            network.listen_value(value)


def test_bootstrap_listen_argument_overrides_environment(monkeypatch, tmp_path):
    boot = script('bootstrap')
    monkeypatch.setattr(boot, 'ROOT', tmp_path)
    monkeypatch.setattr(boot, 'detect_hardware', lambda *a: ('b300', '0,1,2,3,4,5,6,7'))
    monkeypatch.setattr(boot, 'environment_errors', lambda _: [])
    monkeypatch.setattr(boot, 'model_errors', lambda _: [])
    monkeypatch.setenv('REF2VA_FUSED_DELTA', '0')
    monkeypatch.setenv('REF2VA_LISTEN', '127.0.0.1')
    monkeypatch.setattr(sys, 'argv', ['bootstrap', '--gpus', '4', '--listen', '0.0.0.0,::'])
    calls = []
    monkeypatch.setattr(boot, 'run', lambda command, **kw: calls.append((command, kw['env'])))
    boot.main()
    assert len(calls) == 1 and calls[0][1]['REF2VA_LISTEN'] == '0.0.0.0,::'
    assert '--listen' not in calls[0][0]  # Consumed once, never passed as a conflicting UI argument.


def test_fleet_preserves_listener_for_status_but_rebuilds_it_on_new_deployment(monkeypatch):
    fleet = script('fleet')
    plan = fleet.plans('b300', 4, 8188, listen='::')[1]
    monkeypatch.setenv('REF2VA_LISTEN', '0.0.0.0')
    captured = []
    monkeypatch.setattr(fleet.subprocess, 'run', lambda *a, **kw: captured.append(kw['env']))
    fleet.invoke('status', plan)
    assert captured[0]['REF2VA_LISTEN'] == '::' and captured[0]['REF2VA_PORT'] == '8189'
    # Starting on a cloned machine uses current configuration, not old saved IPs.
    assert fleet.plans('b300', 4, 8188)[1]['REF2VA_LISTEN'] == '0.0.0.0'


def test_dual_stack_port_preflight_detects_ipv6_conflict():
    import socket
    try:
        occupied = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        occupied.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        occupied.bind(('::', 0))
        occupied.listen(1)
    except OSError:
        pytest.skip('Host lacks IPv6 loopback')
    port = occupied.getsockname()[1]
    try:
        with pytest.raises(RuntimeError, match='Cannot bind API address ::'):
            network.check_bindings([('0.0.0.0,::', port)])
    finally:
        occupied.close()
    network.check_bindings([('0.0.0.0,::', port)])


def test_dual_stack_readiness_requires_both_listeners_and_current_instance(monkeypatch, tmp_path):
    import asyncio
    from aiohttp import web
    service = script('service')
    monkeypatch.setattr(service, 'BACKEND', tmp_path)
    monkeypatch.setattr(service, 'RECORD', tmp_path/'service.json')
    owner = {'pid': os.getpid(), 'created': psutil.Process().create_time(), 'host': host_identity()}
    atomic_json(tmp_path/'ui.json', owner)
    atomic_json(service.RECORD, {**owner, 'listen': '0.0.0.0,::'})
    # A later status shell's overrides must not hide a broken listener.
    monkeypatch.setenv('REF2VA_LISTEN', '127.0.0.1')
    monkeypatch.setenv('http_proxy', 'http://unreachable.invalid:9')
    monkeypatch.setenv('HTTP_PROXY', 'http://unreachable.invalid:9')
    payload = {'ready': True, 'instance': 'current'}
    ipv6_instance = ['current']
    async def exercise():
        async def handler(request):
            return web.json_response({**payload, 'instance': ipv6_instance[0] if ':' in request.remote else 'current'})
        app = web.Application(); app.router.add_get('/openvdn/health', handler)
        runner = web.AppRunner(app); await runner.setup()
        try:
            ipv4 = web.TCPSite(runner, '0.0.0.0', 0); await ipv4.start()
            port = ipv4._server.sockets[0].getsockname()[1]
            ipv6 = web.TCPSite(runner, '::', port)
            try:
                await ipv6.start()
            except OSError:
                pytest.skip('Host lacks IPv6 wildcard binding')
            monkeypatch.setenv('REF2VA_PORT', str(port))
            assert await asyncio.to_thread(service.api_ready, payload)
            ipv6_instance[0] = 'old-worker'
            assert not await asyncio.to_thread(service.api_ready, payload)
            ipv6_instance[0] = 'current'
            await ipv6.stop()
            assert not await asyncio.to_thread(service.api_ready, payload)
            atomic_json(service.RECORD, {**owner, 'listen': '0.0.0.0'})
            assert await asyncio.to_thread(service.api_ready, payload)
        finally:
            await runner.cleanup()
    asyncio.run(exercise())


def test_old_boot_pid_and_ready_state_cannot_be_reused(monkeypatch,tmp_path):
    owner={'pid':os.getpid(),'created':psutil.Process().create_time(), 'host':{'boot_id':'old-image','machine_id':'old-machine'}}
    assert not backend.same_process(owner)
    monkeypatch.setattr(backend,'BACKEND',tmp_path)
    atomic_json(tmp_path/'owner.json',{**owner,'instance':'old'})
    atomic_json(tmp_path/'state.json',{'instance':'old','status':'ready'})
    assert not backend.health()['ready']
    service=script('service')
    monkeypatch.setattr(service,'controller',lambda:types.SimpleNamespace(pid=os.getpid()))
    monkeypatch.setattr(service,'health',lambda:{'ready':False,'phase':'failed','supervision':{'status':'stopped','controller':owner}})
    status=service.snapshot()
    assert status['running'] and not status['ready'] and not status['supervision']
    assert status['phase']=='checking_environment'


def test_manifest_addresses_are_not_replayed_on_new_machine(monkeypatch):
    fleet=script('fleet')
    monkeypatch.delenv('PUBLIC_BASE_URL',raising=False)
    monkeypatch.delenv('REF2VA_PUBLIC_BASE_URL_0',raising=False)
    captured=[]
    monkeypatch.setattr(fleet.subprocess,'run',lambda *a,**kw:captured.append(kw['env']))
    plan=fleet.plans('b300',4,8188)[0]
    plan['PUBLIC_BASE_URL']='http://old-host:8188'
    fleet.invoke('up',plan)
    assert captured[0]['PUBLIC_BASE_URL']==''


def test_startup_wait_does_not_block_status_logs_or_stop(monkeypatch,tmp_path):
    import fcntl
    fleet=script('fleet')
    monkeypatch.setattr(fleet,'MANIFEST',tmp_path/'fleet.json')
    monkeypatch.setattr(fleet,'check_ports',lambda _:None)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.delenv('PUBLIC_BASE_URL',raising=False)
    monkeypatch.setattr(fleet,'invoke',lambda *a,**k:None)
    def wait(selected, timeout):
        with (tmp_path/'fleet.lock').open('a') as other_command:
            fcntl.flock(other_command,fcntl.LOCK_EX|fcntl.LOCK_NB)
    monkeypatch.setattr(fleet,'wait_ready',wait)
    monkeypatch.setattr(sys,'argv',['fleet','ensure','--gpu-type','b300','--gpus','4'])
    fleet.main()


def test_offline_model_check_reuses_complete_assets_and_finds_missing_shard(monkeypatch,tmp_path):
    check=script('check_install')
    monkeypatch.delenv('REF2VA_MODELS',raising=False)
    stamp={'models':{'vdn':{'revision':'fixed'}}}
    (tmp_path/'sources.lock.json').write_text(json.dumps(stamp))
    models=tmp_path/'models';models.mkdir();(models/'sources.json').write_text(json.dumps(stamp['models']))
    for part in ('vdn/h3-base/transformer','vdn/h3-base/vae','vdn/h3-base/audio_vae',
                 'vdn/stage-dmd-step-250/linear_branch','vdn/stage-dmd-step-250/adapters/default',
                 'vdn/stage-dmd-step-250/adapters/turbo','conditioner/text_encoder'):
        directory=models/part;directory.mkdir(parents=True)
        (directory/'one.safetensors').write_bytes(b'x'*32)
    processor=models/'conditioner/processor';processor.mkdir();(processor/'tokenizer_config.json').write_text('{}')
    index=models/'vdn/h3-base/transformer/weights.safetensors.index.json'
    index.write_text(json.dumps({'weight_map':{'a':'one.safetensors'}}))
    assert check.model_errors(tmp_path)==[]
    index.write_text(json.dumps({'weight_map':{'a':'one.safetensors','b':'missing.safetensors'}}))
    assert any('missing shard' in error for error in check.model_errors(tmp_path))


def test_environment_probe_checks_transitive_dependencies_offline(monkeypatch,tmp_path,capsys):
    probe=script('environment_probe')
    root=tmp_path.resolve();(root/'.deps/ComfyUI').mkdir(parents=True)
    (root/'.deps/ComfyUI/requirements.txt').write_text('torch\naiohttp>=3.11\n')
    (root/'constraints-comfy.txt').write_text('torch==2.10.0\n')
    monkeypatch.setattr(sys,'argv',['probe',str(root),'ui'])
    monkeypatch.setattr(sys,'prefix',str(root/'.venv-ui'))
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(__version__='2.10.0+cpu'))
    versions={'torch':'2.10.0+cpu','aiohttp':'3.14.3','required-package':'2.0'}
    monkeypatch.setattr(probe.metadata,'version',lambda name:versions[name])
    distribution=types.SimpleNamespace(metadata={'Name':'aiohttp'},requires=['required-package>=2','absent; extra == "optional"'])
    monkeypatch.setattr(probe.metadata,'distributions',lambda:[distribution])
    probe.main()
    assert 'verified' in capsys.readouterr().out
    versions['required-package']='1.0'
    with pytest.raises(RuntimeError,match='requires'): probe.main()


def test_pinned_source_check_does_not_fetch_network_and_detects_relocation(monkeypatch,tmp_path):
    check=script('check_install')
    lock={'git':{name:{'revision':'pin','url':'https://example/'+name} for name in ('ComfyUI','openvdn','diffusers')}}
    (tmp_path/'sources.lock.json').write_text(json.dumps(lock))
    patches=tmp_path/'.deps/openvdn/diffusers_patches';patches.mkdir(parents=True)
    patch=patches/'one.patch';patch.write_text('official patch')
    import hashlib
    stamp=tmp_path/'.deps/diffusers/.git/openvdn-patches.json';stamp.parent.mkdir(parents=True)
    stamp.write_text(json.dumps({'base':'pin','head':'patched','patches':hashlib.sha256(patch.name.encode()+patch.read_bytes()).hexdigest()}))
    link=tmp_path/'.deps/ComfyUI/custom_nodes/openvdn_h200';link.parent.mkdir(parents=True);link.symlink_to(tmp_path)
    def git(path,*args):
        if args==('rev-parse','HEAD'): return 'patched' if path.name=='diffusers' else 'pin'
        if args==('status','--porcelain','--untracked-files=no'): return ''
        if args==('config','--get','remote.origin.url'): return lock['git'][path.name]['url']
        raise AssertionError('Read-only check must never fetch or mutate Git')
    monkeypatch.setattr(check,'git',git)
    assert check.source_errors(tmp_path)==[]
    link.unlink();link.symlink_to(tmp_path/'old-clone-path')
    assert check.source_errors(tmp_path)==['ComfyUI node link is missing or belongs to another checkout']


@pytest.mark.parametrize('failure', ['compiler', 'kernel', 'disabled'])
def test_cuda_preflight_happens_before_stopping_live_service(monkeypatch, tmp_path, failure):
    boot = script('bootstrap')
    monkeypatch.setattr(boot, 'ROOT', tmp_path)
    monkeypatch.setattr(boot, 'detect_hardware', lambda *a: ('b300', '0,1,2,3,4,5,6,7'))
    monkeypatch.setattr(boot, 'environment_errors', lambda _: [])
    monkeypatch.setattr(boot, 'model_errors', lambda _: [])
    monkeypatch.setenv('REF2VA_FUSED_DELTA', '0' if failure == 'disabled' else '1')
    monkeypatch.setattr(sys, 'argv', ['bootstrap', '--gpus', '8'])
    calls = []
    def compiler(*_):
        calls.append('compiler')
        if failure == 'compiler': raise RuntimeError('cannot prepare compiler')
        return '/prepared/bin/nvcc'
    def run(command, **kwargs):
        calls.append(command)
        if failure == 'kernel': raise RuntimeError('cannot compile kernel')
    monkeypatch.setattr(boot, 'ensure_toolchain', compiler)
    monkeypatch.setattr(boot, 'run', run)
    monkeypatch.setattr(boot, 'stop_before_repair', lambda _: pytest.fail('Healthy service must remain running'))
    if failure == 'disabled':
        boot.main()
        assert len(calls) == 1 and calls[0][2] == 'ensure'
    else:
        with pytest.raises(RuntimeError, match='cannot'): boot.main()
        assert not any(isinstance(c, list) and c[2:] and c[2] == 'ensure' for c in calls)
