import ast
from contextlib import nullcontext
from dataclasses import asdict, replace
from types import SimpleNamespace, MethodType

import pytest
import torch
import torch.nn.functional as F

from openvdn_comfy.api import normalize_request, prompt_graph
from openvdn_comfy.config import Settings
from openvdn_comfy.boundary_scan import run_boundary_scans, _boundary_frames
from openvdn_comfy.linear_acceleration import LinearAcceleration, boundary_compatible
from openvdn_comfy.linear_kv import LinearKVRuntime
from openvdn_comfy.fine_profile import FineProfiler, kernel_summary, instrument, rule
from openvdn_comfy.window_attention import window_softmax_fast
from tests.test_linear_kv import upstream_path, source_module, native_branch


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def softmax_module():
    path = upstream_path('src/models/softmax_attention/decomposed.py')
    def varlen(q, k, v, cq, ck, *unused):
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        out = []
        for i in range(len(cq)-1):
            qs, ks = slice(cq[i], cq[i+1]), slice(ck[i], ck[i+1])
            out.append(F.scaled_dot_product_attention(q[qs].transpose(0,1)[None],
                k[ks].transpose(0,1)[None], v[ks].transpose(0,1)[None], scale=unused[-1])[0].transpose(0,1))
        return torch.cat(out)
    ns = dict(torch=torch, SequenceLayout=SimpleNamespace, _PLAN_CACHE={}, MAX_CACHED_PLANS=64,
              varlen_kernel=lambda:varlen, _dense_backends=lambda:None, sdpa_kernel=lambda _:nullcontext(),
              scaled_dot_product_attention=F.scaled_dot_product_attention)
    return definitions(path, {'_plan','_Plan','window_softmax_decomposed'}, ns)


def feature_module():
    path = upstream_path('src/models/linear_attention/features.py')
    names = {n.name for n in ast.parse(path.read_text()).body if isinstance(n, (ast.ClassDef,ast.FunctionDef))}
    ns = dict(torch=torch, F=F, nn=torch.nn, _TCONV={'fn':None}, _FEATURES_CACHE={})
    module = definitions(path, names, ns)
    ns['_compiled'] = lambda key, body: body
    ns['temporal_conv_activate'] = ns['_tconv_activate_body']
    return module


def fixture(profile=False):
    branch, _ = native_branch()
    features, scan = feature_module(), source_module('scan')
    backends = branch._delta_backend.__func__.__globals__['DELTA_BACKENDS']
    delta = SimpleNamespace(**{cls.__name__:cls for cls in backends.values()})
    path = upstream_path('src/models/linear_attention/branch.py')
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BidirectionalLinearBranch')
    names = {'_feature_one','_features'}
    nodes = [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    nodes += [n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_HeadSliceSepConv']
    ns = dict(torch=torch, F=F, prepare_linear_features=features.prepare_linear_features,
              prepare_linear_features_inference=features.prepare_linear_features_inference)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'),ns)
    for name in names:
        setattr(branch, name, MethodType(ns[name], branch))
    branch.short_conv = features.LinearAttentionSepConv(16)
    events = []
    runtime = SimpleNamespace(profile_enabled=profile, rank=0, softmax_ranks=5, device=torch.device('cpu'),
        profile_start=lambda:None, profile_end=lambda name, start:events.append(name))
    hybrids = [SimpleNamespace(linear_attention=branch, chunk=3, anchor_frames='none')]
    kv = LinearKVRuntime(hybrids, runtime)
    acceleration = LinearAcceleration(hybrids, kv, runtime, scan)
    profiler = FineProfiler(runtime, kv, softmax_module(), features, scan, delta, acceleration)
    return branch, kv, acceleration, profiler, events


@pytest.mark.parametrize('profile', [False,True])
@pytest.mark.parametrize('ratio', [1.,.5])
@pytest.mark.parametrize('boundary', [False,True])
@pytest.mark.parametrize('fused', [False,True])
def test_actual_readout_features_scans_profile_and_restoration(profile, ratio, boundary, fused):
    torch.manual_seed(131)
    branch, kv, acceleration, profiler, events = fixture(profile)
    acceleration.select(Settings(fused_delta=False, boundary_scan=boundary), torch.device('cpu'))
    # Exercise the actual adapters on CPU with an independent inverse oracle;
    # this tests argument scaling/ordering, not CUDA kernel implementation.
    def factor_oracle(a,b,alpha):
        inverse=torch.linalg.inv(a.double()+torch.eye(a.shape[-1]))
        return (alpha.double().unsqueeze(-1)*inverse).float(),(b.double()@inverse).float()
    acceleration.kernel=factor_oracle
    acceleration.kernel.verification={'checked':False,'cpu_test_oracle':True}
    acceleration.fused_delta=fused
    frames, tokens, heads, dim = 7, 6, 4, 4
    qkv = tuple(torch.randn(frames*tokens,heads,dim) for _ in range(3))
    tqkv = tuple(torch.randn(3,heads,dim) for _ in range(3))
    args = dict(xv=None, num_frames=frames, tokens_per_frame=tokens, bounds=[(i//3*3,min(i//3*3+2,6)) for i in range(frames)],
        qkv_raw=qkv, beta=torch.rand(frames*tokens,heads), gate=torch.rand(frames*tokens,heads,dim),
        frame_size=(2,3), frame_mean=torch.ones(frames,heads), text_qkv_raw=tqkv, text_beta=torch.rand(3,heads), inference=True)
    original = branch._readout_inference, branch._feature_one
    with torch.no_grad(), kv.request(ratio):
        expected = branch.forward(**args)
    events.clear()
    with torch.no_grad(), kv.request(ratio), acceleration.request(), profiler.request():
        actual = branch.forward(**args)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert (branch._readout_inference,branch._feature_one)==original
    if profile:
        assert {'linear_features','linear_factor_apply','linear_fused_delta' if fused else 'linear_cholesky','linear_k_spatial_conv',
                'linear_v_temporal_activate','linear_query_readout'} <= set(events)
        assert ('linear_chunk_scan' if boundary else 'linear_scan_forward') in events
        assert profiler.report()['host_scopes']['linear_features']['calls']==1
    else:
        assert not profiler.report()['host_scopes']
    assert acceleration.report()['calls'].get('boundary_scan',0)==int(boundary)
    assert acceleration.report()['calls'].get('fused_delta',0)==(2 if fused else 0)
    with pytest.raises(RuntimeError,match='injected'), kv.request(ratio), acceleration.request(), profiler.request():
        raise RuntimeError('injected')
    assert not profiler.active and not acceleration.active and not kv.active
    assert (branch._readout_inference,branch._feature_one)==original


@pytest.mark.parametrize('frames', [1,2,7,13,70])
@pytest.mark.parametrize('chunk', [2,3,5])
@pytest.mark.parametrize('offset', [0,1])
@pytest.mark.parametrize('text', [False,True])
def test_chunk_boundaries_match_independent_sequential_recurrence(frames,chunk,offset,text):
    generator = torch.Generator().manual_seed(37)
    t = torch.eye(4,dtype=torch.float64)[None,None].expand(frames,2,4,4).clone()*.8
    t += torch.randn(t.shape,generator=generator,dtype=t.dtype)*.03
    b = torch.randn(t.shape,generator=generator,dtype=t.dtype)
    start = torch.randn(2,4,4,generator=generator,dtype=t.dtype) if text else torch.zeros_like(b[0])
    prefix,suffix = [],[]
    p,s = start.clone(),start.clone()
    for f in range(frames):
        p = p@t[f]+b[f]; prefix.append(p)
        s = s@t[-1-f]+b[-1-f]; suffix.append(s)
    expected = torch.stack(prefix),torch.stack(list(reversed(suffix)))
    actual = run_boundary_scans(t,b,start if text else None,chunk=chunk,frame_offset=offset)
    _,ends,_,starts,_ = _boundary_frames(frames,chunk,offset,'cpu')
    for a,e,ids in zip(actual,expected,(ends,starts)):
        torch.testing.assert_close(a[ids],e[ids],rtol=1e-12,atol=1e-12)


def test_boundary_guard_falls_back_for_unsupported_windows():
    assert boundary_compatible(13,5,0,((0,4),)*5+((5,9),)*5+((10,12),)*3)
    assert boundary_compatible(11,5,1,((-1,3),)*4+((4,8),)*5+((9,13),)*2)
    assert not boundary_compatible(13,5,0,tuple((i,i) for i in range(13)))
    assert not boundary_compatible(13,1,0,((0,12),)*13)
    branch, kv, acceleration, profiler, _ = fixture(True)
    acceleration.boundary_scan = True
    backend = next(cls for cls in profiler.factors if cls.__name__=='VdnDelta')()
    a = torch.eye(4).expand(4,2,4,4).clone(); b=torch.ones_like(a); alpha=torch.full((4,2,4),.9)
    with torch.no_grad():
        expected=acceleration.native_scans(backend,alpha,a,b)
        actual=acceleration.scans(backend,alpha,a,b,bounds=tuple((i,i) for i in range(4)),chunk=5,frame_offset=0)
    for x,y in zip(actual,expected):
        assert torch.equal(x,y)
    assert acceleration.calls=={'native_scan':1}


@pytest.mark.parametrize('anchors',['none','rows','columns','both'])
@pytest.mark.parametrize('strided',[False,True])
@pytest.mark.parametrize('irregular',[False,True])
def test_softmax_copies_preserve_all_reference_global_anchor_and_window_rows(monkeypatch,anchors,strided,irregular):
    import sys
    native=softmax_module()
    # Supply the dependency at the package boundary, retaining the real plan/math.
    monkeypatch.setitem(sys.modules,'src.models.softmax_attention',SimpleNamespace(decomposed=native))
    ns=native.window_softmax_decomposed.__globals__
    native.varlen_kernel=ns['varlen_kernel']; native.sdpa_kernel=ns['sdpa_kernel']
    native._dense_backends=ns['_dense_backends'];native.scaled_dot_product_attention=ns['scaled_dot_product_attention']
    layout=SimpleNamespace(video_start=5,video_end=17,seq_len=20,num_frames=6,tokens_per_frame=2)
    bounds=[(i//2*2,i//2*2+1) for i in range(6)]
    if irregular: bounds=[(0,0),(2,4),(0,0),(1,5),(2,4),(0,5)]
    generator=torch.Generator().manual_seed(61)
    qkv=[torch.randn(20,4 if strided else 2,8,generator=generator,dtype=torch.float64) for _ in range(3)]
    if strided:qkv=[x[:,::2] for x in qkv]
    expected=native.window_softmax_decomposed(*qkv,layout,bounds,8**-.5,anchor_frames=anchors)
    actual=window_softmax_fast(*qkv,layout,bounds,8**-.5,anchor_frames=anchors)
    assert torch.equal(actual,expected)


def test_new_request_controls_roundtrip_and_constraints():
    controls=dict(fused_delta=False,boundary_scan=False,fast_softmax=False,dual_stream=True,softmax_ranks=0,
                  profile=True,profile_kernels=True)
    request,settings=normalize_request(dict(prompt='hello',duration=10,ratio='9:16',resolution=768,
        reference_image_urls=['https://example.com/ref.png'],**controls))
    assert Settings(**asdict(settings)).validate()==settings
    for name,value in controls.items():assert prompt_graph(request)['1']['inputs'][name]==value
    for name in ('fused_delta','boundary_scan','fast_softmax','dual_stream','profile_kernels'):
        with pytest.raises(ValueError,match='boolean'):replace(settings,**{name:1}).validate()
    for changes in ({'softmax_ranks':5},{'inference_kernels':False}):
        with pytest.raises(ValueError,match='dual_stream'):replace(settings,**changes).validate()
    with pytest.raises(ValueError,match='profile_kernels'):replace(settings,profile=False).validate()


def test_kernel_profile_overlaps_and_missing_cuda_are_not_reported_as_zero():
    def event(name,start,end,index=0,kind='CUDA',self_cpu=0):
        return SimpleNamespace(name=name,device_type='DeviceType.'+kind,device_index=index,
            time_range=SimpleNamespace(start=start,end=end,elapsed_us=lambda:end-start),
            is_user_annotation=False,self_cpu_time_total=self_cpu)
    events=[event('gemm',0,1000),event('nccl_alltoall',500,2000),event('foreign',0,99999,index=1),
            event('aten::mm',0,30,kind='CPU',self_cpu=30)]
    result=kernel_summary(events,0)
    assert result['sum_device_event_ms']==2.5 and result['device_busy_union_ms']==2
    assert result['nccl_device_event_ms']==1.5 and result['device_event_count']==2
    assert result['top_cpu_operators'][0]['self_cpu_ms']==.03
    assert kernel_summary(events,2)['device_busy_union_ms'] is None


def test_instrumentation_fails_closed_on_missing_boundaries():
    def test_fn():return 1
    with pytest.raises(RuntimeError,match='Unexpected fine-profile boundaries'):
        instrument(test_fn,'def test_fn(): return 1',{'assign:absent':rule('absent')},{})


@pytest.mark.parametrize('world',[4,8])
def test_dual_stream_transport_exact_roundtrip_with_uneven_sequence_shards(monkeypatch,world):
    from openvdn_comfy.dual_stream import DualStream
    # Simulate the full rank matrix of NCCL splits. Actual transport remains
    # subject to the mandatory first-shape GPU forward parity check.
    lengths=[2+i%3 for i in range(world)]; heads=world*2; width=7
    whole=torch.arange(sum(lengths)*heads*width).view(sum(lengths),heads,width)
    states=[]; pending=[]
    def exchange(recv,send,**kw):
        pending.append((recv,send.clone(),kw));return SimpleNamespace(wait=lambda:None)
    monkeypatch.setattr(torch.distributed,'all_to_all_single',exchange)
    offset=0
    received=[]
    for rank,rows in enumerate(lengths):
        state=DualStream(SimpleNamespace(world_size=world,rank=rank,splits=lengths,
            sequence_length=sum(lengths),softmax_dispatch_group='soft',profile_enabled=False),
            SimpleNamespace(hybrids=[],forwards={}),None)
        received.append(state.to_heads(whole[offset:offset+rows], 'shared')[0])
        offset+=rows;states.append(state)
    def complete():
        sends=[p[1].split(p[2]['input_split_sizes']) for p in pending]
        for dest,(recv,send,kw) in enumerate(pending):
            chunks=[s[dest] for s in sends]
            assert kw['output_split_sizes']==[x.numel() for x in chunks]
            recv.copy_(torch.cat(chunks))
        pending.clear()
    complete()
    for rank,got in enumerate(received):assert torch.equal(got,whole[:,rank*2:(rank+1)*2])
    back=[s.to_rows(t,'return','linear')[0] for s,t in zip(states,received)]
    complete();offset=0
    for rows,got in zip(lengths,back):
        actual=got.permute(1,0,2,3).contiguous().view(rows,heads,width)
        assert torch.equal(actual,whole[offset:offset+rows]);offset+=rows


@pytest.mark.parametrize('kind',['VdnDelta','VdnScaledDelta'])
def test_fused_adapter_scaling_matches_original_delta_backend(kind):
    _,_,engine,_,_=fixture(False)
    backend=getattr(source_module('delta_rule'),kind)(tokens_per_frame=11)
    engine.fused_delta=True
    def oracle(a,b,alpha):
        inv=torch.linalg.inv(a.double()+torch.eye(4))
        return (alpha.double().unsqueeze(-1)*inv).float(),(b.double()@inv).float()
    engine.kernel=oracle
    torch.manual_seed(31)
    k=torch.randn(3,2,9,4); a=k.transpose(-1,-2)@k; b=torch.randn_like(a); alpha=torch.rand(3,2,4)
    with torch.no_grad():
        expected=backend.factor_apply(alpha,a,b); actual=engine.factor(backend,alpha,a,b)
    for x,y in zip(actual,expected):torch.testing.assert_close(x,y,rtol=2e-5,atol=1e-6)


@pytest.mark.parametrize('full_cover',[False,True])
@pytest.mark.parametrize('anchors',['none','both'])
def test_dual_forward_math_matches_pinned_attention_with_reference_rows(monkeypatch,full_cover,anchors):
    import sys
    from openvdn_comfy.dual_stream import DualStream
    from openvdn_comfy import attention_runtime
    # Real pinned projections, gates, head slicing, text state, features/readout.
    # CPU SDPA replaces GPU kernels; one-rank identity transport isolates math.
    # The separate 4/8-rank test checks uneven collective payload ordering.
    branch,kv,engine,profiler,_=fixture(False)
    type(branch).__call__=type(branch).forward
    torch.manual_seed(27)
    heads,dim,hidden,frames,tokens=4,4,16,7,6
    start=5; length=start+frames*tokens+3
    layout=SimpleNamespace(seq_len=length,video_start=start,video_end=start+frames*tokens,
        num_frames=frames,tokens_per_frame=tokens,frame_size=(2,3),text_range=(0,3))
    x=torch.randn(length,hidden)
    gate=definitions(upstream_path('src/models/attention_gates.py'), {'OutputGate'},
        dict(torch=torch,nn=torch.nn,math=__import__('math'))).OutputGate
    branch.beta_proj=torch.nn.Linear(hidden,heads)
    branch.output_gate=gate(hidden,heads,head_dim=dim,bottleneck=3,init='random')
    fake_fp8=type('FakeFP8',(),{})
    def qk(t,w,eps,cos,sin):
        value=F.rms_norm(t,(dim,),w,eps)
        first,last=value.chunk(2,-1)
        return value*cos[:,None]+torch.cat((-last,first),-1)*sin[:,None]
    def softgate(value,weight,inference=False):return (value*weight).reshape(len(value),-1)
    def dispatch(q,k,v,**kwargs):
        return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
    attn_path=upstream_path('src/models/hybrid_attention.py')
    cls=next(n for n in ast.parse(attn_path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='HybridAttention')
    qkvfn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_qkv')
    ns=dict(torch=torch,Fp8Linear=fake_fp8,_qk_prep=qk)
    exec(compile(ast.Module(body=[qkvfn],type_ignores=[]),str(attn_path),'exec'),ns)
    originals=SimpleNamespace(to_q=torch.nn.Linear(hidden,hidden),to_k=torch.nn.Linear(hidden,hidden),
        to_v=torch.nn.Linear(hidden,hidden),norm_q=torch.nn.RMSNorm(dim),norm_k=torch.nn.RMSNorm(dim),
        to_out=[torch.nn.Linear(hidden,hidden),torch.nn.Identity()],heads=heads,processor=SimpleNamespace())
    bounds=[(0,frames-1)]*frames if full_cover else [(i//3*3,min(i//3*3+2,frames-1)) for i in range(frames)]
    attn=SimpleNamespace(orig=originals,num_heads=heads,head_dim=dim,layout=layout,linear_attention=branch,
        enable_softmax_gate=True,softmax_gate=gate(hidden,heads),_bounds=lambda _:bounds,anchor_frames=anchors,
        inference_mode=True,teacher_mode=False,linear_attention_enabled=True,softmax_impl='ref',enable_text_state=True,
        to_out_linear=torch.nn.Linear(hidden,hidden),_ref2va_exact=SimpleNamespace(verify=False,projection_calls=0))
    attn._qkv=MethodType(ns['_qkv'],attn)
    module=softmax_module()
    def softmax(q,k,v,layout,bounds,scale,anchor_frames='none'):
        return module.window_softmax_decomposed(q,k,v,layout,bounds,scale,anchor_frames=anchor_frames)
    monkeypatch.setattr(attention_runtime,'window_attention',lambda a,*args:softmax(*args,anchor_frames=a.anchor_frames))
    un=definitions(upstream_path('src/inference/utils/ulysses.py'),
        {'_linear_branch_shard','_ulysses_attention_forward'},dict(torch=torch,UlyssesRuntime=object,
            dispatch_attention_fn=dispatch,apply_softmax_gate=softgate,window_softmax_reference=softmax))
    work=SimpleNamespace(wait=lambda:None)
    stream=SimpleNamespace(wait_stream=lambda _:None)
    rope=(torch.randn(length,dim).cos(),torch.randn(length,dim).sin())
    rt=SimpleNamespace(world_size=1,rank=0,num_heads=heads,heads_per_rank=heads,splits=[length],sequence_length=length,
        branch_parallel=False,profile_enabled=False,local_start=0,local_end=length,_ref2va_full_rotary=rope,
        video_frame_mean_async=lambda value,l:(value[l.video_start:l.video_end].view(frames,tokens,hidden).sum(1),work),
        gather_sequence=lambda value:value,sequence_to_heads=lambda value:value,heads_to_sequence=lambda value:value,
        softmax_dispatch_group='soft',linear_dispatch_group='linear')
    attn._ulysses_runtime=rt
    monkeypatch.setitem(sys.modules,'src.models.ops.fp8_linear',SimpleNamespace(Fp8Linear=fake_fp8,quantize_activation=None))
    monkeypatch.setitem(sys.modules,'src.models.softmax_attention',SimpleNamespace(window_softmax_reference=softmax))
    monkeypatch.setitem(sys.modules,'src.models.softmax_attention.kernels',SimpleNamespace(_qk_prep=qk,apply_softmax_gate=softgate))
    monkeypatch.setitem(sys.modules,'src.inference.utils.ulysses',un)
    monkeypatch.setitem(sys.modules,'diffusers.models.attention_dispatch',SimpleNamespace(dispatch_attention_fn=dispatch))
    monkeypatch.setattr(torch.cuda,'current_stream',lambda *a:stream)
    monkeypatch.setattr(torch.cuda,'stream',lambda *a:nullcontext())
    monkeypatch.setattr(torch.Tensor,'record_stream',lambda *a:None)
    def exchange(recv,send,**kw):recv.copy_(send);return work
    monkeypatch.setattr(torch.distributed,'all_to_all_single',exchange)
    dual=DualStream(rt,SimpleNamespace(hybrids=[attn],forwards={}),un._ulysses_attention_forward)
    dual.stream=stream
    with torch.no_grad():
        expected=un._ulysses_attention_forward(attn,x[None],rope)
        actual=dual.compute(attn,x[None])
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)


def test_startup_fingerprint_includes_cuda_source_and_compiler_setting(tmp_path,monkeypatch):
    from scripts import fleet
    monkeypatch.setattr(fleet,'ROOT',tmp_path)
    for name in ('deploy.sh','sources.lock.json'):(tmp_path/name).write_text('same')
    native=tmp_path/'openvdn_comfy/_vendor/kernel.cu'
    native.parent.mkdir(parents=True);native.write_text('version 1')
    first=fleet.fingerprint([],[])
    native.write_text('version 2')
    second=fleet.fingerprint([],[])
    assert first!=second
    monkeypatch.setenv('CUDA_HOME','/new/toolkit')
    assert fleet.fingerprint([],[])!=second
