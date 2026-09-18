import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from openvdn_comfy.config import Settings
from openvdn_comfy.api import normalize_request, prompt_graph
from openvdn_comfy.nodes import OpenVDNH200Request
from openvdn_comfy.optimization_options import FIELDS
from openvdn_comfy.attention_runtime import isolated_mask_mod
from openvdn_comfy.communication import rewritten_methods


def test_optimization_controls_roundtrip_and_reject_incompatible_modes():
    body = dict(prompt='hello', reference_image_urls=['https://example.com/ref.png'], duration=10, ratio='9:16', resolution=768,
                fast_communication=False, streaming_output=False, cleanup_policy='always',
                attention_kernel='native', isolate_padding=True)
    normalized, settings = normalize_request(body)
    graph = prompt_graph(normalized)['1']['inputs']
    inputs = OpenVDNH200Request.INPUT_TYPES()['optional']
    assert all(k in inputs and graph[k] == getattr(settings, k) for k in FIELDS)
    with pytest.raises(ValueError, match='requires'):
        replace(settings, attention_kernel='decomposed').validate()
    for name in ('fast_communication','streaming_output','isolate_padding'):
        with pytest.raises(ValueError,match='boolean'):
            replace(settings, **{name:1}).validate()


@pytest.mark.parametrize('prefix', [1, 3, 4])
@pytest.mark.parametrize('full', [False, True])
def test_padding_mask_preserves_native_attention_for_all_real_queries(prefix, full):
    # Packed sequence has a hole before video; compare to a truly compact sequence.
    layout = SimpleNamespace(video_start=4, video_end=12, tokens_per_frame=2, num_frames=4, seq_len=12)
    bounds = [(0,3)]*4 if full else [(i,i) for i in range(4)]
    mod = isolated_mask_mod(layout,bounds,prefix,'cpu')
    pos = torch.arange(12)
    qv,kv,_,_,window,valid = mod(None,None,pos[:,None],pos[None,:])
    mask = (~(qv&kv)|window)&valid
    ids = torch.cat([pos[:prefix],pos[4:]])
    torch.manual_seed(123)
    q,k,v = [torch.randn(1,2,12,8,dtype=torch.float64) for _ in range(3)]
    padded = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
    compact = torch.nn.functional.scaled_dot_product_attention(q[:,:,ids],k[:,:,ids],v[:,:,ids],
                        attn_mask=mask[ids][:,ids])
    torch.testing.assert_close(padded[:,:,ids],compact,rtol=1e-12,atol=1e-12)
    assert not mask[:,prefix:4].any()


def test_pinned_communication_rewrite_contract_and_switch_preserves_collectives():
    path=Path(__file__).resolve().parents[1]/'work/upstream/openvdn/src/inference/utils/ulysses_runtime.py'
    if not path.exists():
        pytest.skip('Pinned upstream source unavailable')
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='UlyssesRuntime')
    names=('dispatch_fields_to_branches_overlapped','branches_to_sequence')
    fns=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns={'torch':torch}
    exec(compile(ast.Module(body=fns,type_ignores=[]),str(path),'exec'),ns)
    methods=rewritten_methods(type('Runtime',(),{k:ns[k] for k in names}))
    assert set(methods)==set(names)
    # Exercise dispatch and pending-work ownership using the real rewritten
    # function, with a tiny mock transport; payload parity is checked on CUDA at startup.
    calls=[]
    class Work:
        def __init__(self,group):self.group=group
        def wait(self):calls.append(('wait',self.group))
    def alltoall(recv,send,**kw):
        calls.append(('a2a',kw.get('group'),kw.get('async_op')))
        recv.fill_(len(calls))
        return Work(kw.get('group'))
    fn=methods[names[0]]
    fn.__globals__.update(dist=SimpleNamespace(all_to_all_single=alltoall),
                         _pack=lambda *args: calls.append(('pack',args[-1])))
    rank=SimpleNamespace(world_size=8,rank=0,splits=(3,)*8,sequence_length=24,num_heads=56,
                         softmax_ranks=6,softmax_head_splits=(10,10,9,9,9,9),linear_head_splits=(28,28),
                         softmax_dispatch_group='soft',linear_dispatch_group='linear',
                         branch_heads=10,branch_kind='softmax',profile_start=lambda:None,profile_end=lambda *a:None)
    q=torch.zeros(3,56,8);gate=torch.zeros(3,56,1);shared=torch.zeros(3,4)
    packed,hidden,pending=fn(rank,q,q,q,gate,q,q,gate,shared)
    assert packed.shape==(24,10,25) and hidden is None
    assert calls==[('pack',False),('a2a','soft',True),('pack',True),('a2a','linear',True),('wait','soft')]
    assert pending[0].group=='linear' and pending[1].numel()>0


def test_adaptive_cleanup_retains_hot_allocator_but_collects_on_compile_and_pressure(monkeypatch):
    from openvdn_comfy import request_cleanup
    calls=[]
    monkeypatch.setattr(request_cleanup.gc,'collect',lambda:calls.append('gc'))
    cuda=SimpleNamespace(mem_get_info=lambda d:(50,100),memory_allocated=lambda d:25,
                         memory_reserved=lambda d:40,empty_cache=lambda:calls.append('empty'))
    policy=request_cleanup.RequestCleanup();t=SimpleNamespace(cuda=cuda)
    assert not policy.run(t,'cuda','adaptive')['performed']
    assert policy.run(t,'cuda','adaptive',compiled=True)['reason']=='compilation'
    assert calls==['gc','empty']
    cuda.mem_get_info=lambda d:(1,100)
    assert policy.run(t,'cuda','adaptive')['reason']=='memory_pressure'
    assert policy.run(t,'cuda','always')['reason']=='always'
