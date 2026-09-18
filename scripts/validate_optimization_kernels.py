"""Optional model-free GPU validation; run with torchrun --nproc_per_node=8.

Covers every 1+7 ... 7+1 layout and compares padding isolation / decomposed
attention to a dense fp32 reference. Does not establish generation-quality parity.
"""
from datetime import timedelta
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'.deps/openvdn')]


def main():
    import torch
    import torch.distributed as dist
    from src.inference.utils.ulysses_runtime import init_ulysses
    from src.models.sequence_layout import SequenceLayout
    from src.models.softmax_attention.flex_attention import window_softmax_flex
    from src.models.softmax_attention.decomposed import window_softmax_decomposed
    from openvdn_comfy.communication import CommunicationRuntime
    from openvdn_comfy.attention_runtime import build_isolated_mask, isolated_mask_mod
    device = torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl',device_id=device,timeout=timedelta(minutes=10))
    runtime = init_ulysses()
    try:
        CommunicationRuntime(runtime).verify(layouts=tuple(range(1,8)))
        torch.manual_seed(7)
        layout = SequenceLayout(seq_len=1280, video_start=256,num_frames=4,tokens_per_frame=256)
        prefix = 193  # crosses a partial 128-token block; padding lies in the middle
        # Same strides as interleaved branch dispatch, rather than contiguous QKV.
        packed = torch.randn(1280,2,289,device=device,dtype=torch.bfloat16)
        q,k,v,_ = packed.split((96,96,96,1),-1)
        pos = torch.arange(1280,device=device)
        for bounds in ([(i,i) for i in range(4)],[(0,3)]*4):
            for anchors in ('none','both'):
                parts = isolated_mask_mod(layout,bounds,prefix,device)
                qv,kv,qf,kf,window,valid = parts(None,None,pos[:,None],pos[None,:])
                if anchors=='both':
                    window=window|(qf==0)|(qf==3)|(kf==0)|(kf==3)
                allowed=(~(qv&kv)|window)&valid
                Q,K,V=[x.transpose(0,1).float() for x in (q,k,v)]
                scores=(Q@K.transpose(-1,-2))*(96**-.5)
                reference=(scores.masked_fill(~allowed,float('-inf')).softmax(-1)@V).transpose(0,1)
                mask=build_isolated_mask(layout,bounds,prefix,device,anchors)
                got=window_softmax_flex(q,k,v,mask,96**-.5,inference=True)
                valid_rows=torch.cat([pos[:prefix],pos[256:]])
                torch.testing.assert_close(got[valid_rows].float(),reference[valid_rows],rtol=.03,atol=.015)
                # Decomposed comparison keeps all bucket keys, matching current unmasked semantics.
                allowed=~(qv&kv)|window
                reference=(scores.masked_fill(~allowed,float('-inf')).softmax(-1)@V).transpose(0,1)
                got=window_softmax_decomposed(q,k,v,layout,bounds,96**-.5,anchor_frames=anchors)
                torch.testing.assert_close(got.float(),reference,rtol=.03,atol=.015)
        dist.barrier()
        if runtime.rank==0:
            print('PASS: 7 communication layouts, uneven NCCL, FA4 isolated/full-cover masks, decomposed vs fp32')
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
