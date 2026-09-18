"""Optional model-free Hopper check. Run while GPUs are idle, before benchmarking."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    import torch.distributed as dist
    from openvdn_comfy.sol_kernel import SolKernel
    from openvdn_comfy.sol_reference import sol_reference
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if distributed:
        dist.init_process_group('nccl', device_id=device)
    try:
        kernel = SolKernel()
        report = {'rank': rank, 'verification': kernel.verify(device), 'layout_heads': []}
        generator = torch.Generator(device=device).manual_seed(33)
        # 56 softmax heads across 6, 5 or 4 ranks. Include every uneven shard size.
        for heads in (9, 10, 11, 12, 14):
            q = torch.randn((2, 129, heads, 128), device=device, dtype=torch.bfloat16, generator=generator)
            k, v = [torch.randn((2, 517, heads, 128), device=device, dtype=torch.bfloat16, generator=generator) for _ in range(2)]
            actual = kernel(q, k, v, scale=128**-.5, tau=1., sink_tokens=65)
            expected = sol_reference(q, k, v, scale=128**-.5, tau=1., sink_tokens=65)
            delta = actual.float()-expected
            relative = float(delta.norm()/expected.norm().clamp_min(1e-8))
            if not torch.isfinite(actual).all() or relative > .025 or float(delta.abs().max()) > .08:
                raise RuntimeError(f'Sol rank {rank}, heads {heads}: arithmetic mismatch {relative}')
            before = kernel.snapshot()
            again = kernel(q, k, v, scale=128**-.5, tau=1., sink_tokens=65)
            torch.cuda.synchronize()
            assert kernel.since(before)['compile_misses'] == 0
            torch.testing.assert_close(actual, again, rtol=0, atol=0)
            report['layout_heads'].append({'heads': heads, 'relative_l2': relative, 'hot_compile_misses': 0})
        records = [report]
        if distributed:
            records = [None]*dist.get_world_size()
            dist.all_gather_object(records, report)
        if rank == 0:
            print(json.dumps({'passed': True, 'by_rank': records}, indent=2))
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
