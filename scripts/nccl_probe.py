"""Fail early on GPU topology/collective failures before loading any model."""
from datetime import timedelta
import os
import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print(f"NCCL_NVLS_ENABLE={os.environ.get('NCCL_NVLS_ENABLE', 'NCCL default')}; "
              "checking 8-rank all_reduce and all_to_all", flush=True)
    dist.init_process_group("nccl", device_id=torch.device('cuda', local_rank), timeout=timedelta(seconds=60))
    try:
        value = torch.tensor([float(local_rank)], device=f"cuda:{local_rank}")
        dist.all_reduce(value)
        if value.item() != 28 or dist.get_world_size() != 8:
            raise RuntimeError("8-rank NCCL all_reduce failed")
        source = torch.arange(8, device=f"cuda:{local_rank}") + 8 * local_rank
        target = torch.empty_like(source)
        dist.all_to_all_single(target, source)
        expected = torch.arange(8, device=f"cuda:{local_rank}") * 8 + local_rank
        if not torch.equal(target, expected):
            raise RuntimeError("8-rank NCCL all_to_all failed")
        dist.barrier()
        if local_rank == 0:
            print("8-rank NCCL all_reduce and all_to_all passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
