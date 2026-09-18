"""Fail early on GPU topology/collective failures before loading any model."""
from datetime import timedelta
import os
import torch
import torch.distributed as dist


def main():
    size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print(f"NCCL_NVLS_ENABLE={os.environ.get('NCCL_NVLS_ENABLE', 'NCCL default')}; "
              f"checking {size}-rank all_reduce and all_to_all", flush=True)
    dist.init_process_group("nccl", device_id=torch.device('cuda', local_rank), timeout=timedelta(seconds=60))
    try:
        value = torch.tensor([float(local_rank)], device=f"cuda:{local_rank}")
        dist.all_reduce(value)
        if value.item() != size * (size - 1) // 2 or dist.get_world_size() != size:
            raise RuntimeError(f"{size}-rank NCCL all_reduce failed")
        source = torch.arange(size, device=f"cuda:{local_rank}") + size * local_rank
        target = torch.empty_like(source)
        dist.all_to_all_single(target, source)
        expected = torch.arange(size, device=f"cuda:{local_rank}") * size + local_rank
        if not torch.equal(target, expected):
            raise RuntimeError(f"{size}-rank NCCL all_to_all failed")
        dist.barrier()
        if local_rank == 0:
            print(f"{size}-rank NCCL all_reduce and all_to_all passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
