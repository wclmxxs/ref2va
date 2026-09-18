"""Release request tensors promptly; avoid purging the CUDA allocator every job."""
import gc


class RequestCleanup:
    def __init__(self):
        self.count = 0

    def run(self, torch, device, policy, *, compiled=False, warmup=False):
        self.count += 1
        free, total = torch.cuda.mem_get_info(device)
        reason = ('always' if policy == 'always' else 'warmup' if warmup else
                  'compilation' if compiled else 'periodic' if self.count % 32 == 0 else
                  'memory_pressure' if free < total * .1 else None)
        if reason:
            gc.collect()
            torch.cuda.empty_cache()
        return {'policy': policy, 'performed': reason is not None, 'reason': reason,
                'request_count': self.count, 'free_bytes_before': free,
                'allocated_bytes_after': torch.cuda.memory_allocated(device),
                'reserved_bytes_after': torch.cuda.memory_reserved(device)}
