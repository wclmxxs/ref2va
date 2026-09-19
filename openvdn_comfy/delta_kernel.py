"""Cached, current-stream CUDA binding to the audited SGLang FP32 delta kernel."""
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
from functools import lru_cache


@lru_cache(maxsize=4)
def compiler_info(major, minor):
    from .cuda_toolchain import find_compiler
    return find_compiler(major, minor)


def build_kernel(major, minor):
    import fcntl
    from .config import ROOT
    source = Path(__file__).parent/'_vendor/sglang_vdn/delta_factors.cu'
    nvcc, version, arch = compiler_info(major, minor)
    flags = ['-O3', '-std=c++17', '--shared', '-Xcompiler', '-fPIC', f'-arch=sm_{arch}']
    digest = hashlib.sha256(source.read_bytes()+repr((version, flags)).encode()).hexdigest()[:24]
    folder = ROOT/'.runtime/cuda-kernels'/digest
    folder.mkdir(parents=True, exist_ok=True)
    library = folder/'delta.so'
    # Eight ranks may launch together. Compile once and publish atomically; the
    # cache key includes toolkit, architecture and source for cloned machines.
    with (folder/'build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            with tempfile.TemporaryDirectory(dir=folder) as tmp:
                built = Path(tmp)/'delta.so'
                result = subprocess.run([nvcc, *flags, str(source), '-o', str(built)],
                                        capture_output=True, text=True, timeout=300)
                (folder/'build.log').write_text(result.stdout+result.stderr)
                if result.returncode:
                    raise RuntimeError(f'Delta CUDA compilation failed; see {folder / "build.log"}: {result.stderr[-1500:]}')
                built.replace(library)
    return library, {'architecture': f'sm_{arch}', 'cache_key': digest, 'library': str(library)}


def load_kernel(device):
    import torch
    library, report = build_kernel(*torch.cuda.get_device_capability(device))
    module = ctypes.CDLL(str(library))
    module.ref2va_delta_factors.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    module.ref2va_delta_factors.restype = ctypes.c_int
    module.ref2va_cuda_error.argtypes = [ctypes.c_int]
    module.ref2va_cuda_error.restype = ctypes.c_char_p
    return module, report


class DeltaKernel:
    def __init__(self):
        self.module = None
        self.verification = {'checked': False}

    def __call__(self, A, B, alpha):
        import torch
        if (A.device.type != 'cuda' or A.dtype != torch.float32 or B.dtype != A.dtype or alpha.dtype != A.dtype
                or B.device != A.device or alpha.device != A.device or A.shape[-2:] != (128,128)
                or B.shape != A.shape or alpha.shape != A.shape[:-1]):
            raise ValueError('Delta CUDA kernel requires same-device FP32 A/B [...,128,128] and alpha [...,128]')
        if self.module is None:
            self.module, self.build = load_kernel(A.device)
        def aligned(t):
            t = t.contiguous()
            return t if t.data_ptr() % 16 == 0 else t.clone()
        A, B, alpha = map(aligned, (A, B, alpha))
        transition, injection = torch.empty_like(A), torch.empty_like(B)
        code = self.module.ref2va_delta_factors(A.data_ptr(), B.data_ptr(), alpha.data_ptr(),
            transition.data_ptr(), injection.data_ptr(), A.numel()//16384,
            torch.cuda.current_stream(A.device).cuda_stream, A.device.index)
        if code:
            raise RuntimeError('Delta CUDA launch: ' + self.module.ref2va_cuda_error(code).decode())
        return transition, injection

    def verify(self, device):
        import torch
        if self.verification['checked']:
            return self.verification
        generator = torch.Generator(device=device).manual_seed(731)
        records = []
        # Include a stronger-conditioned system. Do not touch the render RNG.
        for scale in (1., 50.):
            k = torch.nn.functional.normalize(torch.randn(4,256,128,device=device,generator=generator),dim=-1)
            A = (k.transpose(-1,-2) @ k) * scale
            A = .5 * (A + A.transpose(-1,-2))
            B = torch.randn(A.shape,device=device,generator=generator)
            alpha = torch.rand(4,128,device=device,generator=generator)
            got = self(A, B, alpha)
            inv = torch.linalg.inv(A.double()+torch.eye(128,device=device,dtype=torch.float64))
            reference = (alpha.double().unsqueeze(-1)*inv, B.double()@inv)
            errors = [float((x.double()-ref).norm()/ref.norm().clamp_min(1e-30)) for x,ref in zip(got,reference)]
            if not all(torch.isfinite(x).all().item() for x in got) or max(errors) > 3e-5:
                raise RuntimeError(f'Fused delta arithmetic check failed: relative={errors}')
            records.append({'scale':scale, 'relative_errors':errors})
        self.verification = {'checked':True, 'passed':True, 'fp64_reference':records, **self.build}
        return self.verification
