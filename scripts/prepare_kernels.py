"""Build native code before replacing the live GPU service; no CUDA context needed."""
from pathlib import Path
import ctypes
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openvdn_comfy.delta_kernel import build_kernel
from openvdn_comfy.hardware import GPU_SPECS, Hardware

if __name__ == '__main__':
    library, report = build_kernel(*GPU_SPECS[Hardware.from_env().gpu_type][0])
    ctypes.CDLL(str(library))
    print(f'Fused delta compiled and loadable: {report["architecture"]}; {library}', flush=True)
