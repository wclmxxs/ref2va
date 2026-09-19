Derived from sgl-project/sglang, revision 5e9342d16f03621f8f434baca2bc4bbdfa4800c7, Apache-2.0.

Source: python/sglang/kernels/jit/csrc/diffusion/vdn_delta_factors.cuh
The device kernel is unchanged. The SGLang/TVM-FFI host binding is replaced by a small CUDA C ABI launcher.
The Python boundary-scan implementation in ../../boundary_scan.py is adapted from
python/sglang/multimodal_gen/runtime/models/dits/minimax_h3_vdn.py at the same revision.
Empty boundary-index tensors explicitly use int64; input/geometry checks are added by the adapter.
No SGLang runtime dependency is required. See LICENSE in this directory.
