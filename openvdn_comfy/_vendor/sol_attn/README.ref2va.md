# Vendored Sol SM90 subset

Source: NVIDIA NVlabs/Sana, `sol-engine`, revision
`bb60499af0e675095ff67424196d8c18e265f32a`.
Only the Hopper kernel, its preprocessing, and required helpers are included.
`manifest.json` records each source and private-namespace file hash.
Root `__init__.py` deliberately does not load the square-only public interface.

Sol-H3's project page declares the code Apache-2.0:
https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/
The Apache-2.0 license text is included as `LICENSE.sol`.
FlashAttention-derived BSD-3-Clause license is retained as
`LICENSE.flash-attention` at this directory (relocated from upstream `sm100/`).
`THIRD_PARTY_NOTICES.md` is retained verbatim; its SM100/SM120 sections describe
upstream components not shipped in this SM90 subset.

Local rectangular host/preprocess and VDN-window adaptations live outside the
vendor directory in `sol_kernel.py`, `sol_plan.py`, and `sol_attention.py`.
The host disables the square recipe's unconditional full-Q-tile route reduction
when Tq is not divisible by 64, using SM90's existing guarded reduction.
The SM90 attention math is otherwise unchanged. No SM100/SM120 backend is exposed here.

The SM90 compatibility converter accepts CuTe 4.6's keyword arguments. Its
legacy NamedTuple handling is scoped to Sol compilation and restored in a
finally block, rather than changing the resident FA4 converter on import.

The private FlashAttention `fmax` helper uses the inferred-result NVVM binding
shipped in pinned CuTe 4.6.0.dev0, including its CUDA 12.9 build. The upstream
CUDA-version test incorrectly selected the old explicit-result signature on
that build. The operation, operands, attributes and reduction order are unchanged.

K/V summary reductions explicitly accumulate in FP32 before storing BF16
centroids/value sums. Triton 3.7.1's `sum` implementation retains floating input
dtype unless one is specified (despite its generated documentation describing
automatic promotion). BF16 intermediate rounding can change near-threshold
sparse routing decisions. Both the fused and separate summary kernels use
the same FP32 accumulation; summary storage and the attention kernel stay BF16.
Source: https://github.com/triton-lang/triton/blob/v3.7.1/python/triton/language/standard.py
