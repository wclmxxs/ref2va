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
CUDA math is otherwise unchanged. No SM100/SM120 backend is exposed here.
