"""Fixed packed-prefix capacities without changing any real token or image geometry.

The native sequence is [text | references | audio | generated video]. Insert a
gap immediately before generated video. Text state still sees exactly the native
text rows (including its native 1/sqrt(length) scaling); softmax excludes gap
keys with a score modifier, including in fully covered blocks. Position IDs and
the sampler's noise draw order are untouched. This is not activation caching.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PrefixBucket:
    text_tokens: int
    reference_tokens: int
    audio_tokens: int
    video_tokens: int
    stride: int

    @property
    def prefix_tokens(self):
        return self.text_tokens + self.reference_tokens + self.audio_tokens

    @property
    def capacity(self):
        return ((self.prefix_tokens + self.stride - 1) // self.stride * self.stride
                if self.stride else self.prefix_tokens)

    @property
    def padding(self):
        return self.capacity - self.prefix_tokens

    def metadata(self):
        actual = self.prefix_tokens + self.video_tokens
        return {"enabled": bool(self.stride), "stride": self.stride,
                "conditioning_tokens": self.text_tokens, "conditioning_includes_vision": True,
                "reference_tokens": self.reference_tokens,
                "audio_tokens": self.audio_tokens, "video_tokens": self.video_tokens,
                "prefix_tokens": self.prefix_tokens, "prefix_capacity": self.capacity,
                "padding_tokens": self.padding, "actual_tokens": actual,
                "packed_tokens": actual + self.padding,
                "padding_fraction": self.padding / max(1, actual),
                "policy": "prefix_gap_v1" if self.stride else "native"}


def describe_bucket(embeds, conditions, plan, patch, audio_tokens, latent_frames, stride):
    references = sum(math.prod(c.shape[-3:]) // math.prod(patch)
                     for c in conditions[1]) if conditions else 0
    video = latent_frames * (plan.generation_height // 16 // patch[1]) * (plan.generation_width // 16 // patch[2])
    return PrefixBucket(embeds.shape[0], references, audio_tokens, video, stride)


def padding_score_mod(limits):
    """Tensor captures, not Python lengths: new valid lengths reuse the same graph.

    Use a coarse window BlockMask that includes the gap. Filtering in score_mod
    is necessary even for full blocks, which do not evaluate mask_mod in FA4.
    Padded queries can attend real keys, so they cannot produce all-masked NaNs.
    Their outputs never enter a real query, the linear text state or DBCache RDT.
    """
    def score_mod(score, batch, head, query, key):
        import torch
        valid = (key < limits[0]) | (key >= limits[1])
        return torch.where(valid, score, float("-inf"))
    return score_mod


class TokenBuckets:
    def __init__(self, stride=1024):
        self.stride = stride
        self.current = None
        self.limits = None
        self.score_mod = None

    @property
    def active(self):
        return self.current is not None and bool(self.current.stride)

    def prepare(self, bucket, device):
        import torch
        self.current = bucket
        if bucket.stride:
            if self.limits is None or self.limits.device != torch.device(device):
                self.limits = torch.empty(2, dtype=torch.int64, device=device)
                self.score_mod = padding_score_mod(self.limits)
            # Requests and their CUDA work are serialized by the resident worker.
            self.limits.copy_(torch.tensor([bucket.prefix_tokens, bucket.capacity], device=device))

    def pack_layout(self, layout):
        import torch
        if not self.active:
            return layout
        positions, tags, video, audio, text, conditions, extra = layout
        bucket = self.current
        start = int(video[conditions])
        if (start != bucket.prefix_tokens or len(positions) != bucket.prefix_tokens + bucket.video_tokens
                or conditions != bucket.reference_tokens or len(text) != bucket.text_tokens
                or len(audio) != bucket.audio_tokens):
            raise RuntimeError("Native packed layout does not match token bucket plan")
        if not bucket.padding:
            return layout
        # Keep all actual coordinates. Only the generated video's packed indices
        # move; the reference-video rows and text/audio mappings remain native.
        shifted_video = video.clone()
        shifted_video[conditions:] += bucket.padding
        positions = torch.cat((positions[:start], positions.new_zeros((bucket.padding, 3)), positions[start:]))
        tags = torch.cat((tags[:start], tags.new_zeros(bucket.padding), tags[start:]))
        return positions, tags, shifted_video, audio, text, conditions, extra

    def window_attention(self, attn, query, key, value, layout, bounds, scale):
        import torch
        from src.models.softmax_attention import flex_attention as flex
        mask = flex.build_window_block_mask(layout, bounds, value.device,
                                            anchor_frames=attn.anchor_frames)
        if query.is_cuda and torch.cuda.get_device_capability(query.device)[0] >= 10:
            query, key, value = (t.contiguous() for t in (query, key, value))
        compiled = flex._flex_attention_fn(
            inference=attn.inference_mode and flex.flash_backend_available(query.device))
        heads = value.shape[1]
        chunk = max(1, min(heads, (2**31 - 1) // (value.shape[0] * value.shape[2])))
        outputs = []
        for first in range(0, heads, chunk):
            tensors = [t[:, first:first + chunk].unsqueeze(0).transpose(1, 2)
                       for t in (query, key, value)]
            out = compiled(*tensors, block_mask=mask, scale=scale, score_mod=self.score_mod)
            outputs.append(out.squeeze(0).transpose(0, 1))
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)


def bucket_window_attention(attn, *args, **kwargs):
    state = attn._ref2va_buckets
    if state.active:
        return state.window_attention(attn, *args, **kwargs)
    return state.native_window_attention(attn, *args, **kwargs)


def attention_replacements():
    # Force even a full-cover softmax through the padding-aware Flex path. Keep
    # the native full-cover decision for the linear branch (which remains off).
    return [("if full_cover:", "if full_cover and not self._ref2va_buckets.active:"),
            ("elif self.softmax_impl in (\"flex\", \"decomposed\") and x.is_cuda:",
             'elif self._ref2va_buckets.active or (self.softmax_impl in ("flex", "decomposed") and x.is_cuda):')]


def verify_cuda_padding(state, device):
    """Exercise actual FA4, dense/full blocks, changing valid lengths and windows.

    A component check, not an end-to-end quality or bitwise-equality claim. Run
    before readiness: an unsupported score modifier must fail at startup.
    """
    import torch
    from types import SimpleNamespace
    from src.models.sequence_layout import SequenceLayout
    attn = SimpleNamespace(anchor_frames="none", inference_mode=True,
                           _window_kernel=lambda *args: "flex")
    generator = torch.Generator(device=device).manual_seed(112358)
    results = []
    for prefix in (35, 49):
        bucket = PrefixBucket(prefix - 16, 8, 8, 128, 64)
        state.prepare(bucket, device)
        native_layout = SequenceLayout(prefix + 128, prefix, 4, 32)
        padded_layout = SequenceLayout(bucket.capacity + 128, bucket.capacity, 4, 32)
        tensors = [torch.randn(prefix + 128, 2, 128, dtype=torch.bfloat16,
                               device=device, generator=generator) * .5 for _ in range(3)]
        padded = [torch.cat((t[:prefix], t.new_full((bucket.padding, 2, 128), 7.), t[prefix:]))
                  for t in tensors]
        for full in (False, True):
            bounds = [(0, 3)] * 4 if full else [(max(0, i - 1), min(3, i + 1)) for i in range(4)]
            expected = state.native_window_attention(attn, *tensors, native_layout, bounds, 128**-.5)
            actual = state.window_attention(attn, *padded, padded_layout, bounds, 128**-.5)
            actual = torch.cat((actual[:prefix], actual[bucket.capacity:]))
            torch.testing.assert_close(actual, expected, rtol=.02, atol=.002)
            results.append({"prefix": prefix, "full_cover": full,
                            "max_abs_error": (actual.float() - expected.float()).abs().max().item()})
    state.current = None
    return {"passed": True, "scope": "FA4 padding exclusion only", "rtol": .02, "atol": .002,
            "cases": results}
