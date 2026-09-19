"""Experimental unmasked prefix padding for fixed attention shapes.

The native sequence is [text | references | audio | generated video]. Insert a
gap immediately before generated video. Text state still sees exactly the native
text rows (including its native 1/sqrt(length) scaling). The native attention
path includes gap keys: padding may change generated content. Position IDs and
the sampler's noise draw order are untouched. This is not activation caching.
"""
from dataclasses import dataclass
import math

DEFAULT_BUCKET_STRIDE = 2048
BUCKET_POLICY = "prefix_gap_unmasked_v2"


def effective_bucket_stride(softmax_backend, configured_stride):
    """Flex and decomposed share the padded layout; ref remains an unpadded control.

    Decomposed includes every global-prefix row in both dense and window KV
    groups, so the gap has the same unmasked semantics as Flex. The selected
    attention implementation does not otherwise change.
    """
    return configured_stride if softmax_backend in ("flex", "decomposed") else 0


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
                "policy": BUCKET_POLICY if self.stride else "native",
                "padding_attention": "unmasked" if self.stride else "not_applicable"}


def describe_bucket(embeds, conditions, plan, patch, audio_tokens, latent_frames, stride):
    references = sum(math.prod(c.shape[-3:]) // math.prod(patch)
                     for c in conditions[1]) if conditions else 0
    video = latent_frames * (plan.generation_height // 16 // patch[1]) * (plan.generation_width // 16 // patch[2])
    return PrefixBucket(embeds.shape[0], references, audio_tokens, video, stride)


class TokenBuckets:
    def __init__(self, stride=DEFAULT_BUCKET_STRIDE):
        self.stride = stride
        self.current = None

    @property
    def active(self):
        return self.current is not None and bool(self.current.stride)

    def prepare(self, bucket, device):
        self.current = bucket

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
