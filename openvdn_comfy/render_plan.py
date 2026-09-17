"""Convert requested seconds/aspect/short edge to model and delivery geometry."""
from dataclasses import asdict, dataclass
import math
import re

FPS = 24


@dataclass(frozen=True)
class RenderPlan:
    width: int
    height: int
    generation_width: int
    generation_height: int
    output_frames: int
    sampling_frames: int
    fps: int = FPS

    def validate(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"Invalid render plan field: {name}")
        if self.fps != FPS or not 96 <= self.output_frames <= 362:
            raise ValueError("Output must be 4–15 seconds at 24 fps (legacy 17n+5 frames accepted)")
        if self.sampling_frames < self.output_frames or self.sampling_frames > 362 or self.sampling_frames % 17 != 5:
            raise ValueError("Invalid VAE frame alignment")
        if self.width % 2 or self.height % 2 or min(self.width, self.height) < 256:
            raise ValueError("Output dimensions must be even and at least 256 pixels")
        if self.generation_width != math.ceil(self.width / 32) * 32 or self.generation_height != math.ceil(self.height / 32) * 32:
            raise ValueError("Generation canvas must align the output to multiples of 32")
        if self.generation_width * self.generation_height > 1920 * 1088:
            raise ValueError("Generation canvas exceeds 1920×1088 pixels; reduce resolution")
        return self

    def metadata(self):
        return {**asdict(self), "duration": self.output_frames / self.fps}


def make_plan(num_frames=345, duration=None, ratio=None, resolution=None):
    if duration is None:
        output_frames = sampling_frames = num_frames
    else:
        if type(duration) not in (float, int) or not math.isfinite(duration) or not 4 <= duration <= 15:
            raise ValueError("duration must be 4–15 seconds")
        output_frames = round(duration * FPS)
        sampling_frames = output_frames + (5 - output_frames) % 17
    if ratio is None and resolution is None:
        width, height = 1344, 768
    else:
        ratio = ratio if ratio is not None else "16:9"
        resolution = resolution if resolution is not None else 768
        if type(resolution) is not int or resolution % 2 or not 256 <= resolution <= 1080:
            raise ValueError("resolution must be an even short-edge pixel count between 256 and 1080")
        if not isinstance(ratio, str) or not re.fullmatch(r"[1-9]\d{0,3}:[1-9]\d{0,3}", ratio):
            raise ValueError("ratio must be width:height, e.g. 16:9 or 9:16")
        rw, rh = map(int, ratio.split(":"))
        if not .25 <= rw / rh <= 4:
            raise ValueError("ratio must be between 1:4 and 4:1")
        if rw >= rh:
            height, width = resolution, 2 * round(resolution * rw / rh / 2)
        else:
            width, height = resolution, 2 * round(resolution * rh / rw / 2)
    return RenderPlan(width, height, math.ceil(width / 32) * 32, math.ceil(height / 32) * 32,
                      output_frames, sampling_frames).validate()
