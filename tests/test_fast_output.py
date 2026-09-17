import ast
from fractions import Fraction
from pathlib import Path
import sys
import types

import av
import pytest
import torch

from openvdn_comfy import fast_output
from openvdn_comfy.encode_output import output_chunks
from openvdn_comfy.render_plan import make_plan


@pytest.fixture
def native_audio(monkeypatch):
    # Execute the pinned upstream audio helpers, including the real AAC encoder.
    root = Path(__file__).resolve().parents[1]
    path = root / 'work/patch-check/diffusers/src/diffusers/utils/export_utils.py'
    if not path.exists():
        path = root / '.deps/diffusers/src/diffusers/utils/export_utils.py'
    if not path.exists():
        pytest.skip('Pinned Diffusers sources not installed')
    tree = ast.parse(path.read_text())
    names = {'_prepare_audio_stream', '_write_audio', '_resample_audio'}
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = types.ModuleType('diffusers.utils.export_utils')
    module.__dict__['Fraction'] = Fraction
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_pixel_chunks_match_previous_pixel_math_and_trim():
    torch.manual_seed(42)
    plan = make_plan(duration=4, ratio='9:16', resolution=256)
    video = torch.randn(1, 3, 107, 48, 32, dtype=torch.float16)
    mean, std = (.485, .456, .406), (.229, .224, .225)
    full = (video.float() * torch.tensor(std).view(1,3,1,1,1) + torch.tensor(mean).view(1,3,1,1,1)).clamp(0,1)
    legacy = (full[0].permute(1,2,3,0)*255).round().to(torch.uint8)
    expected = torch.cat(list(output_chunks(legacy,plan)))
    times = {}
    actual = torch.cat([fast_output.pixel_chunk(video,i,plan,mean,std,times) for i in range(0,96,8)])
    assert torch.equal(actual, expected)
    assert actual.shape == (96,456,256,3)
    assert times['pixel_prepare_seconds'] > 0 and times['device_to_host_seconds'] >= 0


def test_real_mp4_has_requested_geometry_duration_audio_and_timings(tmp_path, native_audio):
    plan = make_plan(duration=4, ratio='9:16', resolution=256)
    video = torch.zeros(1,3,107,48,32)
    video[:,0,:48] = 1  # distinguish first/last frames and preserve order through prefetch
    audio = torch.zeros(2, 96000)  # pad 2-second audio to match 4-second video
    times, phases = {}, []
    path = tmp_path/'out.mp4'
    encoding = fast_output.write_mp4(video,audio,48000,plan,path,(.5,)*3,(.5,)*3,times,phases.append)
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        assert (stream.width,stream.height,stream.frames)==(256,456,96)
        assert float(stream.duration*stream.time_base)==4
        assert float(container.streams.audio[0].duration*container.streams.audio[0].time_base)==4
        frames=list(container.decode(video=0))
        assert frames[0].to_ndarray(format='rgb24')[0,0,0] > frames[-1].to_ndarray(format='rgb24')[0,0,0]+90
    assert not (tmp_path/'out.partial.mp4').exists()
    assert encoding['preset']=='veryfast' and encoding['component_times_overlap']
    assert all(times[key]>=0 for key in ('pixel_prepare_seconds','device_to_host_seconds','h264_encode_seconds','mux_seconds','audio_encode_and_mux_seconds','output_commit_seconds'))
    assert phases==['encoding_mp4']


def test_failed_pixel_stage_never_exposes_partial_mp4(tmp_path,native_audio,monkeypatch):
    def fail(*a,**k):raise RuntimeError('transfer failed')
    monkeypatch.setattr(fast_output,'pixel_chunk',fail)
    plan=make_plan(duration=4,ratio='9:16',resolution=256)
    with pytest.raises(RuntimeError,match='transfer failed'):
        fast_output.write_mp4(torch.zeros(1,3,107,48,32),torch.zeros(2,192000),48000,plan,tmp_path/'out.mp4',(.5,)*3,(.5,)*3,{})
    assert list(tmp_path.iterdir())==[]


def test_predecoded_video_is_not_decoded_again_and_time_is_counted_once(tmp_path, native_audio):
    plan = make_plan(duration=4, ratio='9:16', resolution=256)
    video = torch.zeros(1, 3, 107, 48, 32)
    audio_vae = types.SimpleNamespace(
        config=types.SimpleNamespace(latents_mean=[0.], latents_std=[1.], sampling_rate=48000),
        decode=lambda z, **kw: (torch.zeros(2, 1, 192000),))
    times, _ = fast_output.decode_and_save(None, torch.zeros(1, 1, 1), None, audio_vae,
                                         tmp_path/'parallel.mp4', 'cpu', plan, (.5,)*3, (.5,)*3,
                                         decoded_video=video, video_decode_seconds=2.)
    assert times['video_vae_decode_seconds'] == 2.
    assert times['output_wall_seconds'] >= 2. + times['audio_vae_decode_seconds']
    with av.open(str(tmp_path/'parallel.mp4')) as container:
        assert container.streams.video[0].frames == 96


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('resolution', [256, 258])
def test_preparation_matches_previous_math_with_uneven_tail(dtype, resolution):
    torch.manual_seed(7)
    plan = make_plan(duration=4.125, ratio='9:16', resolution=resolution)
    video = torch.randn(1, 3, plan.output_frames + 5, 48, 32, dtype=dtype)
    mean, std = (.485, .456, .406), (.229, .224, .225)
    legacy = (video.float() * torch.tensor(std).view(1,3,1,1,1) + torch.tensor(mean).view(1,3,1,1,1)).clamp(0,1)
    legacy = (legacy[0].permute(1,2,3,0)*255).round().to(torch.uint8)
    expected = torch.cat(list(output_chunks(legacy, plan)))
    actual = torch.cat([fast_output.prepare_pixels(video, i, plan, mean, std)
                        for i in range(0, plan.output_frames, 8)])
    assert torch.equal(actual, expected)


def test_encoder_failure_closes_prefetch_and_removes_partial(tmp_path, native_audio, monkeypatch):
    closed = []
    def broken_chunks(*args):
        try:
            yield 0, torch.zeros(8, 456, 256, 3, dtype=torch.uint8)
            raise RuntimeError('prefetch failed after first batch')
        finally:
            closed.append(True)
    monkeypatch.setattr(fast_output, 'threaded_pixels', broken_chunks)
    plan = make_plan(duration=4, ratio='9:16', resolution=256)
    with pytest.raises(RuntimeError, match='prefetch failed'):
        fast_output.write_mp4(torch.zeros(1,3,107,48,32), torch.zeros(2,192000), 48000, plan,
                             tmp_path/'broken.mp4', (.5,)*3, (.5,)*3, {})
    assert closed == [True] and not list(tmp_path.iterdir())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA pinned D2H pipeline required')
def test_cuda_two_buffers_preserve_pixels_and_stream_dependencies():
    plan = make_plan(duration=4.125, ratio='9:16', resolution=256)
    source_stream = torch.cuda.Stream()
    with torch.cuda.stream(source_stream):
        video = torch.randn(1, 3, plan.output_frames + 4, 48, 32, device='cuda')
        producer = fast_output.PinnedPixels(video, plan, (.5,)*3, (.5,)*3, {}, verify=True)
    chunks = list((start, chunk.clone()) for start, chunk in producer.chunks())
    expected = torch.cat([fast_output.pixel_chunk(video, i, plan, (.5,)*3, (.5,)*3, {})
                          for i in range(0, plan.output_frames, 8)])
    assert torch.equal(torch.cat([chunk for _, chunk in chunks]), expected)
    assert [start for start, _ in chunks] == list(range(0, plan.output_frames, 8))
    assert producer.checked_chunks == len(chunks)
    assert len(producer.buffers) == 2 and all(b.is_pinned() for b in producer.buffers)
    assert not producer.pending


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA pinned D2H pipeline required')
def test_cuda_real_mp4_matches_synchronous_encoder(tmp_path, native_audio, monkeypatch):
    plan = make_plan(duration=4.125, ratio='9:16', resolution=256)
    video = torch.randn(1,3,plan.output_frames+4,48,32, device='cuda')
    audio = torch.randn(2,198000, device='cuda') * .01
    files = [tmp_path/'sync.mp4', tmp_path/'async.mp4']
    for option, path in zip(('0', '1'), files):
        monkeypatch.setenv('REF2VA_ASYNC_OUTPUT', option)
        encoding = fast_output.write_mp4(video, audio, 48000, plan, path, (.5,)*3, (.5,)*3, {}, verify=True)
        assert encoding['async_pinned_output'] == (option == '1')
    for media in ('video', 'audio'):
        with av.open(str(files[0])) as a, av.open(str(files[1])) as b:
            first = [torch.from_numpy(f.to_ndarray()) for f in a.decode(**{media: 0})]
            second = [torch.from_numpy(f.to_ndarray()) for f in b.decode(**{media: 0})]
            assert len(first) == len(second)
            assert all(torch.equal(x, y) for x, y in zip(first, second))
