import asyncio
import io
import json
from pathlib import Path
import sys
import types
import uuid

import pytest
import torch
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from openvdn_comfy import api, gpu_check, jobs, references
from openvdn_comfy.config import Settings
from openvdn_comfy.encode_output import requested_encoder
from openvdn_comfy.render_plan import make_plan


def request_body(**changes):
    return {"prompt": "Use <Picture 1> as the character reference.", "duration": 10, "ratio": "9:16",
            "resolution": 720, "reference_image_urls": ["https://example.com/first.png"], **changes}


def test_duration_ratio_short_edge_reach_the_worker():
    request, settings = api.normalize_request(request_body())
    plan = settings.render_plan()
    assert (plan.width, plan.height) == (720, 1280)
    assert (plan.generation_width, plan.generation_height) == (736, 1280)
    assert (plan.output_frames, plan.sampling_frames) == (240, 243)
    assert settings.inference_config("prompt.pt", "out.mp4")["render"]["num_frames"] == 243
    assert api.prompt_graph(request)["1"]["inputs"]["resolution"] == 720


def test_legacy_and_even_pixel_geometry():
    legacy = Settings().render_plan()
    assert (legacy.width, legacy.height, legacy.output_frames) == (1344, 768, 345)
    assert make_plan(duration=10, ratio="9:16", resolution=768).height == 1366
    assert make_plan(duration=15, ratio="1:1", resolution=512).sampling_frames == 362
    body = request_body(reference_image_url="https://example.com/a.png")
    del body["reference_image_urls"]
    assert api.normalize_request(body)[0]["reference_image_urls"] == ["https://example.com/a.png"]


def test_sampler_geometry_rejects_landscape_latents_for_portrait_request():
    plan = make_plan(duration=10, ratio="9:16", resolution=768)
    for shape in [(1, 24, 15, 48, 84), (1, 24, 15, 86, 46), (2, 24, 15, 86, 48), (86, 48)]:
        with pytest.raises(RuntimeError, match="Sampler geometry mismatch"):
            plan.validate_latent_shape(shape)
    assert plan.validate_latent_shape((1, 24, 15, 86, 48)) == {
        "validated": True, "latent_shape": [1, 24, 15, 86, 48],
        "generation_width": 768, "generation_height": 1376}


@pytest.mark.parametrize("changes", [{"duration": 0}, {"duration": float("nan")}, {"duration": True},
                                     {"duration": None}, {"ratio": "0:16"}, {"ratio": "1:8"},
                                     {"resolution": 721}, {"resolution": 2160}, {"resolution": "720p"},
                                     {"reference_image_urls": []}, {"prompt": ""}, {"unknown": 5}])
def test_invalid_api_input(changes):
    with pytest.raises(ValueError):
        api.normalize_request(request_body(**changes))


def test_output_frames_size_and_audio_are_adjusted_before_encoding():
    plan = make_plan(duration=4, ratio="9:16", resolution=256)
    frames = torch.full((107, 480, 256, 3), 73, dtype=torch.uint8)
    audio = torch.ones(2, 107 * 2000)  # 48 kHz / 24 fps
    recorded = {}
    def original(chunks, **kwargs):
        recorded.update(kwargs)
        actual = list(chunks)
        assert sum(chunk.shape[0] for chunk in actual) == 96
        assert all(chunk.shape[1:] == (456, 256, 3) and chunk.max() == 73 for chunk in actual)
        assert all(chunk.shape[0] <= 8 for chunk in actual)
    requested_encoder(original, plan)(frames, fps=24, output_path="unused.mp4", audio=audio, audio_sample_rate=48000)
    assert recorded["audio"].shape == (2, 192000)
    assert recorded["video_chunks_number"] == 12


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://127.0.0.1/x", "http://169.254.169.254/x",
                                  "http://[::1]/x", "http://[::ffff:127.0.0.1]/x", "http://localhost/x",
                                  "http://user:pass@example.com/x", "ftp://example.com/x"])
def test_reference_url_destinations(url):
    with pytest.raises(ValueError):
        references.validate_url(url)


def test_image_validation_and_content_cache(tmp_path):
    stream = io.BytesIO()
    Image.new("RGB", (64, 96), "blue").save(stream, format="PNG")
    first = references.save_image(stream.getvalue(), tmp_path)
    assert references.save_image(stream.getvalue(), tmp_path) == first
    assert Image.open(first).size == (64, 96)
    with pytest.raises(OSError):
        references.save_image(b"not an image", tmp_path)


def test_gpu_conflict_fails_before_loading(monkeypatch):
    output = "\n".join(f"{i}, GPU-{i}, {60000 if i == 3 else 140000}" for i in range(8))
    monkeypatch.setattr(gpu_check.subprocess, "check_output", lambda *a, **k: output)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.delenv("REF2VA_MIN_FREE_GPU_GIB", raising=False)
    with pytest.raises(RuntimeError, match="GPU 3.*SGLang"):
        gpu_check.ensure_free_gpus()
    monkeypatch.setattr(gpu_check.subprocess, "check_output", lambda *a, **k: output.replace("60000", "140000"))
    gpu_check.ensure_free_gpus()


def test_rest_submission_status_validation_and_restart(monkeypatch, tmp_path):
    async def prepare(urls):
        return [str(tmp_path / 'prepared.png') for _ in urls]
    monkeypatch.setattr(api, 'download_references', prepare)
    monkeypatch.setattr(jobs, "RUNTIME", tmp_path)
    state = {"ready": True, "profile": {key: getattr(Settings(), key) for key in api.PROFILE_FIELDS}}
    monkeypatch.setattr(api, "health", lambda: state)
    class Queue:
        def __init__(self):
            self.pending = []
        def put(self, item):
            self.pending.append(item)
        def get_history(self, job_id):
            return {}
        def get_current_queue_volatile(self):
            return [], self.pending
    async def validate(job_id, graph, target):
        assert graph["1"]["class_type"] == "OpenVDNH200BusinessRequest"
        return True, None, ["1"], {}
    monkeypatch.setitem(sys.modules, "execution", types.SimpleNamespace(validate_prompt=validate))
    server = types.SimpleNamespace(routes=web.RouteTableDef(), number=0, prompt_queue=Queue())
    api.register_routes(server)
    async def exercise():
        app = web.Application()
        app.add_routes(server.routes)
        async with TestClient(TestServer(app)) as client:
            assert (await client.get("/openvdn/health")).status == 200
            response = await client.post("/openvdn/jobs", json=request_body())
            assert response.status == 202
            accepted = await response.json()
            assert accepted["render_plan"]["width"] == 720
            job_id = accepted["job_id"]
            response = await client.get(accepted["status_url"])
            assert (await response.json())["status"] == "queued"
            jobs.update_job(job_id, status="succeeded", video_url="/view?filename=example.mp4")
            assert (await (await client.get(accepted["status_url"])).json())["video_url"].endswith("example.mp4")
            response = await client.post("/openvdn/jobs", json=request_body(duration=1))
            assert response.status == 400
            assert (await client.post("/openvdn/jobs", json=request_body(fp8=False))).status == 400
            assert (await client.post("/openvdn/jobs", json=request_body(cache_dit_threshold=-1))).status == 400
            response = await client.post("/openvdn/jobs", json=request_body(
                profile=True, softmax_ranks=4, cache_dit=True, cache_dit_threshold=.04))
            assert response.status == 202
            inputs = jobs.read_job(server.prompt_queue.pending[-1][1])["settings"]
            assert inputs["cache_dit"] is True and inputs["cache_dit_threshold"] == .04
            assert inputs["softmax_ranks"] == 4 and inputs["profile"] is True
            assert (await client.get("/openvdn/jobs/not-a-uuid")).status == 400
            assert (await client.get("/openvdn/jobs/" + str(uuid.uuid4()))).status == 404
            accepted = await (await client.post("/openvdn/jobs", json=request_body())).json()
            server.prompt_queue.pending.clear()
            assert (await (await client.get(accepted["status_url"])).json())["status"] == "interrupted"
            state["ready"] = False
            assert (await client.get("/openvdn/health")).status == 503
            assert (await client.post("/openvdn/jobs", json=request_body())).status == 503
    asyncio.run(exercise())
