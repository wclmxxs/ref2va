"""Small JSON API backed by ComfyUI's existing queue and execution engine."""
from dataclasses import asdict
import json
import sys
import time
import uuid

from aiohttp import web

from .config import Settings
from .jobs import create_job, read_job, update_job
from .references import parse_urls
from .backend import health, validate_profile, PROFILE_FIELDS

OPTIONS = ("seed", "reference_short_edge", "fp8", "inference_kernels", "softmax_backend",
           "softmax_ranks", "warmup_steps", "profile")


def normalize_request(body):
    if not isinstance(body, dict):
        raise ValueError("Request must be a JSON object")
    allowed = {"prompt", "duration", "ratio", "resolution", "reference_image_urls", "reference_image_url", *OPTIONS}
    unknown = set(body) - allowed
    if unknown:
        raise ValueError(f"Unknown request fields: {', '.join(sorted(unknown))}")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 24000:
        raise ValueError("prompt must be a non-empty string of at most 24000 characters")
    if "reference_image_url" in body and "reference_image_urls" in body:
        raise ValueError("Pass reference_image_url or reference_image_urls, not both")
    urls = parse_urls([body["reference_image_url"]] if "reference_image_url" in body else body.get("reference_image_urls"))
    settings = Settings(duration=body.get("duration", 5), ratio=body.get("ratio", "16:9"),
                        resolution=body.get("resolution", 720),
                        **{name: body[name] for name in OPTIONS if name in body}).validate()
    if settings.duration is None or settings.ratio is None or settings.resolution is None:
        raise ValueError("duration, ratio and resolution cannot be null")
    request = {name: getattr(settings, name) for name in ("duration", "ratio", "resolution", *OPTIONS)}
    request.update(prompt=prompt, reference_image_urls=urls)
    return request, settings


def prompt_graph(request):
    inputs = {**request, "reference_image_urls": json.dumps(request["reference_image_urls"])}
    return {"1": {"class_type": "OpenVDNH200Request", "inputs": inputs}}


def register_routes(server=None):
    if server is None:
        server_module = sys.modules.get("server")
        server = getattr(getattr(server_module, "PromptServer", None), "instance", None)
    if server is None or getattr(server, "_openvdn_routes_registered", False):
        return

    @server.routes.post("/openvdn/jobs")
    async def submit(request):
        backend = health()
        if not backend["ready"]:
            return web.json_response({"error": "OpenVDN backend is not ready", "backend": backend}, status=503)
        try:
            content = bytearray()
            async for chunk in request.content.iter_chunked(16384):
                content.extend(chunk)
                if len(content) > 131072:
                    raise web.HTTPRequestEntityTooLarge(max_size=131072, actual_size=len(content))
            body = json.loads(content)
            if isinstance(body, dict):
                body = {**backend["profile"], **body}
            normalized, settings = normalize_request(body)
            validate_profile(settings, backend)
        except (ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)
        import execution
        job_id = str(uuid.uuid4())
        graph = prompt_graph(normalized)
        valid = await execution.validate_prompt(job_id, graph, None)
        if not valid[0]:
            return web.json_response({"error": valid[1], "node_errors": valid[3]}, status=400)
        number = server.number
        server.number += 1
        create_job(job_id, normalized, settings.render_plan())
        server.prompt_queue.put((number, job_id, graph, {"create_time": int(time.time() * 1000)}, valid[2], {}))
        return web.json_response({"job_id": job_id, "status": "queued", "status_url": f"/openvdn/jobs/{job_id}",
                                  "render_plan": settings.render_plan().metadata()}, status=202)

    @server.routes.get("/openvdn/jobs/{job_id}")
    async def status(request):
        job_id = request.match_info["job_id"]
        try:
            record = read_job(job_id)
        except (ValueError, TypeError):
            return web.json_response({"error": "Invalid job ID"}, status=400)
        if record is None:
            return web.json_response({"error": "Job not found"}, status=404)
        if record["status"] not in ("succeeded", "failed", "interrupted"):
            history = server.prompt_queue.get_history(job_id).get(job_id)
            running, pending = server.prompt_queue.get_current_queue_volatile()
            if history and history.get("status", {}).get("status_str") == "error":
                errors = [message[1] for message in history["status"].get("messages", []) if message[0] == "execution_error"]
                update_job(job_id, status="failed", error=errors[-1].get("exception_message", "Execution failed") if errors else "Execution interrupted")
                record = read_job(job_id)
            elif not any(entry[1] == job_id for entry in [*running, *pending]):
                update_job(job_id, status="interrupted", error="Job removed from queue or server restarted")
                record = read_job(job_id)
        return web.json_response(record)

    @server.routes.get("/openvdn/health")
    async def backend_health(request):
        state = health()
        return web.json_response(state, status=200 if state["ready"] else 503)

    server._openvdn_routes_registered = True
