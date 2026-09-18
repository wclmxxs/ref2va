"""8b200-style HTTP routes backed by ComfyUI's existing serialized GPU queue."""
import asyncio
from dataclasses import asdict
from functools import wraps
import json
import logging
import os
from pathlib import Path
import time
import uuid

import aiohttp
from aiohttp import web

from . import api
from .business_contract import PREFIX, job_id, normalize_request, require_object, resolve_geometry, task_id, task_payload, validate_model
from .business_media import ImageTooLarge, MAX_BASE64_CHARS, prepare_sources
from .jobs import create_job, update_job

MAX_BODY_BYTES = 9 * MAX_BASE64_CHARS + 131072


def error_response(status, message, **extra):
    kind = {400: 'invalid_request_error', 404: 'not_found_error', 409: 'conflict_error',
            413: 'payload_too_large_error', 502: 'upstream_error', 503: 'upstream_unavailable_error',
            504: 'timeout_error'}.get(status, 'internal_error')
    return web.json_response({'error': {'type': kind, 'message': message, 'http_code': status}, **extra}, status=status)


def business_errors(handler):
    @wraps(handler)
    async def wrapped(request):
        try:
            return await handler(request)
        except ImageTooLarge as error:
            return error_response(413, str(error))
        except (ValueError, TypeError, UnicodeError) as error:
            return error_response(400, str(error))
        except web.HTTPException as error:
            return error_response(error.status, error.reason)
        except asyncio.TimeoutError:
            return error_response(504, 'Reference image download timed out')
        except aiohttp.ClientError:
            return error_response(502, 'Unable to download reference image')
        except Exception:
            logging.exception('Gateway request failed: %s', request.path)
            return error_response(500, 'Internal gateway error; see server logs')
    return wrapped


async def read_body(request, maximum=MAX_BODY_BYTES):
    if request.content_length is not None and request.content_length > maximum:
        raise ImageTooLarge('Request body exceeds the supported size')
    content = bytearray()
    async for chunk in request.content.iter_chunked(65536):
        content.extend(chunk)
        if len(content) > maximum:
            raise ImageTooLarge('Request body exceeds the supported size')
    return require_object(json.loads(content), 'request')


def public_base(request):
    return os.environ.get('PUBLIC_BASE_URL', '').rstrip('/') or str(request.url.origin())


def get_record(server, value):
    record = api.current_job(server, job_id(value))
    if not record or 'business' not in record:
        raise web.HTTPNotFound(reason='Task not found')
    return record


async def submit(server, request):
    started, created_at = time.monotonic(), time.time()
    state = api.health()
    if not state.get('ready'):
        raise web.HTTPServiceUnavailable(reason='OpenVDN backend is not ready')
    body, settings, sources = normalize_request(await read_body(request), state)
    api.validate_profile(settings, state)
    paths, metadata = await prepare_sources(sources)
    settings = resolve_geometry(body, settings, metadata)
    current = api.health()
    if not current.get('ready') or current.get('instance') != state.get('instance'):
        raise web.HTTPServiceUnavailable(reason='OpenVDN backend changed during input preparation; submit again')
    api.validate_profile(settings, current)
    internal_id = str(uuid.uuid4())
    graph = {'1': {'class_type': 'OpenVDNH200BusinessRequest', 'inputs': {'job_id': internal_id}}}
    import execution
    valid = await execution.validate_prompt(internal_id, graph, None)
    if not valid[0]:
        raise ValueError('Invalid gateway execution graph: ' + str(valid[1]))
    # A queue item contains only an opaque ID, never user-provided paths or base64.
    queued_at = time.time()
    create_job(internal_id, {'prompt': body['prompt'], 'reference_images': metadata}, settings.render_plan(),
               business={k: v for k, v in body.items() if k != 'prompt'}, settings=asdict(settings),
               resolved_references=paths, created_at=created_at, queued_at=queued_at,
               reference_prepare_seconds=time.monotonic() - started)
    number = server.number
    server.number += 1
    try:
        server.prompt_queue.put((number, internal_id, graph, {'create_time': int(queued_at * 1000)}, valid[2], {}))
    except Exception:
        update_job(internal_id, status='failed', error='Failed to enqueue task')
        raise
    return task_id(internal_id)


def register_routes(server):
    @server.routes.post(PREFIX + '/video_generation')
    @business_errors
    async def generation(request):
        return web.json_response({'task_id': await submit(server, request)})

    @server.routes.post(PREFIX + '/query/video_generation')
    @business_errors
    async def query(request):
        body = await read_body(request, 131072)
        validate_model(body.get('model'))
        return web.json_response({'task': task_payload(get_record(server, body.get('task_id')), public_base(request))})

    @business_errors
    async def sync(request):
        timeout = float(os.environ.get('REF2VA_SYNC_TIMEOUT_SECONDS', '1800'))
        if not 0 < timeout <= 86400:
            raise web.HTTPServiceUnavailable(reason='Invalid server sync timeout setting')
        value = await submit(server, request)
        deadline = time.monotonic() + timeout
        while True:
            task = task_payload(get_record(server, value), public_base(request))
            if task['status'] in ('succeeded', 'failed', 'cancelled'):
                return web.json_response({'task': task}, status=200 if task['status'] == 'succeeded' else 500)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return error_response(504, 'Synchronous wait timed out; query the existing task_id', task_id=value)
            await asyncio.sleep(min(1., remaining))

    server.routes.post('/sync_infer')(sync)
    server.routes.post(PREFIX + '/sync_infer')(sync)

    @server.routes.get(PREFIX + '/video_generation/{task_id}/content')
    @business_errors
    async def content(request):
        record = get_record(server, request.match_info['task_id'])
        if record['status'] != 'succeeded':
            raise web.HTTPConflict(reason='Task is not succeeded')
        import folder_paths
        root = (Path(folder_paths.get_output_directory()) / 'openvdn').resolve()
        output = Path(record.get('metrics', {}).get('output', '')).resolve()
        if not output.is_relative_to(root) or output.suffix != '.mp4' or not output.is_file():
            raise web.HTTPNotFound(reason='Video output is unavailable')
        return web.FileResponse(output, headers={'Content-Type': 'video/mp4',
            'Content-Disposition': f'inline; filename="{request.match_info["task_id"]}.mp4"'})
