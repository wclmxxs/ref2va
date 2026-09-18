"""8b200 gateway wire format mapped to the existing OpenVDN settings."""
from dataclasses import replace
import math
import os
import re
import secrets
import uuid

from .business_media import select_source
from .cache_dit import FIELDS as CACHE_FIELDS
from .config import Settings
from .optimization_options import FIELDS as OPTIMIZATION_FIELDS

PREFIX = '/ic/capcut/edit_gateway/v2'
CACHE_MAPPING = {'enabled': 'cache_dit', 'warmup': 'cache_dit_warmup_steps',
                 'rdt': 'cache_dit_threshold', 'max_continuous_cached_steps': 'cache_dit_max_consecutive',
                 'fn_blocks': 'cache_dit_fn_blocks', 'bn_blocks': 'cache_dit_bn_blocks',
                 'max_cached_steps': 'cache_dit_max_cached_steps', 'last_steps': 'cache_dit_last_steps'}
OPTIMIZATIONS = (*OPTIMIZATION_FIELDS, 'softmax_ranks', 'profile')
FIELDS = ('model', 'content', 'resolution', 'duration', 'ratio', 'num_inference_steps',
          'seed', 'reference_short_edge', 'optimization')


def model_name():
    return os.environ.get('BUSINESS_MODEL', 'MiniMax-H3')


def require_object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be an object')
    return value


def reject_unknown(value, allowed, name):
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f'Unknown {name} fields: {", ".join(sorted(unknown))}')


def validate_model(value):
    # Like 8b200, model aliases are accepted; they select this deployed model only.
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError('model must be a non-empty string of at most 128 characters')


def normalize_request(body, state):
    require_object(body, 'request')
    reject_unknown(body, FIELDS, 'request')
    validate_model(body.get('model'))
    content = body.get('content')
    if not isinstance(content, list) or not 1 <= len(content) <= 64:
        raise ValueError('content must contain 1–64 items')
    texts, sources = [], []
    for item in content:
        require_object(item, 'content item')
        kind = item.get('type')
        if kind == 'text':
            reject_unknown(item, ('type', 'role', 'text'), 'text content')
            if not isinstance(item.get('text'), str) or not item['text'].strip():
                raise ValueError('text must be non-empty')
            texts.append(item['text'].strip())
        elif kind == 'image_url' and item.get('role') == 'reference_image':
            reject_unknown(item, ('type', 'role', 'image_url'), 'image content')
            image = require_object(item.get('image_url'), 'image_url')
            reject_unknown(image, ('url', 'base64'), 'image_url')
            sources.append(select_source(image))
        else:
            raise ValueError('This service accepts text and image_url with role=reference_image only')
    prompt = '\n'.join(texts)
    if not prompt or len(prompt) > 24000:
        raise ValueError('Combined prompt must contain 1–24000 characters')
    if not 1 <= len(sources) <= 9:
        raise ValueError('Provide 1–9 reference images')
    resolution = body.get('resolution')
    if not isinstance(resolution, str) or not re.fullmatch(r'\d{3,4}P', resolution):
        raise ValueError('resolution must be an output short edge such as 768P or 512P')
    nfe = body.get('num_inference_steps')
    if nfe is not None and (type(nfe) is not int or nfe != 8):
        raise ValueError('This VDN checkpoint requires num_inference_steps=8')
    if 'duration' not in body or body['duration'] is None:
        raise ValueError('duration is required and must be 4–15 seconds')
    ratio = body.get('ratio')
    ratio = 'adaptive' if ratio is None else ratio
    seed = body.get('seed')
    seed = secrets.randbits(63) if seed is None else seed
    options = {k: v for k, v in state.get('profile', {}).items()
               if k in ('fp8', 'inference_kernels', 'softmax_backend', 'softmax_ranks', 'profile')}
    defaults = state.get('request_options', {})
    options.update({k: v for k, v in defaults.get('optimizations', {}).items() if k in OPTIMIZATIONS})
    options.update({k: v for k, v in defaults.get('cache_dit', {}).items() if k in CACHE_FIELDS})
    opt = body.get('optimization')
    if opt is not None:
        require_object(opt, 'optimization')
        reject_unknown(opt, (*OPTIMIZATIONS, 'cache_dit'), 'optimization')
        options.update({k: v for k, v in opt.items() if k in OPTIMIZATIONS and v is not None})
        cache = opt.get('cache_dit')
        if cache is not None:
            require_object(cache, 'optimization.cache_dit')
            reject_unknown(cache, CACHE_MAPPING, 'optimization.cache_dit')
            options.update({CACHE_MAPPING[k]: v for k, v in cache.items() if v is not None})
    settings = Settings(**options, seed=seed, reference_short_edge=body.get('reference_short_edge', 768),
                        duration=body['duration'], resolution=int(resolution[:-1]),
                        ratio='1:1' if ratio == 'adaptive' else ratio).validate()
    return {'model': model_name(), 'prompt': prompt, 'resolution': resolution, 'duration': settings.duration,
            'ratio': ratio, 'num_inference_steps': 8, 'seed': settings.seed,
            'reference_short_edge': settings.reference_short_edge}, settings, sources


def resolve_geometry(request, settings, metadata):
    if request['ratio'] == 'adaptive':
        width, height = metadata[0]['oriented_size']
        divisor = math.gcd(width, height)
        settings = replace(settings, ratio=f'{width // divisor}:{height // divisor}').validate()
    return settings


def task_id(job_id):
    return 'video_' + uuid.UUID(job_id).hex


def job_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'video_[0-9a-f]{32}', value):
        raise ValueError('Invalid task_id')
    return str(uuid.UUID(value[6:]))


def task_payload(record, base_url):
    body = record['business']
    metrics = record.get('metrics', {})
    timings = metrics.get('timings', {})
    upstream = metrics.get('upstream', {})
    status = {'interrupted': 'cancelled'}.get(record['status'], record['status'])
    task = {'id': task_id(record['job_id']), 'model': body['model'], 'status': status,
            'created_at': int(record['created_at']), 'updated_at': int(record['updated_at']),
            'inference_time_s': round(timings['worker_wall_seconds'], 3) if timings.get('worker_wall_seconds') is not None else None,
            **{k: body[k] for k in ('resolution', 'duration', 'ratio', 'seed', 'num_inference_steps', 'reference_short_edge')},
            'task_type': 'generation', 'modality': 'video', 'phase': record.get('phase'),
            'render_plan': record['render_plan'], 'timings': timings,
            'compilation': upstream.get('compilation'), 'cache_dit': upstream.get('cache_dit'),
            'optimizations': upstream.get('optimizations')}
    if status == 'succeeded':
        task['content'] = {'url': f'{base_url}{PREFIX}/video_generation/{task["id"]}/content'}
    elif status in ('failed', 'cancelled'):
        task['error'] = {'type': 'upstream_error' if status == 'failed' else 'cancelled',
                         'message': record.get('error', 'Video generation did not complete'), 'http_code': 500}
    return task
