"""Gateway image inputs, resolved before queueing without storing base64 in jobs."""
import base64
import asyncio
import binascii
import hashlib
import io
from pathlib import Path

import aiohttp
from PIL import Image, UnidentifiedImageError

from .config import RUNTIME
from .references import MAX_BYTES, PublicResolver, fetch_one, save_image, validate_url

MAX_BASE64_CHARS = 4 * ((MAX_BYTES + 2) // 3)
MAX_PIXELS = 40_000_000


class ImageTooLarge(ValueError):
    pass


def select_source(image):
    if not isinstance(image, dict):
        raise ValueError('image_url must be an object containing url or base64')
    inline = image.get('base64')
    if inline is not None and not isinstance(inline, str):
        raise ValueError('image_url.base64 must be a string or null')
    value = inline.strip() if inline and inline.strip() else image.get('url')
    if not isinstance(value, str) or not value.strip():
        raise ValueError('image_url requires a non-empty base64 or url')
    value = value.strip()
    if inline and inline.strip():
        return {'kind': 'inline', 'value': value}
    if value.lower().startswith(('http://', 'https://')):
        return {'kind': 'url', 'value': validate_url(value)}
    # Match the gateway's legacy url-only data URI / base64 input.
    return {'kind': 'inline', 'value': value}


def decode_inline(value):
    if value.startswith('data:'):
        header, separator, value = value.partition(',')
        if not separator or header.lower() not in (
                'data:image/png;base64', 'data:image/jpeg;base64', 'data:image/webp;base64'):
            raise ValueError('Expected a JPEG, PNG or WebP base64 data URI')
    if len(value) > MAX_BASE64_CHARS:
        raise ImageTooLarge('Reference image exceeds 20 MiB')
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError('Invalid image base64') from error
    if not data:
        raise ValueError('Image base64 is empty')
    return data


def inspect_image(data):
    if len(data) > MAX_BYTES:
        raise ImageTooLarge('Reference image exceeds 20 MiB')
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ('PNG', 'JPEG', 'WEBP'):
                raise ValueError('Reference images must be JPEG, PNG or WebP')
            if image.width * image.height > MAX_PIXELS:
                raise ImageTooLarge('Reference image exceeds 40 million pixels')
            meta = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
                    'format': image.format, 'width': image.width, 'height': image.height}
            image.verify()
            return meta
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise ValueError('Invalid or damaged reference image') from error


async def prepare_sources(sources):
    directory = RUNTIME / 'business-references'
    paths, metadata = [], []
    connector = aiohttp.TCPConnector(resolver=PublicResolver(), use_dns_cache=False, limit=2)
    timeout = aiohttp.ClientTimeout(total=60, connect=10, sock_read=15)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=False) as session:
        for source in sources:
            info = {}

            def saver(data, target):
                info.update(inspect_image(data))
                try:
                    return save_image(data, target)
                except (OSError, SyntaxError) as error:
                    raise ValueError('Invalid or damaged reference image') from error

            if source['kind'] == 'inline':
                path = await asyncio.to_thread(lambda: saver(decode_inline(source['value']), directory))
            else:
                path = await fetch_one(session, source['value'], directory, lambda: None,
                                       saver=saver, size_error=ImageTooLarge)
                info['url'] = source['value']
            with Image.open(path) as image:
                info['oriented_size'] = [image.width, image.height]
            paths.append(str(Path(path).resolve()))
            metadata.append({'kind': source['kind'], **info})
    return paths, metadata
