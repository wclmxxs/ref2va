"""Bounded HTTP image fetching with destination validation on every connection."""
import asyncio
import hashlib
import io
import ipaddress
import json
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
from aiohttp.resolver import ThreadedResolver
from PIL import Image, ImageOps
from yarl import URL

from .config import RUNTIME

MAX_BYTES = 20 * 1024 * 1024


def require_public_ip(value):
    address = ipaddress.ip_address(value)
    if not address.is_global or (address.version == 6 and address.ipv4_mapped and not address.ipv4_mapped.is_global):
        raise ValueError("Reference image URLs must resolve to public IP addresses")


def validate_url(value):
    if not isinstance(value, str) or len(value) > 8192:
        raise ValueError("Invalid reference image URL")
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Reference images require HTTP(S) URLs without embedded credentials")
    if parsed.fragment:
        raise ValueError("Reference image URLs must not contain fragments")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        if parsed.hostname.lower() == "localhost" or parsed.hostname.lower().endswith(".localhost"):
            raise ValueError("Local reference URLs are not supported")
    else:
        require_public_ip(str(address))
    return value


def parse_urls(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [line.strip() for line in value.splitlines() if line.strip()]
    if not isinstance(value, list) or not 1 <= len(value) <= 9:
        raise ValueError("Provide 1–9 reference_image_urls")
    return [validate_url(url) for url in value]


class PublicResolver(ThreadedResolver):
    async def resolve(self, host, port=0, family=0):
        addresses = await super().resolve(host, port, family)
        for item in addresses:
            require_public_ip(item["host"])
        return addresses


def save_image(content, directory):
    if len(content) > MAX_BYTES:
        raise ValueError("Reference image exceeds 20 MiB")
    with Image.open(io.BytesIO(content)) as source:
        width, height = source.size
        if min(width, height) < 1 or max(width, height) > 8192 or not .25 <= width / height <= 4:
            raise ValueError("Reference image must be within 8192 pixels per side and 1:4–4:1 aspect ratio")
        source.load()
        image = ImageOps.exif_transpose(source).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    data = buffer.getvalue()
    path = directory / (hashlib.sha256(data).hexdigest() + ".png")
    directory.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(data)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return str(path)


async def fetch_one(session, url, directory, interrupt, *, saver=save_image, size_error=ValueError):
    for redirect in range(4):
        validate_url(url)
        interrupt()
        async with session.get(url, allow_redirects=False) as response:
            if response.status in (301, 302, 303, 307, 308):
                if "Location" not in response.headers:
                    raise ValueError("Image URL redirected without Location")
                url = str(URL(url).join(URL(response.headers["Location"])))
                continue
            response.raise_for_status()
            if response.content_length is not None and response.content_length > MAX_BYTES:
                raise size_error("Reference image exceeds 20 MiB")
            content = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                interrupt()
                content.extend(chunk)
                if len(content) > MAX_BYTES:
                    raise size_error("Reference image exceeds 20 MiB")
            return await asyncio.to_thread(saver, content, directory)
    raise ValueError("Too many reference image URL redirects")


async def download_references(urls, interrupt=lambda: None):
    connector = aiohttp.TCPConnector(resolver=PublicResolver(), use_dns_cache=False, limit=2)
    timeout = aiohttp.ClientTimeout(total=60, connect=10, sock_read=15)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=False) as session:
        # Preserve caller order for <Picture 1>, <Picture 2>, ...
        return [await fetch_one(session, url, RUNTIME / "url-references", interrupt) for url in parse_urls(urls)]
