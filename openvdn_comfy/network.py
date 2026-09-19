"""Shared API bind addresses and local readiness targets; no machine IP is saved by default."""
from contextlib import ExitStack
import ipaddress
import os
import socket

DEFAULT_LISTEN = '0.0.0.0,::'


def listen_addresses(value=None):
    value = os.environ.get('REF2VA_LISTEN', DEFAULT_LISTEN) if value is None else value
    addresses = []
    for item in value.split(','):
        item = item.strip()
        if item.startswith('[') and item.endswith(']'):
            item = item[1:-1]
        try:
            address = str(ipaddress.ip_address(item))
        except ValueError as error:
            raise ValueError('--listen / REF2VA_LISTEN requires comma-separated IPv4/IPv6 addresses without ports') from error
        if address not in addresses:
            addresses.append(address)
    return tuple(addresses)


def listen_value(value=None):
    return ','.join(listen_addresses(value))


def health_urls(port, listen=None):
    urls = []
    for address in listen_addresses(listen):
        host = {'0.0.0.0': '127.0.0.1', '::': '::1'}.get(address, address)
        if ':' in host:
            host = '[' + host.replace('%', '%25') + ']'
        urls.append(f'http://{host}:{port}/openvdn/health')
    return urls


def check_bindings(bindings):
    """Check every (listen, port) together, matching asyncio's IPv6-only sockets."""
    with ExitStack() as stack:
        for listen, port in bindings:
            for address in listen_addresses(listen):
                family = socket.AF_INET6 if ':' in address else socket.AF_INET
                try:
                    probe = stack.enter_context(socket.socket(family, socket.SOCK_STREAM))
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    if family == socket.AF_INET6:
                        # Without this Linux's dual-mode wildcard can falsely conflict
                        # with the separate IPv4 socket used by aiohttp/ComfyUI.
                        probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                    probe.bind((address, int(port)))
                    probe.listen(1)
                except OSError as error:
                    raise RuntimeError(f'Cannot bind API address {address}, port {port}: {error}. '
                                       'Ensure the address is available and IPv6 is enabled; '
                                       'use --listen 0.0.0.0 only for IPv4-only deployments.') from error
