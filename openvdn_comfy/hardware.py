"""Deployment topology; request parameters never select devices or runtimes."""
from dataclasses import dataclass
import json
import os
import re


@dataclass(frozen=True)
class Hardware:
    gpu_type: str = 'h200'
    world_size: int = 8

    @classmethod
    def from_env(cls):
        gpu = os.environ.get('REF2VA_GPU_TYPE', 'h200').lower()
        size = int(os.environ.get('REF2VA_GPUS', '8'))
        if gpu not in ('h200', 'b200') or size not in (4, 8):
            raise ValueError('Choose --gpu-type h200|b200 and --gpus 4|8 (GPUs per worker)')
        return cls(gpu, size)

    @property
    def softmax_backend(self):
        return 'flex' if self.gpu_type == 'h200' else 'decomposed'

    @property
    def softmax_ranks(self):
        return (6 if self.gpu_type == 'h200' else 5) if self.world_size == 8 else (3 if self.gpu_type == 'h200' else 2)

    def visible_devices(self):
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(map(str, range(self.world_size)))).split(',')
        devices = [device.strip() for device in devices]
        if len(devices) != self.world_size or len(set(devices)) != self.world_size or not all(devices):
            raise ValueError(f'Expose exactly {self.world_size} distinct GPUs per worker')
        return devices

    def validate_device(self, name, memory, capability):
        expected = (9, 0) if self.gpu_type == 'h200' else (10, 0)
        minimum = 130 if self.gpu_type == 'h200' else 165
        if self.gpu_type.upper() not in name.upper() or tuple(capability) != expected or memory < minimum * 1024**3:
            raise RuntimeError(f'Expected full {self.gpu_type.upper()} GPU with CC {expected} and >= {minimum} GiB; '
                               f'got {name}, CC {capability}, {memory/1024**3:.1f} GiB')

    def metadata(self):
        return {'gpu_type': self.gpu_type, 'world_size': self.world_size, 'devices': self.visible_devices()}


def runtime_directory(root):
    instance = os.environ.get('REF2VA_INSTANCE', '')
    if instance and not re.fullmatch(r'worker-[01]', instance):
        raise ValueError('REF2VA_INSTANCE must be worker-0 or worker-1')
    return root / '.runtime' / 'instances' / instance if instance else root / '.runtime'


def install_workflows(root, user_directory, settings):
    """Separate starter names per hardware profile; preserve saved user edits."""
    hardware = Hardware.from_env()
    destination = user_directory / 'default/workflows'
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('openvdn_ref2va_like', 'openvdn_url_request'):
        target = destination / f'{name}_{hardware.gpu_type}_{hardware.world_size}.json'
        if target.exists():
            continue
        workflow = json.loads((root / 'workflows' / f'{name}.json').read_text())
        for node in workflow['nodes']:
            offset = {'OpenVDNH200Generate': 8, 'OpenVDNH200Request': 10}.get(node['type'])
            if offset is not None:
                node['widgets_values'][offset:offset + 2] = [settings.softmax_backend, settings.softmax_ranks]
        target.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + '\n')
