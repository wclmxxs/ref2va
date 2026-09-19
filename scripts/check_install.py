"""Read-only, offline readiness checks for reusable installations and VM images."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True, stderr=subprocess.STDOUT).strip()


def source_errors(root=ROOT):
    errors = []
    lock = json.loads((root / 'sources.lock.json').read_text())
    for name, source in lock['git'].items():
        path = root / '.deps' / name
        try:
            head = git(path, 'rev-parse', 'HEAD')
            if git(path, 'status', '--porcelain', '--untracked-files=no'):
                raise ValueError('managed source contains local edits')
            if git(path, 'config', '--get', 'remote.origin.url') != source['url']:
                raise ValueError('source remote changed')
            if name == 'diffusers':
                stamp = json.loads((path / '.git/openvdn-patches.json').read_text())
                patches = sorted((root / '.deps/openvdn/diffusers_patches').glob('*.patch'))
                digest = hashlib.sha256()
                for patch in patches:
                    digest.update(patch.name.encode()); digest.update(patch.read_bytes())
                if not patches or stamp != {'base': source['revision'], 'head': head, 'patches': digest.hexdigest()}:
                    raise ValueError('official Diffusers patch identity changed')
            elif head != source['revision']:
                raise ValueError('source pin changed')
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            errors.append(f'{name}: {error}')
    link = root / '.deps/ComfyUI/custom_nodes/openvdn_h200'
    if not link.is_symlink() or link.resolve() != root:
        errors.append('ComfyUI node link is missing or belongs to another checkout')
    return errors


def environment_errors(root=ROOT):
    errors = source_errors(root)
    for kind in ('ui', 'vdn'):
        python = root / f'.venv-{kind}/bin/python'
        try:
            probe = subprocess.run([str(python), str(root / 'scripts/environment_probe.py'), str(root), kind],
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            if probe.returncode:
                errors.append(f'{kind}: {probe.stdout[-4000:]}')
        except (OSError, subprocess.TimeoutExpired) as error:
            errors.append(f'{kind}: {error}')
    return errors


def model_errors(root=ROOT):
    models = Path(os.environ.get('REF2VA_MODELS', root / 'models')).resolve()
    lock = json.loads((root / 'sources.lock.json').read_text())['models']
    errors = []
    try:
        if json.loads((models / 'sources.json').read_text()) != lock:
            errors.append('model revisions do not match sources.lock.json')
    except (OSError, ValueError):
        errors.append('model download marker missing')
    for component in ('vdn/h3-base/transformer', 'vdn/h3-base/vae', 'vdn/h3-base/audio_vae',
                      'vdn/stage-dmd-step-250/linear_branch', 'vdn/stage-dmd-step-250/adapters/default',
                      'vdn/stage-dmd-step-250/adapters/turbo', 'conditioner/text_encoder'):
        files = list((models / component).glob('*.safetensors'))
        if not files or any(path.stat().st_size < 16 for path in files):
            errors.append(f'missing/empty weights: {component}')
    for index in models.rglob('*.safetensors.index.json'):
        try:
            for name in set(json.loads(index.read_text())['weight_map'].values()):
                if not (index.parent / name).is_file():
                    errors.append(f'missing shard: {index.parent / name}')
        except (OSError, ValueError, KeyError):
            errors.append(f'invalid shard index: {index}')
    if not (models / 'conditioner/processor/tokenizer_config.json').is_file():
        errors.append('missing Qwen3-VL processor')
    return errors
