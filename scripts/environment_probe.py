"""Offline installed-package and editable-path checks, run inside each venv."""
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import sys
import tomllib
from urllib.parse import unquote, urlparse


def main():
    from packaging.requirements import Requirement
    root, kind = Path(sys.argv[1]).resolve(), sys.argv[2]
    expected_prefix = root / f'.venv-{kind}'
    if sys.version_info[:2] != (3, 12) or Path(sys.prefix).resolve() != expected_prefix:
        raise RuntimeError(f'{kind}: Python 3.12 venv must belong to {expected_prefix}')
    constraints = root / f'constraints-{"comfy" if kind == "ui" else "vdn"}.txt'
    requirements = [line.split('#', 1)[0].strip() for line in constraints.read_text().splitlines()]
    if kind == 'ui':
        requirements += (root / '.deps/ComfyUI/requirements.txt').read_text().splitlines()
    else:
        requirements += tomllib.loads((root / '.deps/openvdn/pyproject.toml').read_text())['project']['dependencies']
        for name, subdirectory in [('vdn', 'openvdn'), ('diffusers', 'diffusers')]:
            dist = metadata.distribution(name)
            direct = json.loads(dist.read_text('direct_url.json') or '{}')
            if (not direct.get('dir_info', {}).get('editable') or
                    Path(unquote(urlparse(direct.get('url', '')).path)).resolve() != root / '.deps' / subdirectory):
                raise RuntimeError(f'{name}: editable install belongs to another checkout')
        spec = importlib.util.find_spec('diffusers')
        if not spec or not Path(spec.origin).is_relative_to(root / '.deps/diffusers'):
            raise RuntimeError('Diffusers import resolves outside the patched source checkout')
    for text in requirements:
        text = text.split('#', 1)[0].strip()
        if not text:
            continue
        req = Requirement(text)
        if req.marker and not req.marker.evaluate({'extra': ''}):
            continue
        version = metadata.version(req.name)
        if not req.specifier.contains(version, prereleases=True):
            raise RuntimeError(f'{kind}: {req}, installed {version}')
    # Also validate transitive requirements; avoids depending on uv being present
    # in a working machine image merely to perform a read-only check.
    for dist in metadata.distributions():
        for text in dist.requires or []:
            req = Requirement(text)
            if req.marker and not req.marker.evaluate({'extra': ''}):
                continue
            version = metadata.version(req.name)
            if not req.specifier.contains(version, prereleases=True):
                raise RuntimeError(f'{dist.metadata["Name"]} requires {req}; installed {version}')
    import torch
    expected = '2.10.0+cpu' if kind == 'ui' else '2.13.0+cu129'
    if torch.__version__ != expected:
        raise RuntimeError(f'{kind}: expected torch {expected}, got {torch.__version__}')
    print(f'{kind}: pinned packages and Python paths verified')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
