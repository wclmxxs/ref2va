"""Compiler deployment is CPU-only, reproducible and happens before GPU replacement."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

import pytest

from openvdn_comfy import cuda_toolchain as cuda


def executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\n')
    path.chmod(0o755)
    return path


def test_compiler_search_skips_incompatible_path_and_uses_versioned_install(tmp_path, monkeypatch):
    old = executable(tmp_path / 'cuda-11/bin/nvcc')
    current = executable(tmp_path / 'cuda-12.9/bin/nvcc')
    monkeypatch.setattr(cuda, 'compiler_candidates', lambda: [tmp_path / 'absent', old, current])
    def output(command, **kwargs):
        if command[1] == '--version': return command[0]
        return 'sm_80\nsm_90\n' + ('sm_100\nsm_103\n' if command[0] == str(current) else '')
    monkeypatch.setattr(cuda.subprocess, 'check_output', output)
    assert cuda.find_compiler(10, 3) == (str(current), str(current), '103')
    assert cuda.find_compiler(9, 0)[0] == str(old)


def test_explicit_compiler_is_authoritative(tmp_path, monkeypatch):
    monkeypatch.setenv('REF2VA_NVCC', str(tmp_path / 'custom/bin/nvcc'))
    assert cuda.compiler_candidates() == [tmp_path / 'custom/bin/nvcc']
    with pytest.raises(RuntimeError, match='No usable nvcc'):
        cuda.find_compiler(10, 3)


def test_prepared_compiler_is_found_without_path_or_cuda_home(tmp_path, monkeypatch):
    monkeypatch.setattr(cuda, 'ROOT', tmp_path)
    monkeypatch.delenv('REF2VA_NVCC', raising=False)
    monkeypatch.delenv('CUDA_HOME', raising=False)
    nvcc = executable(cuda.toolkit_directory() / 'bin/nvcc')
    assert cuda.compiler_candidates()[0] == nvcc


def test_download_verifies_hash_before_replacing_cache(tmp_path, monkeypatch):
    target = tmp_path / 'archive.tar.xz'
    target.write_bytes(b'old verified copy')
    monkeypatch.setattr(cuda.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b'new content'))
    with pytest.raises(RuntimeError, match='SHA256 mismatch'):
        cuda.download('https://example.test/archive', target, 'bad')
    assert target.read_bytes() == b'old verified copy'
    assert list(tmp_path.iterdir()) == [target]
    cuda.download('https://example.test/archive', target, hashlib.sha256(b'new content').hexdigest())
    assert target.read_bytes() == b'new content'


def archive_bytes(name, version, files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:xz') as tar:
        for relative, data in files.items():
            entry = tarfile.TarInfo(f'{name}-linux-x86_64-{version}-archive/{relative}')
            entry.size = len(data)
            entry.mode = 0o755 if '/bin/' in relative or relative.startswith('bin/') else 0o644
            tar.addfile(entry, io.BytesIO(data))
    return buffer.getvalue()


def test_install_merges_real_layout_and_reuses_offline_after_relocation(tmp_path, monkeypatch):
    monkeypatch.setattr(cuda, 'ROOT', tmp_path)
    monkeypatch.setattr(cuda.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(cuda.platform, 'machine', lambda: 'x86_64')
    components = {
        'cuda_nvcc': {'bin/nvcc': b'nvcc', 'bin/ptxas': b'ptxas', 'nvvm/bin/cicc': b'cicc',
                      'nvvm/libdevice/libdevice.10.bc': b'libdevice', 'include/crt/host_config.h': b'host', 'LICENSE': b'nvcc-license'},
        'cuda_cudart': {'include/cuda_runtime.h': b'cuda', 'lib/libcudart_static.a': b'cudart',
                        'lib/libcudadevrt.a': b'device', 'lib/libculibos.a': b'culibos', 'LICENSE': b'cudart-license'},
    }
    archives = {name: archive_bytes(name, '1.2', files) for name, files in components.items()}
    monkeypatch.setattr(cuda, 'COMPONENTS', {name: ('1.2', hashlib.sha256(data).hexdigest()) for name, data in archives.items()})
    downloads = []
    def open_url(url, **kwargs):
        name = url.removeprefix(cuda.BASE_URL).split('/')[0]
        downloads.append(name)
        return io.BytesIO(archives[name])
    monkeypatch.setattr(cuda.urllib.request, 'urlopen', open_url)
    target = cuda.install_toolkit()
    assert all((target / path).is_file() for path in cuda.REQUIRED)
    assert (target / 'lib64').readlink() == Path('lib')
    assert (target / 'licenses/cuda_nvcc/LICENSE').read_bytes() == b'nvcc-license'
    assert len(downloads) == 2
    # A copied image neither downloads again nor retains absolute filesystem links.
    moved = tmp_path.parent / (tmp_path.name + '-image')
    tmp_path.rename(moved)
    monkeypatch.setattr(cuda, 'ROOT', moved)
    assert cuda.install_toolkit() == moved / '.runtime/toolchains/cuda-12.9.1'
    assert len(downloads) == 2
    # Incomplete installation is repaired from verified local archives.
    (cuda.toolkit_directory() / 'include/cuda_runtime.h').unlink()
    assert (cuda.install_toolkit() / 'include/cuda_runtime.h').is_file()
    assert len(downloads) == 2


@pytest.mark.parametrize('failure', ['missing', 'incomplete'])
def test_ensure_prepares_compiler_when_missing_or_headers_unusable(tmp_path, monkeypatch, failure):
    monkeypatch.delenv('REF2VA_NVCC', raising=False)
    monkeypatch.setattr(cuda.shutil, 'which', lambda _: '/usr/bin/g++')
    installed = []
    def find(*_):
        if failure == 'missing': raise RuntimeError('nvcc missing')
        return ('old-nvcc', 'old', '103')
    monkeypatch.setattr(cuda, 'find_compiler', find)
    def check(info):
        if info[0] == 'old-nvcc': raise RuntimeError('missing cuda_runtime.h')
        installed.append('checked')
    monkeypatch.setattr(cuda, 'check_compile', check)
    monkeypatch.setattr(cuda, 'install_toolkit', lambda: installed.append('installed') or tmp_path)
    monkeypatch.setattr(cuda, 'inspect_compiler', lambda path, *a: (str(path), 'new', '103'))
    assert cuda.ensure_toolchain(10, 3) == str(tmp_path / 'bin/nvcc')
    assert installed == ['installed', 'checked']


def test_native_build_is_reused_across_instances_and_separated_by_architecture(tmp_path, monkeypatch):
    from openvdn_comfy import delta_kernel, config
    monkeypatch.setattr(config, 'ROOT', tmp_path)
    monkeypatch.setattr(delta_kernel, 'compiler_info', lambda a, b: ('nvcc', '12.9', f'{a}{b}'))
    commands = []
    def compile(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b'shared-library')
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(delta_kernel.subprocess, 'run', compile)
    library, report = delta_kernel.build_kernel(10, 3)
    monkeypatch.setenv('REF2VA_INSTANCE', 'worker-1')
    assert delta_kernel.build_kernel(10, 3)[0] == library
    assert len(commands) == 1
    assert delta_kernel.build_kernel(9, 0)[0] != library
    assert len(commands) == 2
    assert report['architecture'] == 'sm_103'
