"""Project-local NVIDIA compiler provisioning, without touching drivers or Python wheels."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
# Official redistrib_12.9.1.json; keep components and hashes reproducible/offline reusable.
BASE_URL = 'https://developer.download.nvidia.com/compute/cuda/redist/'
COMPONENTS = {
    'cuda_nvcc': ('12.9.86', '7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35'),
    'cuda_cudart': ('12.9.79', '1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99'),
    'cuda_cccl': ('12.9.27', '8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa'),
}
RELEASE = '12.9.1'
REQUIRED = ('bin/nvcc', 'bin/ptxas', 'nvvm/bin/cicc', 'nvvm/libdevice/libdevice.10.bc',
            'include/cuda_runtime.h', 'include/crt/host_config.h', 'lib64/libcudart_static.a',
            'lib64/libcudadevrt.a', 'lib64/libculibos.a')


def toolkit_directory():
    return ROOT / '.runtime/toolchains' / ('cuda-' + RELEASE)


def compiler_candidates():
    # An explicit selection is authoritative; otherwise keep a prepared image self-contained.
    explicit = os.environ.get('REF2VA_NVCC')
    if explicit:
        return [Path(explicit)]
    paths = []
    if os.environ.get('CUDA_HOME'):
        paths.append(Path(os.environ['CUDA_HOME']) / 'bin/nvcc')
    paths.append(toolkit_directory() / 'bin/nvcc')
    if shutil.which('nvcc'):
        paths.append(Path(shutil.which('nvcc')))
    for pattern in ('/usr/local/cuda*/bin/nvcc', '/opt/cuda*/bin/nvcc', '/opt/nvidia/cuda*/bin/nvcc'):
        paths.extend(sorted(Path('/').glob(pattern.lstrip('/')), reverse=True))
    return list(dict.fromkeys(paths))


def inspect_compiler(path, major, minor):
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f'nvcc is missing or not executable: {path}')
    try:
        version = subprocess.check_output([str(path), '--version'], text=True, stderr=subprocess.STDOUT, timeout=15)
        supported = subprocess.check_output([str(path), '--list-gpu-code'], text=True, stderr=subprocess.STDOUT, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f'Cannot run nvcc {path}: {error}') from error
    arch = f'{major}{minor}'
    if f'sm_{arch}' not in supported.split():
        raise RuntimeError(f'nvcc {path} does not support sm_{arch}')
    return str(path), version, arch


def find_compiler(major, minor):
    failures = []
    for path in compiler_candidates():
        if not path.exists():
            continue
        try:
            return inspect_compiler(path, major, minor)
        except RuntimeError as error:
            failures.append(str(error))
    raise RuntimeError('No usable nvcc for sm_' + str(major) + str(minor) + '. '
                       'Run bash deploy.sh to prepare the CUDA compiler. ' + '; '.join(failures))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination, expected):
    # Publish only verified complete files; a failed download cannot poison image reuse.
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as out:
        partial = Path(out.name)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, out)
            out.flush()
            if sha256(partial) != expected:
                raise RuntimeError(f'CUDA archive SHA256 mismatch: {url}')
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)


def install_toolkit():
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('Automatic CUDA compiler installation requires Linux x86_64')
    target = toolkit_directory()
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(json.dumps(COMPONENTS))
    with (target.parent / 'install.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if (json.loads((target / 'components.json').read_text()) == manifest
                    and all((target / name).is_file() for name in REQUIRED)):
                return target
        except (OSError, ValueError):
            pass
        archives = target.parent / 'downloads'
        archives.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as staging:
            staging = Path(staging)
            assembled = staging / 'toolkit'
            assembled.mkdir()
            for name, (version, digest) in COMPONENTS.items():
                stem = f'{name}-linux-x86_64-{version}-archive'
                archive = archives / (stem + '.tar.xz')
                if not archive.exists() or sha256(archive) != digest:
                    print(f'Preparing CUDA compiler: downloading {name} {version}', flush=True)
                    download(BASE_URL + f'{name}/linux-x86_64/{archive.name}', archive, digest)
                extracted = staging / name
                with tarfile.open(archive) as package:
                    package.extractall(extracted, filter='data')
                component = extracted / stem
                for child in component.iterdir():
                    if child.is_dir():
                        shutil.copytree(child, assembled / child.name, dirs_exist_ok=True, symlinks=True)
                    elif child.name.startswith(('LICENSE', 'EULA')):
                        licenses = assembled / 'licenses' / name
                        licenses.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(child, licenses / child.name)
                    else:
                        shutil.copy2(child, assembled / child.name)
            # NVIDIA redistrib uses lib/; nvcc's Linux host linker searches lib64/.
            (assembled / 'lib64').symlink_to('lib', target_is_directory=True)
            if not all((assembled / name).is_file() for name in REQUIRED):
                raise RuntimeError('Downloaded CUDA compiler is incomplete')
            (assembled / 'components.json').write_text(json.dumps(manifest, indent=2))
            if target.exists():
                target.rename(staging / 'previous-toolkit')
            assembled.rename(target)
    return target


def check_compile(info):
    nvcc, _, arch = info
    with tempfile.TemporaryDirectory(prefix='ref2va-nvcc-') as folder:
        source = Path(folder) / 'probe.cu'
        source.write_text('#include <cuda_runtime.h>\n__global__ void probe(float* x) { x[0] += 1.f; }\n')
        result = subprocess.run([nvcc, '-std=c++17', '--shared', '-Xcompiler', '-fPIC',
                                 f'-arch=sm_{arch}', str(source), '-o', str(Path(folder) / 'probe.so')],
                                text=True, capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError(f'CUDA compiler preflight failed ({nvcc}):\n{result.stderr[-4000:]}')


def ensure_toolchain(major, minor):
    if not shutil.which('g++'):
        raise RuntimeError('CUDA compilation requires g++; install gcc-c++ (Amazon Linux/RHEL) '
                           'or build-essential (Ubuntu) and rerun the same startup command')
    try:
        info = find_compiler(major, minor)
        check_compile(info)
    except RuntimeError as error:
        if os.environ.get('REF2VA_NVCC'):
            raise  # Never override an explicit operator selection.
        print(f'{error}\nPreparing project-local CUDA {RELEASE} compiler.', flush=True)
        directory = install_toolkit()
        info = inspect_compiler(directory / 'bin/nvcc', major, minor)
        check_compile(info)
    print(f'CUDA compiler ready: {info[0]} (sm_{info[2]})', flush=True)
    return info[0]
