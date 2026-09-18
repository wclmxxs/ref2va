"""Install Sol without upgrading the resident FA4/CUDA stack; repair the 4.7 migration."""
from importlib import metadata
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CUTLASS_VERSION = '4.6.0.dev0'
PROTECTED = ('torch', 'torchvision', 'triton', 'flash-attn-4', 'quack-kernels')
# Introduced by CuTe 4.7, not used by 4.6. In particular cu12 pins libs-base
# to 4.7. Removing it after reinstalling 4.6 could remove overlapping files.
SPLIT_PACKAGES = ('nvidia-cutlass-dsl-libs-cu12', 'nvidia-cutlass-dsl-libs-core')


def versions():
    result = {}
    for name in (*PROTECTED, *SPLIT_PACKAGES, 'nvidia-cutlass-dsl', 'nvidia-cutlass-dsl-libs-base'):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return result


def install(installed, run=subprocess.run, *, root=ROOT, python=sys.executable):
    missing = set(PROTECTED)-set(installed)
    if missing:
        raise RuntimeError('Run bash deploy.sh install first; missing '+', '.join(sorted(missing)))
    if installed['flash-attn-4'] != '4.0.0b26':
        raise RuntimeError('This deployment requires pinned flash-attn-4==4.0.0b26; refusing to replace another FA4 stack')
    uv = str(root/'.runtime/bin/uv')
    # These are explicit roots, not merely constraints, so uv resolves quack's
    # and FA4's CuTe requirements together BEFORE mutating the environment.
    pinned = [f'{name}=={installed[name]}' for name in PROTECTED]
    command = [uv, 'pip', 'install', '--python', python, '--prerelease=allow',
               '-r', str(root/'requirements-sol.txt'), '-c', str(root/'constraints-vdn.txt'), *pinned]
    stale = [name for name in SPLIT_PACKAGES if name in installed]
    repair = bool(stale) or any(installed.get(name) != CUTLASS_VERSION for name in
                              ('nvidia-cutlass-dsl', 'nvidia-cutlass-dsl-libs-base'))
    if repair:
        # Also repairs partially overlapping installations from an earlier retry.
        command += ['--reinstall-package', 'nvidia-cutlass-dsl', '--reinstall-package', 'nvidia-cutlass-dsl-libs-base']
    print('Checking Sol with the installed FA4/quack/CUDA package versions', flush=True)
    run([*command, '--dry-run'], check=True)
    if stale:
        print('Removing CuTe 4.7 split libraries before restoring 4.6:', ', '.join(stale), flush=True)
        run([uv, 'pip', 'uninstall', '--python', python, *stale], check=True)
    run(command, check=True)
    run([uv, 'pip', 'check', '--python', python], check=True)
    # A fresh interpreter is essential after replacing files in our own venv.
    run([python, '-c',
         'from importlib.metadata import version; '
         'import flash_attn.cute.interface; import quack; '
         'from openvdn_comfy.sol_kernel import SolKernel; SolKernel.dependencies(); '
         'print("Native FA4/quack and Sol imports OK; CuTe=" + version("nvidia-cutlass-dsl") + '
         '"; GPU arithmetic check runs on first Sol request")'], cwd=root, check=True)


if __name__ == '__main__':
    install(versions())
