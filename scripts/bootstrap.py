"""One foreground command checks assets, loads services and returns only when ready."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import signal
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from check_install import environment_errors, model_errors
from openvdn_comfy.hardware import detect_hardware, GPU_SPECS
from openvdn_comfy.cuda_toolchain import ensure_toolchain


def run(command, **kwargs):
    process = subprocess.Popen(command, cwd=ROOT, **kwargs)
    try:
        code = process.wait()
    except KeyboardInterrupt:
        # The foreground child also receives SIGINT. Let fleet reap its detached
        # controllers before this launcher exits, instead of killing fleet first.
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=30)
        finally:
            signal.signal(signal.SIGINT, previous)
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def stop_before_repair(env):
    python = ROOT / '.venv-ui/bin/python'
    if python.exists():
        try:
            result = subprocess.run([str(python), str(ROOT / 'scripts/fleet.py'), 'stop'], cwd=ROOT, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except OSError as error:
            result = subprocess.CompletedProcess([], 1, stdout=str(error))
        if result.returncode:
            # On a partial install there may be no usable management environment.
            # Do not mutate a live service's packages if stopping it was impossible.
            for path in Path('/proc').glob('[0-9]*/cmdline'):
                try:
                    if str(ROOT / 'scripts/serve.py').encode() in path.read_bytes().split(b'\0'):
                        raise RuntimeError('Cannot stop an existing service before repairing dependencies: ' + result.stdout[-4000:])
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-type', choices=['auto', 'h200', 'b200', 'b300'], default=os.environ.get('REF2VA_GPU_TYPE', 'auto'))
    parser.add_argument('--gpus', type=int, choices=[4,8], default=int(os.environ.get('REF2VA_GPUS', '8')))
    parser.add_argument('--port', type=int, default=int(os.environ.get('REF2VA_PORT', '8188')))
    parser.add_argument('--wait-timeout', type=float, default=None)
    args, extras = parser.parse_known_args()
    if not 1 <= args.port <= 65535 - (args.gpus == 4):
        parser.error('Invalid API port range')
    if args.wait_timeout is not None and (not math.isfinite(args.wait_timeout) or args.wait_timeout <= 0):
        parser.error('--wait-timeout must be positive and finite')
    (ROOT / '.runtime').mkdir(parents=True, exist_ok=True)
    with (ROOT / '.runtime/deploy.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        gpu, devices = detect_hardware(args.gpu_type, os.environ.get('CUDA_VISIBLE_DEVICES'))
        env = {**os.environ, 'REF2VA_GPU_TYPE': gpu, 'REF2VA_GPUS': str(args.gpus),
               'REF2VA_PORT': str(args.port), 'CUDA_VISIBLE_DEVICES': devices}
        print(f'Detected {gpu.upper()}: {8//args.gpus} worker(s), {args.gpus} GPUs per worker', flush=True)
        errors = environment_errors(ROOT)
        if errors:
            print('Preparing Python environments and pinned sources:\n' + '\n'.join(errors), flush=True)
            stop_before_repair(env)
            run(['bash', str(ROOT / 'deploy.sh'), 'install'], env=env)
            errors = environment_errors(ROOT)
            if errors:
                raise RuntimeError('Environment verification failed after install:\n' + '\n'.join(errors))
        else:
            print('Installed dependencies and sources verified; reusing them without downloads.', flush=True)
        errors = model_errors(ROOT)
        if errors:
            print('Completing pinned model downloads:\n' + '\n'.join(errors), flush=True)
            stop_before_repair(env)
            run(['bash', str(ROOT / 'deploy.sh'), 'download'], env=env)
            errors = model_errors(ROOT)
            if errors:
                raise RuntimeError('Model verification failed after download:\n' + '\n'.join(errors))
        else:
            print('Pinned models verified; reusing existing weights.', flush=True)
        fused_delta = env.get('REF2VA_FUSED_DELTA', '1')
        if fused_delta not in ('0', '1'):
            raise ValueError('REF2VA_FUSED_DELTA must be 0 or 1')
        if fused_delta == '1':
            # Finish download/compiler/linker checks before fleet stops a live worker.
            env['REF2VA_NVCC'] = ensure_toolchain(*GPU_SPECS[gpu][0])
            run([str(ROOT / '.venv-vdn/bin/python'), str(ROOT / 'scripts/prepare_kernels.py')], env=env)
        command = [str(ROOT / '.venv-ui/bin/python'), str(ROOT / 'scripts/fleet.py'), 'ensure',
                   '--gpu-type', gpu, '--gpus', str(args.gpus), '--port', str(args.port)]
        if args.wait_timeout is not None:
            command += ['--wait-timeout', str(args.wait_timeout)]
        run([*command, *extras], env=env)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Startup interrupted.', file=sys.stderr)
        sys.exit(130)
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f'Startup failed: {error}', file=sys.stderr)
        sys.exit(1)
