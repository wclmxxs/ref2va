#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
action=deploy
if [[ $# -gt 0 && "$1" != --* ]]; then action="$1"; shift; fi
if [[ "${1:-}" == --help ]]; then action=help; shift; fi
install_environment() {
    [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Deployment requires Linux x86_64'; exit 1; }
    for binary in git curl tar sha256sum gcc; do
      command -v "$binary" >/dev/null || { echo 'Install: git curl ca-certificates build-essential'; exit 1; }
    done
    mkdir -p .runtime/bin
    if [[ ! -x .runtime/bin/uv ]]; then
      curl --fail --location --retry 3 https://github.com/astral-sh/uv/releases/download/0.8.22/uv-x86_64-unknown-linux-gnu.tar.gz -o .runtime/uv.tar.gz
      printf '%s  %s\n' 741ff1f5742c5a4a25d2f829e8395355e43f7a5ae2ebc6368e9ae2df0efb69cf .runtime/uv.tar.gz | sha256sum -c -
      tar -xzf .runtime/uv.tar.gz -C .runtime/bin --strip-components=1
    fi
    uv="$ROOT/.runtime/bin/uv"
    "$uv" python install 3.12
    for kind in ui vdn; do
      if ! ".venv-$kind/bin/python" -c 'import sys; from pathlib import Path; assert sys.version_info[:2] == (3,12) and Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve()' "$ROOT/.venv-$kind" 2>/dev/null; then
        "$uv" venv --clear --python 3.12 ".venv-$kind"
      fi
    done
    .venv-ui/bin/python scripts/install_sources.py
    "$uv" pip install --python .venv-ui/bin/python --index-url https://download.pytorch.org/whl/cpu \
      torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
    "$uv" pip install --python .venv-ui/bin/python -r .deps/ComfyUI/requirements.txt -c constraints-comfy.txt
    "$uv" pip install --python .venv-vdn/bin/python --index-url https://download.pytorch.org/whl/cu129 torch==2.13.0 torchvision==0.28.0
    "$uv" pip install --python .venv-vdn/bin/python --prerelease=allow -e .deps/openvdn -c constraints-vdn.txt
    "$uv" pip install --python .venv-vdn/bin/python --prerelease=allow -e .deps/diffusers -c constraints-vdn.txt
    "$uv" pip check --python .venv-ui/bin/python
    "$uv" pip check --python .venv-vdn/bin/python
}

download_models() {
    [[ -x .venv-vdn/bin/python ]] || { echo 'Run ./deploy.sh install first'; exit 1; }
    .venv-vdn/bin/python scripts/download.py "$@"
}

check_runtime() {
    [[ -x .venv-vdn/bin/python ]] || { echo 'Run ./deploy.sh install first'; exit 1; }
    .venv-vdn/bin/python scripts/doctor.py "$@"
}

start_ui() {
    [[ -x .venv-ui/bin/python ]] || { echo 'Run ./deploy.sh install first'; exit 1; }
    runtime_dir=.runtime
    if [[ -n "${REF2VA_INSTANCE:-}" ]]; then
      [[ "$REF2VA_INSTANCE" == worker-0 || "$REF2VA_INSTANCE" == worker-1 ]] || { echo 'Invalid REF2VA_INSTANCE'; exit 1; }
      runtime_dir=".runtime/instances/$REF2VA_INSTANCE"
    fi
    mkdir -p output input "$runtime_dir/comfy-user/default/workflows"
    # Install the starter workflow once; keep any edits saved from the UI.
    if [[ ! -e "$runtime_dir/comfy-user/default/workflows/openvdn_ref2va_like.json" ]]; then
      cp workflows/openvdn_ref2va_like.json "$runtime_dir/comfy-user/default/workflows/"
    fi
    if [[ ! -e "$runtime_dir/comfy-user/default/workflows/openvdn_url_request.json" ]]; then
      cp workflows/openvdn_url_request.json "$runtime_dir/comfy-user/default/workflows/"
    fi
    exec .venv-ui/bin/python "$ROOT/scripts/serve.py" "$@"
}

case "$action" in
  deploy|up|restart)
    command -v python3 >/dev/null || { echo 'Install system python3 for the deployment launcher'; exit 1; }
    exec python3 "$ROOT/scripts/bootstrap.py" "$@"
    ;;
  install) install_environment ;;
  download) download_models "$@" ;;
  check) check_runtime "$@" ;;
  start)
    if [[ "${REF2VA_MANAGED_INSTANCE:-0}" != 1 ]]; then
      exec python3 "$ROOT/scripts/bootstrap.py" "$@"
    fi
    check_runtime
    start_ui "$@"
    ;;
  stop|status|logs)
    [[ -x .venv-ui/bin/python ]] || { echo 'Run ./deploy.sh install first'; exit 1; }
    exec .venv-ui/bin/python "$ROOT/scripts/fleet.py" "$action" "$@"
    ;;
  render)
    exec .venv-ui/bin/python scripts/render.py "$@"
    ;;
  help|-h|--help)
    echo 'Usage: bash deploy.sh [--gpu-type auto|h200|b200|b300] [--gpus 4|8] [--port 8188] [--wait-timeout seconds]'
    echo 'Checks/reuses dependencies and models, reloads changed code/config, waits for every API, then returns with services running.'
    echo 'Default: auto-detect GPU model, one 8-GPU worker. --gpus 4: two workers, ports 8188/8189.'
    echo 'Controls: status | logs | stop. Legacy deploy/start/up/restart are aliases for the unified startup.'
    echo 'Maintenance only: install | download | check [--nccl] | render --help'
    ;;
  *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
