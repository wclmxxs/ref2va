#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
action="${1:-deploy}"
if [[ $# -gt 0 ]]; then shift; fi
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
    [[ -x .venv-ui/bin/python ]] || "$uv" venv --python 3.12 .venv-ui
    [[ -x .venv-vdn/bin/python ]] || "$uv" venv --python 3.12 .venv-vdn
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
    mkdir -p output input .runtime/comfy-user/default/workflows
    # Install the starter workflow once; keep any edits saved from the UI.
    if [[ ! -e .runtime/comfy-user/default/workflows/openvdn_ref2va_like.json ]]; then
      cp workflows/openvdn_ref2va_like.json .runtime/comfy-user/default/workflows/
    fi
    if [[ ! -e .runtime/comfy-user/default/workflows/openvdn_url_request.json ]]; then
      cp workflows/openvdn_url_request.json .runtime/comfy-user/default/workflows/
    fi
    exec .venv-ui/bin/python "$ROOT/scripts/serve.py" "$@"
}

case "$action" in
  deploy)
    echo '[1/4] Installing pinned sources and Python environments'
    install_environment
    echo '[2/4] Downloading pinned model weights'
    download_models
    echo '[3/4] Checking H200 environment'
    check_runtime
    echo "[4/4] Releasing GPUs, preloading models, warming up, then starting ComfyUI on ${REF2VA_PORT:-8188}"
    start_ui "$@"
    ;;
  install) install_environment ;;
  download) download_models "$@" ;;
  check) check_runtime "$@" ;;
  start)
    check_runtime
    start_ui "$@"
    ;;
  up|restart|stop|status|logs)
    [[ -x .venv-ui/bin/python ]] || { echo 'Run ./deploy.sh install first'; exit 1; }
    exec .venv-ui/bin/python "$ROOT/scripts/service.py" "$action" "$@"
    ;;
  render)
    exec .venv-ui/bin/python scripts/render.py "$@"
    ;;
  help|-h|--help)
    echo 'Usage: bash deploy.sh [deploy | install | download | check [--nccl] | start | up | restart | stop | status | logs | render --help]'
    echo 'No arguments: install, download, check eight GPUs, and start ComfyUI.'
    echo 'up: background start with worker auto-recovery; restart: reload; stop/status/logs: service controls.'
    ;;
  *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
