"""Exercise the real deployment shell with fake external programs, without GPUs/downloads."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def executable(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    path.chmod(0o755)


@pytest.fixture
def deployment(tmp_path):
    root = tmp_path / "deployment with spaces"
    root.mkdir()
    shutil.copy(ROOT / "deploy.sh", root / "deploy.sh")
    (root / "workflows").mkdir()
    (root / "workflows/openvdn_ref2va_like.json").write_text('{"starter": true}')
    (root / "workflows/openvdn_url_request.json").write_text('{"url_starter": true}')
    mock = '#!/bin/sh\nprintf "%s:%s\\n" "$0" "$*" >> "$DEPLOY_TEST_LOG"\n'
    executable(root / ".runtime/bin/uv", mock)
    for env in ("ui", "vdn"):
        executable(root / f".venv-{env}/bin/python", mock +
                   'if [ "$1" = "$DEPLOY_TEST_FAIL" ]; then exit 7; fi\n')
    executable(root / "bin/python3", mock + 'if [ "$1" = "$DEPLOY_TEST_FAIL" ]; then exit 7; fi\n')
    for tool in ("git", "curl", "tar", "sha256sum", "gcc"):
        executable(root / "bin" / tool, "#!/bin/sh\nexit 0\n")
    executable(root / "bin/uname", '#!/bin/sh\nif [ "$1" = -s ]; then echo Linux; else echo x86_64; fi\n')
    env = {**os.environ, "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"],
           "DEPLOY_TEST_LOG": str(root / "calls.log"), "DEPLOY_TEST_FAIL": ""}
    return root, env


@pytest.mark.parametrize("prefix", [[], ["deploy"], ["start"], ["up"], ["restart"]])
def test_unified_launch_forwards_options_without_requiring_existing_venvs(deployment, prefix):
    root, env = deployment
    shutil.rmtree(root / '.venv-ui')
    shutil.rmtree(root / '.venv-vdn')
    subprocess.run(['bash', str(root / 'deploy.sh'), *prefix, '--gpu-type', 'b300', '--gpus', '4'],
                   cwd=root.parent, env=env, check=True, capture_output=True)
    calls = (root / 'calls.log').read_text().splitlines()
    assert len(calls) == 1
    assert calls[0].endswith(str(root / 'scripts/bootstrap.py') + ' --gpu-type b300 --gpus 4')


def test_managed_instance_checks_before_loading_and_preserves_user_workflow(deployment):
    root, env = deployment
    env.update(REF2VA_MANAGED_INSTANCE='1', REF2VA_INSTANCE='worker-1')
    for iteration in range(2):
        subprocess.run(['bash', str(root / 'deploy.sh'), 'start'], cwd=root.parent, env=env, check=True,
                       capture_output=True, text=True)
        workflow = root / '.runtime/instances/worker-1/comfy-user/default/workflows/openvdn_ref2va_like.json'
        assert workflow.read_text() == ('{"starter": true}' if iteration == 0 else '{"edited": true}')
        workflow.write_text('{"edited": true}')
    calls = (root / 'calls.log').read_text().splitlines()
    assert [line.split(':', 1)[1] for line in calls] == ['scripts/doctor.py', str(root / 'scripts/serve.py')] * 2


def test_failed_managed_precheck_does_not_spawn_worker(deployment):
    root, env = deployment
    env.update(REF2VA_MANAGED_INSTANCE='1', DEPLOY_TEST_FAIL='scripts/doctor.py')
    result = subprocess.run(['bash', str(root / 'deploy.sh'), 'start'], env=env, capture_output=True)
    assert result.returncode == 7
    assert 'serve.py' not in (root / 'calls.log').read_text()


@pytest.mark.parametrize("action", ["stop", "status", "logs"])
def test_background_controls_forward_without_loading_models(deployment, action):
    root, env = deployment
    subprocess.run(['bash', str(root / 'deploy.sh'), action], env=env, check=True, capture_output=True)
    calls = (root / 'calls.log').read_text().splitlines()
    assert len(calls) == 1
    assert calls[0].endswith(str(root / 'scripts/fleet.py') + ' ' + action)
