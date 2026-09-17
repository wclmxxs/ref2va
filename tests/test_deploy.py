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
    mock = '#!/bin/sh\nprintf "%s:%s\\n" "$0" "$*" >> "$DEPLOY_TEST_LOG"\n'
    executable(root / ".runtime/bin/uv", mock)
    for env in ("ui", "vdn"):
        executable(root / f".venv-{env}/bin/python", mock +
                   'if [ "$1" = "$DEPLOY_TEST_FAIL" ]; then exit 7; fi\n')
    for tool in ("git", "curl", "tar", "sha256sum", "gcc"):
        executable(root / "bin" / tool, "#!/bin/sh\nexit 0\n")
    executable(root / "bin/uname", '#!/bin/sh\nif [ "$1" = -s ]; then echo Linux; else echo x86_64; fi\n')
    env = {**os.environ, "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"],
           "DEPLOY_TEST_LOG": str(root / "calls.log"), "DEPLOY_TEST_FAIL": ""}
    return root, env


def test_no_arguments_runs_every_stage_and_preserves_user_workflow(deployment):
    root, env = deployment
    for iteration in range(2):
        subprocess.run(["bash", str(root / "deploy.sh")], cwd=root.parent, env=env, check=True,
                       capture_output=True, text=True)
        workflow = root / ".runtime/comfy-user/default/workflows/openvdn_ref2va_like.json"
        assert workflow.read_text() == ('{"starter": true}' if iteration == 0 else '{"edited": true}')
        workflow.write_text('{"edited": true}')
    calls = (root / "calls.log").read_text().splitlines()
    stages = [line.split(":", 1)[1] for line in calls if "/bin/python:" in line]
    expected = ["scripts/install_sources.py", "scripts/download.py", "scripts/doctor.py --nccl"]
    assert stages[:3] == expected
    assert stages[3].startswith(".deps/ComfyUI/main.py --cpu")
    assert "--user-directory " + str(root / ".runtime/comfy-user") in stages[3]
    assert "--database-url sqlite:///" + str(root / ".runtime/comfy-user/comfyui.db") in stages[3]
    assert (root / ".runtime/comfy-user").is_dir()
    assert stages[4:7] == expected


@pytest.mark.parametrize("stage", ["scripts/install_sources.py", "scripts/download.py", "scripts/doctor.py"])
def test_deployment_stops_on_failed_stage(deployment, stage):
    root, env = deployment
    env["DEPLOY_TEST_FAIL"] = stage
    result = subprocess.run(["bash", str(root / "deploy.sh")], env=env, capture_output=True)
    assert result.returncode == 7
    calls = (root / "calls.log").read_text()
    assert ".deps/ComfyUI/main.py" not in calls
    if stage == "scripts/install_sources.py":
        assert "scripts/download.py" not in calls
    if stage != "scripts/doctor.py":
        assert "scripts/doctor.py" not in calls
