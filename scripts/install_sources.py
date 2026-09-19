"""Fetch pinned public code and apply the exact official Diffusers patch set."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openvdn_comfy.config import DEPS, source_lock


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def patch_digest(patches):
    digest = hashlib.sha256()
    for patch in patches:
        digest.update(patch.name.encode())
        digest.update(patch.read_bytes())
    return digest.hexdigest()


def checkout(name, source):
    path = DEPS / name
    if not path.exists():
        subprocess.run(["git", "init", str(path)], check=True)
        git(path, "remote", "add", "origin", source["url"])
    # Compare the configured URL; get-url expands the user's legitimate insteadOf rules.
    if git(path, "config", "--get", "remote.origin.url") != source["url"]:
        raise RuntimeError(f"Unexpected Git remote: {path}")
    if git(path, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(f"Local modifications in {path}; preserve them before updating")
    pin = source["revision"]
    if name == "diffusers":
        patches = sorted((DEPS / "openvdn/diffusers_patches").glob("*.patch"))
        if not patches:
            raise RuntimeError("Official Diffusers patches missing")
        stamp = path / ".git/openvdn-patches.json"
        identity = {"base": pin, "patches": patch_digest(patches)}
        if stamp.exists():
            previous = json.loads(stamp.read_text())
            if all(previous.get(k) == v for k, v in identity.items()) and previous.get("head") == git(path, "rev-parse", "HEAD"):
                return
            raise RuntimeError(f"Managed patched Diffusers changed: preserve {path} then rerun install")
    if name != 'diffusers':
        try:
            if git(path, 'rev-parse', 'HEAD') == pin:
                return
        except subprocess.CalledProcessError:
            pass
    git(path, "fetch", "--depth", "1", "origin", pin)
    git(path, "checkout", "--detach", pin)
    if name == "diffusers":
        git(path, "-c", "user.name=OpenVDN local patches", "-c", "user.email=local@localhost",
            "am", *map(str, patches))
        stamp.write_text(json.dumps({**identity, "head": git(path, "rev-parse", "HEAD")}))


def main():
    DEPS.mkdir(parents=True, exist_ok=True)
    for name, source in source_lock()["git"].items():
        checkout(name, source)
    link = DEPS / "ComfyUI/custom_nodes/openvdn_h200"
    if link.is_symlink():
        if link.resolve() != ROOT:
            # A managed node link from a relocated VM image may be absolute.
            link.unlink()
            link.symlink_to(ROOT, target_is_directory=True)
    elif link.exists():
        raise RuntimeError(f"Custom node path already exists: {link}")
    else:
        link.symlink_to(ROOT, target_is_directory=True)
    print("Pinned sources and ComfyUI node installed")


if __name__ == "__main__":
    main()
