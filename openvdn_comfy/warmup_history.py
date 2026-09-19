"""Replay bounded, successful local conditioning caches before serving requests."""
from dataclasses import asdict
import json
from pathlib import Path

from .config import Settings, atomic_json


class WarmupHistory:
    def __init__(self, path, sources, profile, capacity=32):
        self.path, self.sources, self.profile = Path(path), sources, profile
        self.capacity = capacity
        self.entries = {}
        try:
            data = json.loads(self.path.read_text())
            if data.get("sources") == sources and data.get("profile") == profile:
                self.entries = dict(list(data.get("entries", {}).items())[-capacity:])
        except (OSError, ValueError, TypeError):
            pass

    def compatible(self, request):
        try:
            settings = Settings(**request["settings"]).validate()
            return (all(getattr(settings, name) == value for name, value in self.profile.items())
                    and Path(request["prompt_file"]).is_file())
        except (KeyError, TypeError, ValueError, OSError):
            return False

    def remember(self, geometry_id, request):
        if not self.compatible(request):
            return
        self.entries.pop(geometry_id, None)
        # Images and prompt text are already encoded in the local .pt. Never
        # redownload references or silently rebuild missing caches at startup.
        self.entries[geometry_id] = {"settings": request["settings"],
                                     "prompt_file": request["prompt_file"]}
        while len(self.entries) > self.capacity:
            self.entries.pop(next(iter(self.entries)))
        atomic_json(self.path, {"sources": self.sources, "profile": self.profile,
                               "entries": self.entries})

    def requests(self, limit, jobs_directory=None):
        if not limit:
            return []
        candidates = list(self.entries.values())
        if jobs_directory is not None:
            # Migration from schema 2: reuse completed requests from this
            # checkout. Bound metadata reads; ignore failed/incomplete jobs.
            paths = sorted(Path(jobs_directory).glob("*/result.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)[:200]
            migrated = []
            for path in reversed(paths):
                try:
                    record = json.loads(path.read_text())
                    if record.get("status") == "complete" and record.get("sources") == self.sources:
                        migrated.append(record)
                except (OSError, ValueError, TypeError):
                    continue
            # Fill missing older shapes even when an existing history only kept
            # eight entries. History entries remain the most recent candidates.
            candidates = migrated + candidates
        selected, seen = [], set()
        for candidate in reversed(candidates):
            if len(selected) >= limit:
                break
            if not self.compatible(candidate):
                continue
            settings = asdict(Settings(**candidate["settings"]).validate())
            # Seed/output settings do not alter the compiled geometry.
            settings["seed"] = 42
            settings["cache_dit"] = False
            settings["profile"] = False
            identity = (candidate["prompt_file"], settings["softmax_ranks"], settings['linear_kv_keep_ratio'],
                        json.dumps(Settings(**settings).render_plan().metadata(), sort_keys=True))
            if identity in seen:
                continue
            seen.add(identity)
            selected.append({"settings": settings, "prompt_file": candidate["prompt_file"],
                             "prompt": "", "references": [], "require_cached_prompt": True})
        return list(reversed(selected))
