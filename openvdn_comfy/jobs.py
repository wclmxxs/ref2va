import json
import time
import uuid

from .config import RUNTIME, atomic_json


def record_path(job_id):
    if str(uuid.UUID(job_id)) != job_id:
        raise ValueError("Invalid job ID")
    return RUNTIME / "api/jobs" / f"{job_id}.json"


def read_job(job_id):
    path = record_path(job_id)
    return json.loads(path.read_text()) if path.exists() else None


def create_job(job_id, request, plan, **metadata):
    record = {"job_id": job_id, "status": "queued", "phase": "queued", "request": request,
              "render_plan": plan.metadata(), "created_at": time.time(), "updated_at": time.time(), **metadata}
    atomic_json(record_path(job_id), record)
    return record


def update_job(job_id, **changes):
    if not job_id:
        return
    record = read_job(job_id)
    if record is not None:
        atomic_json(record_path(job_id), {**record, **changes, "updated_at": time.time()})
