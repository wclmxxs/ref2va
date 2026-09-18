"""Check the selected cards before loading another full model on them."""
import csv
import io
import os
import subprocess
from .hardware import Hardware


def selected_gpus():
    command = ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"]
    output = subprocess.check_output(command, text=True, timeout=15)
    rows = list(csv.reader(io.StringIO(output), skipinitialspace=True))
    by_id = {key: row for row in rows for key in row[:2]}
    devices = Hardware.from_env().visible_devices()
    try:
        selected = [by_id[device.strip()] for device in devices]
        if len({row[1] for row in selected}) != len(selected):
            raise ValueError("GPU indices and UUIDs refer to duplicate physical GPUs")
        return selected
    except KeyError as error:
        raise ValueError(f"GPU not found: {error.args[0]}") from error


def ensure_free_gpus():
    minimum_gib = float(os.environ.get("REF2VA_MIN_FREE_GPU_GIB", "90"))
    if not 0 < minimum_gib <= 280:
        raise ValueError("REF2VA_MIN_FREE_GPU_GIB must be in (0, 280]")
    selected = selected_gpus()
    busy = [row for row in selected if float(row[2]) < minimum_gib * 1024]
    if busy:
        detail = "; ".join(f"GPU {row[0]}: {float(row[2]) / 1024:.1f} GiB free" for row in busy)
        raise RuntimeError(f"Insufficient free GPU memory before model loading ({minimum_gib:g} GiB required per card). "
                           f"{detail}. Stop other GPU services, such as SGLang, and retry. No existing process was stopped.")
