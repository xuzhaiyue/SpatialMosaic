"""UNI2-h features from existing HEST patches. Same PIL-bicubic + BF16 chain as T4."""
from __future__ import annotations

import datetime
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from t8_common import BATCH, HEST, OUT, PAUSE_SECONDS, SAMPLES, WEIGHTS, WORK, as_str_list
from uni2h_inference import load_encoder, prepare


def stamp() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def write_status(**payload: object) -> None:
    path = WORK / "run_status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"time": stamp(), **payload}, indent=2))
    temporary.replace(path)


def gpu_conflicts() -> list[str]:
    """List other running GPU inference scripts; extraction waits while any are present."""
    text = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
    hits: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            if int(pid_text) == os.getpid():
                continue
            parts = shlex.split(command)
        except ValueError:
            continue
        names = [Path(part).name.lower() for part in parts if part.endswith(".py")]
        if any("cellvit" in name for name in names):
            hits.append(stripped[:220])
    return hits


def device_utilization() -> list[int]:
    try:
        text = subprocess.run(["ioreg", "-r", "-c", "AGXAccelerator", "-l"], capture_output=True, text=True).stdout
    except OSError:
        return []
    return [int(value) for value in re.findall(r'"Device Utilization %"=(\d+)', text)]


def wait_for_gpu() -> None:
    while True:
        conflicts = gpu_conflicts()
        usage = device_utilization()
        busy = bool(conflicts) or (bool(usage) and max(usage) > 3)
        write_status(phase="waiting_for_gpu" if busy else "gpu_free", conflicts=conflicts, device_utilization=usage)
        if not busy:
            return
        print("WAIT GPU", conflicts, usage, flush=True)
        time.sleep(30)


def extract_sample(model: torch.nn.Module, sample: str) -> None:
    source = HEST / "patches" / f"{sample}.h5"
    destination = OUT / "features" / f"{sample}.h5"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as incoming, h5py.File(destination, "a") as outgoing:
        barcodes = as_str_list(incoming["barcode"][:])
        coords = np.asarray(incoming["coords"][:], dtype=np.int64)
        n_patches = len(barcodes)
        if "features" not in outgoing:
            outgoing.create_dataset("features", (n_patches, 1536), dtype="float32")
            outgoing.create_dataset("done", data=np.zeros(n_patches, dtype=bool))
            outgoing.create_dataset("barcode", data=np.asarray(barcodes, dtype=object), dtype=h5py.string_dtype())
            outgoing.create_dataset("patch_xy", data=coords)
            outgoing.attrs.update(source=str(source), interpolation="pil_bicubic", precision="bfloat16", fov_um=112.0, read_only_hest=True)
        else:
            stored = as_str_list(outgoing["barcode"][:])
            if stored != barcodes or not np.array_equal(outgoing["patch_xy"][:], coords):
                raise RuntimeError(f"{sample} existing feature file does not match the HEST patches")
        done = outgoing["done"]
        for start in range(0, n_patches, BATCH):
            indexes = [index for index in range(start, min(n_patches, start + BATCH)) if not bool(done[index])]
            if not indexes:
                continue
            while True:
                conflicts = gpu_conflicts()
                if not conflicts:
                    break
                write_status(phase="paused_for_other_gpu", sample=sample, conflicts=conflicts)
                time.sleep(30)
            tensors = []
            for index in indexes:
                image = Image.fromarray(np.asarray(incoming["img"][index]))
                tensors.append(prepare(image, "pil_bicubic"))
            with torch.inference_mode(), torch.autocast("mps", dtype=torch.bfloat16):
                values = model(torch.cat(tensors).to("mps")).float().cpu().numpy()
            torch.mps.synchronize()
            if not np.isfinite(values).all():
                raise RuntimeError(f"{sample} produced non-finite UNI2-h features")
            for row, index in enumerate(indexes):
                outgoing["features"][index] = values[row]
                done[index] = True
            outgoing.flush()
            finished = int(np.asarray(done[:]).sum())
            write_status(phase="extracting", sample=sample, done=finished, total=n_patches, batch=BATCH, pause_seconds=PAUSE_SECONDS)
            print(sample, finished, "/", n_patches, flush=True)
            time.sleep(PAUSE_SECONDS)
        if not np.asarray(done[:]).all():
            raise RuntimeError(f"{sample} extraction incomplete")
        outgoing.attrs["complete"] = True
        outgoing.flush()


def main() -> None:
    torch.set_num_threads(2)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    pending = []
    for sample in SAMPLES:
        path = OUT / "features" / f"{sample}.h5"
        if not path.exists():
            pending.append(sample)
            continue
        with h5py.File(path, "r") as handle:
            if not bool(handle.attrs.get("complete", False)):
                pending.append(sample)
    if not pending:
        write_status(phase="features_already_complete", samples=SAMPLES)
        print("FEATURES ALREADY COMPLETE", flush=True)
        return
    wait_for_gpu()
    write_status(phase="loading_encoder", weights=str(WEIGHTS))
    model = load_encoder(WEIGHTS).to("mps")
    for sample in SAMPLES:
        extract_sample(model, sample)
        time.sleep(PAUSE_SECONDS)
    (OUT / "features_complete.json").write_text(json.dumps({"samples": SAMPLES, "time": stamp()}, indent=2))
    write_status(phase="features_complete", samples=SAMPLES)
    print("FEATURES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
