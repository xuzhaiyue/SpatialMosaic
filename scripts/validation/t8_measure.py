"""Measured E2F and IFNg UCell on the fixed background, plus the scale audit."""
from __future__ import annotations

import collections
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from t8_common import (
    BACKGROUND,
    FOV_UM,
    HEST,
    MEMBERS,
    OUT,
    PROGRAMS,
    SAMPLES,
    SHORT,
    SYMBOL_MAP,
    assert_definitions,
    as_str_list,
    load_json,
    ucell_matrix,
)


def gene_columns(handle: h5py.File, mapping: dict[str, str], background: list[str]) -> dict[str, int]:
    names = as_str_list(handle["var"]["_index"][:])
    if "gene_ids" in handle["var"]:
        raw = as_str_list(handle["var"]["gene_ids"][:])
    else:
        raw = names
    identifiers: list[str | None] = []
    for name, value in zip(names, raw):
        if value.startswith("ENSG"):
            identifiers.append(value.split(".")[0])
        else:
            hit = mapping.get(name)
            identifiers.append(hit if isinstance(hit, str) and hit.startswith("ENSG") else None)
    counts = collections.Counter(identifiers)
    background_index = set(background)
    columns: dict[str, int] = {}
    for index, gene in enumerate(identifiers):
        if gene and counts[gene] == 1 and gene in background_index:
            columns[gene] = index
    return columns


def read_counts(path: Path, columns: np.ndarray) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        matrix = handle["X"]
        n_spots = int(handle["obs"]["_index"].shape[0])
        out = np.zeros((n_spots, len(columns)), dtype=np.float64)
        if isinstance(matrix, h5py.Dataset):
            for start in range(0, n_spots, 256):
                block = np.asarray(matrix[start : start + 256], dtype=np.float64)
                out[start : start + len(block)] = block[:, columns]
            return out
        indptr = matrix["indptr"][:]
        indices = matrix["indices"][:]
        data = matrix["data"][:]
    mapped = np.full(int(max(indices.max() if len(indices) else 0, columns.max() if len(columns) else 0)) + 1, -1, dtype=np.int32)
    for position, column in enumerate(columns):
        mapped[int(column)] = position
    rows = np.repeat(np.arange(n_spots, dtype=np.int32), np.diff(indptr))
    destination = mapped[indices]
    keep = destination >= 0
    out[rows[keep], destination[keep]] = data[keep]
    return out


def nearest_pitch(coordinates: np.ndarray) -> float:
    distances, _ = cKDTree(coordinates).query(coordinates, k=2)
    return float(np.median(distances[:, 1]))


def main() -> None:
    assert_definitions()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "measured").mkdir(exist_ok=True)
    background = [line.strip() for line in BACKGROUND.read_text().splitlines() if line.strip()]
    members = load_json(MEMBERS)
    mapping = load_json(SYMBOL_MAP)
    scale_rows: list[dict] = []
    gene_rows: list[dict] = []
    for sample in SAMPLES:
        expression = HEST / "st" / f"{sample}.h5ad"
        patches = HEST / "patches" / f"{sample}.h5"
        meta = load_json(HEST / "metadata" / f"{sample}.json")
        mpp = float(meta["pixel_size_um_estimated"])
        embedded = meta.get("pixel_size_um_embedded")
        embedded_value = float(embedded) if embedded == embedded and embedded is not None else float("nan")
        with h5py.File(expression, "r") as handle:
            barcodes = as_str_list(handle["obs"]["_index"][:])
            spatial = np.asarray(handle["obsm"]["spatial"][:], dtype=np.float64)
            columns = gene_columns(handle, mapping, background)
        with h5py.File(patches, "r") as handle:
            patch_barcodes = as_str_list(handle["barcode"][:])
            patch_coords = np.asarray(handle["coords"][:], dtype=np.float64)
        lookup = {barcode: index for index, barcode in enumerate(barcodes)}
        if any(barcode not in lookup for barcode in patch_barcodes):
            raise RuntimeError(f"{sample} has a patch barcode missing from expression")
        matched = np.array([lookup[barcode] for barcode in patch_barcodes])
        half = (FOV_UM / 2) / mpp
        residual = np.linalg.norm(patch_coords + half - spatial[matched], axis=1)
        pitch_px = nearest_pitch(spatial)
        present = [gene for gene in background if gene in columns]
        member_indexes = []
        missing_members: dict[str, list[str]] = {}
        for program in PROGRAMS:
            kept = [gene for gene in members[program] if gene in columns]
            missing_members[program] = [gene for gene in members[program] if gene not in columns]
            member_indexes.append(np.array([present.index(gene) for gene in kept], dtype=np.int64))
        counts = read_counts(expression, np.array([columns[gene] for gene in present], dtype=np.int64))
        if counts.min() < 0:
            raise RuntimeError(f"{sample} counts are negative")
        scores = ucell_matrix(counts, member_indexes)
        if not np.isfinite(scores).all():
            raise RuntimeError(f"{sample} UCell is not finite")
        np.savez_compressed(
            OUT / "measured" / f"{sample}.npz",
            barcodes=np.asarray(barcodes),
            spatial_px=spatial,
            spatial_um=spatial * mpp,
            scores=scores,
            programs=np.asarray(PROGRAMS),
            paired_barcodes=np.asarray(patch_barcodes),
            mpp_estimated=np.array(mpp),
            background_genes=np.asarray(present),
        )
        scale_rows.append(
            {
                "sample": sample,
                "st_technology": meta.get("st_technology"),
                "preservation_method": meta.get("preservation_method"),
                "disease_comment": meta.get("disease_comment"),
                "n_expression": len(barcodes),
                "n_patches": len(patch_barcodes),
                "n_paired": len(patch_barcodes),
                "n_without_patch": len(barcodes) - len(patch_barcodes),
                "pixel_size_um_estimated": mpp,
                "pixel_size_um_embedded": embedded_value,
                "nearest_neighbor_px": pitch_px,
                "pitch_um_estimated": pitch_px * mpp,
                "pitch_um_embedded": pitch_px * embedded_value,
                "patch_center_residual_px_median": float(np.median(residual)),
                "patch_center_residual_px_max": float(residual.max()),
                "hest_patch_fov_um": FOV_UM,
                "training_fov_um": 128.0,
                "metadata_inter_spot_dist": meta.get("inter_spot_dist"),
                "metadata_spot_diameter": meta.get("spot_diameter"),
            }
        )
        gene_rows.append(
            {
                "sample": sample,
                "background_present": len(present),
                "background_missing": len(background) - len(present),
                "E2F_present": len(members[PROGRAMS[0]]) - len(missing_members[PROGRAMS[0]]),
                "E2F_missing": len(missing_members[PROGRAMS[0]]),
                "IFNg_present": len(members[PROGRAMS[1]]) - len(missing_members[PROGRAMS[1]]),
                "IFNg_missing": len(missing_members[PROGRAMS[1]]),
                "missing_E2F": ";".join(missing_members[PROGRAMS[0]]),
                "missing_IFNg": ";".join(missing_members[PROGRAMS[1]]),
                "count_min": float(counts.min()),
                "count_max": float(counts.max()),
                "integer_counts": bool(np.all(np.isclose(counts, np.round(counts)))),
            }
        )
        print(sample, "spots", len(barcodes), "pitch_um", round(pitch_px * mpp, 3), "UCell", scores.mean(0), flush=True)
    pd.DataFrame(scale_rows).to_csv(OUT / "scale_audit.csv", index=False)
    pd.DataFrame(gene_rows).to_csv(OUT / "gene_background_audit.csv", index=False)
    (OUT / "measured_complete.json").write_text(json.dumps({"samples": SAMPLES, "programs": SHORT}, indent=2))
    print("MEASURE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
