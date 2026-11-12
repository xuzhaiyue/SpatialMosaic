"""Predict with the frozen breast model and draw the T8-1 comparison figures."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from t8_common import (
    BLOCK_UM,
    FOV_UM,
    HEST,
    MODEL,
    MODEL_SHA256,
    OUT,
    PROGRAMS,
    SAMPLES,
    SHORT,
    WEIGHTS,
    as_str_list,
    coloc_pair,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def predict_st(model: dict, features: np.ndarray) -> np.ndarray:
    return (features @ model["coef"].T + model["intercept"]) * model["ST_scale"] + model["ST_mean"]


def safe_spearman(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(left) & np.isfinite(right)
    n_pairs = int(mask.sum())
    if n_pairs < 3 or np.ptp(left[mask]) == 0 or np.ptp(right[mask]) == 0:
        return float("nan"), n_pairs
    return float(spearmanr(left[mask], right[mask]).statistic), n_pairs


def percentile(values: np.ndarray) -> np.ndarray:
    return 100.0 * rankdata(values, method="average") / len(values)


def load_image(sample: str) -> tuple[np.ndarray, float]:
    with h5py.File(HEST / "st" / f"{sample}.h5ad", "r") as handle:
        image = np.asarray(handle["uns"]["spatial"]["ST"]["images"]["downscaled_fullres"][:])
        scale = float(handle["uns"]["spatial"]["ST"]["scalefactors"]["tissue_downscaled_fullres_scalef"][()])
    return image, scale


def draw_map(sample: str, frame: pd.DataFrame, technology: str, spot_rho: dict[str, float]) -> None:
    image, scale = load_image(sample)
    mpp = float(frame["mpp"].iloc[0])
    width_um = image.shape[1] / scale * mpp
    height_um = image.shape[0] / scale * mpp
    figure, axes = plt.subplots(2, 2, figsize=(8.4, 8.6), constrained_layout=True)
    panels = [
        (0, 0, "E2F_observed", "E2F measured"),
        (0, 1, "E2F_predicted", "E2F predicted"),
        (1, 0, "IFNg_observed", "IFNg measured"),
        (1, 1, "IFNg_predicted", "IFNg predicted"),
    ]
    xy = frame[["x_um", "y_um"]].to_numpy()
    inside = ((xy[:, 0] >= 0) & (xy[:, 0] <= width_um) & (xy[:, 1] >= 0) & (xy[:, 1] <= height_um)).mean()
    for row, column, field, title in panels:
        axis = axes[row, column]
        axis.imshow(image, extent=(0, width_um, height_um, 0), interpolation="bilinear")
        colors = percentile(frame[field].to_numpy())
        points = axis.scatter(xy[:, 0], xy[:, 1], c=colors, s=7, cmap="viridis", vmin=0, vmax=100, linewidths=0)
        axis.set_xlim(0, width_um)
        axis.set_ylim(height_um, 0)
        axis.set_aspect("equal")
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("x (µm)")
        axis.set_ylabel("y (µm)")
        axis.tick_params(labelsize=8)
    figure.colorbar(points, ax=axes, shrink=0.72, label="Within-section percentile")
    figure.suptitle(
        f"{sample} · {technology}\nE2F Spearman {spot_rho['E2F']:.3f}    IFNγ Spearman {spot_rho['IFNg']:.3f}    spots {len(frame)}    on image {inside:.1%}",
        fontsize=12,
    )
    stem = OUT / "figures" / "maps" / sample
    figure.savefig(stem.with_suffix(".png"), dpi=150)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)
    if inside < 0.9:
        raise RuntimeError(f"{sample} spots are not on the downscaled H&E ({inside:.3f})")


def draw_blocks(blocks: pd.DataFrame, summary: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(9.2, 6.4), constrained_layout=True)
    for axis, sample in zip(axes.ravel(), SAMPLES):
        part = blocks[blocks["sample"].eq(sample) & np.isfinite(blocks["observed_coloc"]) & np.isfinite(blocks["predicted_coloc"])]
        axis.plot([-1, 1], [-1, 1], color="#b0b0b0", lw=0.8)
        axis.scatter(part["observed_coloc"], part["predicted_coloc"], s=22, c="#1f4e79", linewidths=0)
        row = summary[summary["sample"].eq(sample)].iloc[0]
        rho = row["block_spearman"]
        rho_text = "NA" if not np.isfinite(rho) else f"{rho:.3f}"
        axis.set_xlim(-1.05, 1.05)
        axis.set_ylim(-1.05, 1.05)
        axis.set_aspect("equal")
        axis.set_title(f"{sample}  ρ {rho_text}\nblocks {int(row['n_blocks_finite'])}/{int(row['n_blocks'])}", fontsize=9)
        axis.set_xlabel("Measured coloc")
        axis.set_ylabel("Predicted coloc")
        axis.tick_params(labelsize=8)
    figure.suptitle("E2F–IFNγ colocalization inside 1 mm blocks", fontsize=12)
    stem = OUT / "figures" / "blocks" / "block_colocalization"
    figure.savefig(stem.with_suffix(".png"), dpi=150)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)


def draw_summary(summary: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), constrained_layout=True)
    positions = np.arange(len(summary))
    axes[0].axvline(0, color="#b0b0b0", lw=0.8)
    axes[0].barh(positions - 0.16, summary["E2F_spearman"], height=0.3, color="#1f4e79", label="E2F")
    axes[0].barh(positions + 0.16, summary["IFNg_spearman"], height=0.3, color="#b85c38", label="IFNγ")
    axes[0].set_yticks(positions, summary["sample"])
    axes[0].set_xlim(-1, 1)
    axes[0].set_xlabel("Spot Spearman, measured vs predicted")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Spot-level agreement")
    axes[1].axvline(0, color="#b0b0b0", lw=0.8)
    for position, row in enumerate(summary.itertuples()):
        axes[1].plot([row.observed_coloc, row.predicted_coloc], [position, position], color="#9a9a9a", lw=1)
    axes[1].scatter(summary["observed_coloc"], positions, c="#1f4e79", s=28, label="Measured", zorder=3)
    axes[1].scatter(summary["predicted_coloc"], positions, c="#b85c38", s=28, label="Predicted", zorder=3)
    axes[1].set_yticks(positions, summary["sample"])
    axes[1].set_xlim(-1, 1)
    axes[1].set_xlabel("Whole-section E2F–IFNγ colocalization")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Whole-section colocalization")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)
    stem = OUT / "figures" / "summary" / "spearman_and_colocalization"
    figure.savefig(stem.with_suffix(".png"), dpi=150)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)


def write_index(summary: pd.DataFrame) -> None:
    lines = [
        "<!doctype html><meta charset='utf-8'><title>T8-1 HEST breast</title>",
        "<style>body{font:16px system-ui;max-width:1100px;margin:28px auto;color:#222}img{max-width:100%}a{margin-right:14px}</style>",
        "<h1>T8-1 · HEST breast spatial check</h1>",
        "<p>Measured and predicted colors are each within-section percentiles. Absolute scores are in the tables. Visium HD matrices in this local HEST subset are already ~128 µm bins, so the 8 µm cell-scale panel is not in this atlas.</p>",
        "<p><a href='summary/spearman_and_colocalization.png'>Summary</a><a href='blocks/block_colocalization.png'>1 mm blocks</a></p>",
    ]
    lines.append("<img src='summary/spearman_and_colocalization.png'>")
    lines.append("<img src='blocks/block_colocalization.png'>")
    for sample in SAMPLES:
        rho = summary[summary["sample"].eq(sample)].iloc[0]
        lines.append(f"<h2>{html.escape(sample)}</h2><p>E2F ρ {rho.E2F_spearman:.3f}, IFNγ ρ {rho.IFNg_spearman:.3f}</p>")
        lines.append(f"<img src='maps/{html.escape(sample)}.png'>")
    (OUT / "figures" / "index.html").write_text("\n".join(lines) + "\n")


def main() -> None:
    digest = file_sha256(MODEL)
    if digest != MODEL_SHA256:
        raise RuntimeError(f"model sha changed: {digest}")
    model = joblib.load(MODEL)
    if tuple(model["coef"].shape) != (4, 1536):
        raise RuntimeError(f"unexpected coef shape {model['coef'].shape}")
    for folder in ["maps", "blocks", "summary"]:
        (OUT / "figures" / folder).mkdir(parents=True, exist_ok=True)
    scale = pd.read_csv(OUT / "scale_audit.csv")
    spot_rows: list[dict] = []
    block_rows: list[dict] = []
    summary_rows: list[dict] = []
    for sample in SAMPLES:
        measured = np.load(OUT / "measured" / f"{sample}.npz", allow_pickle=True)
        barcodes = as_str_list(measured["barcodes"])
        score_lookup = {barcode: measured["scores"][index] for index, barcode in enumerate(barcodes)}
        spatial_lookup = {barcode: measured["spatial_um"][index] for index, barcode in enumerate(barcodes)}
        with h5py.File(OUT / "features" / f"{sample}.h5", "r") as handle:
            if not bool(handle.attrs.get("complete", False)):
                raise RuntimeError(f"{sample} features are incomplete")
            patch_barcodes = as_str_list(handle["barcode"][:])
            features = np.asarray(handle["features"][:], dtype=np.float64)
        predicted = predict_st(model, features)
        records = []
        for barcode, prediction in zip(patch_barcodes, predicted):
            observed = score_lookup[barcode]
            x_um, y_um = spatial_lookup[barcode]
            records.append(
                {
                    "sample": sample,
                    "barcode": barcode,
                    "x_um": float(x_um),
                    "y_um": float(y_um),
                    "block_x": int(np.floor(x_um / BLOCK_UM)),
                    "block_y": int(np.floor(y_um / BLOCK_UM)),
                    "E2F_observed": float(observed[0]),
                    "IFNg_observed": float(observed[1]),
                    "E2F_predicted": float(prediction[0]),
                    "IFNg_predicted": float(prediction[1]),
                    "Hypoxia_predicted": float(prediction[2]),
                    "EMT_predicted": float(prediction[3]),
                    "mpp": float(measured["mpp_estimated"]),
                }
            )
        frame = pd.DataFrame(records)
        spot_rows.extend(records)
        observed_coloc = coloc_pair(frame[["E2F_observed", "IFNg_observed"]].to_numpy())
        predicted_coloc = coloc_pair(frame[["E2F_predicted", "IFNg_predicted"]].to_numpy())
        e2f_rho, n_rho = safe_spearman(frame["E2F_observed"].to_numpy(), frame["E2F_predicted"].to_numpy())
        ifn_rho, _ = safe_spearman(frame["IFNg_observed"].to_numpy(), frame["IFNg_predicted"].to_numpy())
        finite_blocks = 0
        observed_blocks = []
        predicted_blocks = []
        for (block_x, block_y), part in frame.groupby(["block_x", "block_y"], sort=True):
            observed_value = coloc_pair(part[["E2F_observed", "IFNg_observed"]].to_numpy())
            predicted_value = coloc_pair(part[["E2F_predicted", "IFNg_predicted"]].to_numpy())
            block_rows.append(
                {
                    "sample": sample,
                    "block_x": int(block_x),
                    "block_y": int(block_y),
                    "n_spots": len(part),
                    "observed_coloc": observed_value,
                    "predicted_coloc": predicted_value,
                }
            )
            if np.isfinite(observed_value) and np.isfinite(predicted_value):
                finite_blocks += 1
                observed_blocks.append(observed_value)
                predicted_blocks.append(predicted_value)
        block_rho, n_block_pairs = safe_spearman(np.asarray(observed_blocks), np.asarray(predicted_blocks))
        technology = str(scale.loc[scale["sample"].eq(sample), "st_technology"].iloc[0])
        summary_rows.append(
            {
                "sample": sample,
                "technology": technology,
                "n_spots": len(frame),
                "E2F_spearman": e2f_rho,
                "IFNg_spearman": ifn_rho,
                "observed_coloc": observed_coloc,
                "predicted_coloc": predicted_coloc,
                "n_blocks": frame.groupby(["block_x", "block_y"]).ngroups,
                "n_blocks_finite": finite_blocks,
                "block_spearman": block_rho,
                "block_pairs": n_block_pairs,
                "spearman_spots": n_rho,
            }
        )
        draw_map(sample, frame, technology, {"E2F": e2f_rho, "IFNg": ifn_rho})
        print(sample, "E2F", round(e2f_rho, 3), "IFNg", round(ifn_rho, 3), "coloc", round(observed_coloc, 3), round(predicted_coloc, 3), "blocks", round(block_rho, 3) if np.isfinite(block_rho) else None, flush=True)
    summary = pd.DataFrame(summary_rows)
    blocks = pd.DataFrame(block_rows)
    pd.DataFrame(spot_rows).to_csv(OUT / "paired_spots.csv", index=False)
    blocks.to_csv(OUT / "block_colocalization.csv", index=False)
    summary.to_csv(OUT / "sample_summary.csv", index=False)
    draw_blocks(blocks, summary)
    draw_summary(summary)
    write_index(summary)
    provenance = {
        "model": str(MODEL),
        "model_sha256": digest,
        "weights": str(WEIGHTS),
        "prediction": "(x @ coef.T + intercept) * ST_scale + ST_mean",
        "programs_predicted": ["E2F_TARGETS", "INTERFERON_GAMMA_RESPONSE", "HYPOXIA", "EPITHELIAL_MESENCHYMAL_TRANSITION"],
        "programs_compared": PROGRAMS,
        "coloc": "t5_common.coloc rank Pearson; E2F vs IFNg only",
        "block_um": BLOCK_UM,
        "fov_um": FOV_UM,
        "cell_scale_8um": "not run; local HD bins are already ~128 µm and 8 µm matrices are absent",
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (OUT / "ab_complete.json").write_text(json.dumps({"ab_complete": True, "cell_scale_complete": False, "samples": summary_rows}, indent=2))
    print("ANALYZE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
