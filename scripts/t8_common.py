"""T8-1 fixed definitions. Colocalization matches scripts/t5_common.py coloc."""
from __future__ import annotations

import os

os.environ.update(
    OMP_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    NUMEXPR_NUM_THREADS="2",
    MPLBACKEND="Agg",
    HF_HUB_OFFLINE="1",
)

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
HEST = Path("/Volumes/INTENSO_BIG/HEST1k")
OUT = ROOT / "results" / "t8"
WORK = ROOT / "docs" / "worklog" / "2026-10-27_T8"
MODEL = Path(
    "/Volumes/INTENSO/SpatialMosaic_data/02_derived/project_large_artifacts/t6/models/TCGA-BRCA_fold0.joblib"
)
MODEL_SHA256 = "efed6e3080b2fdbd8de25b308af7209cfde84f95831f31a8419906e3cc4d326a"
WEIGHTS = Path(
    "/Users/Zhaiyue/.cache/huggingface/hub/models--MahmoodLab--UNI2-h/snapshots/d517a8dd47902dd7c308b3c36f63bce47e7b9a43/pytorch_model.bin"
)
BACKGROUND = ROOT / "results/t3b/protein_coding_background.txt"
MEMBERS = ROOT / "results/t3b/program_members.json"
SYMBOL_MAP = ROOT / "references/2026-10-23_STPath_PathPrism/STPath/code/utils_data/symbol2ensembl.json"
SAMPLES = ["TENX39", "TENX53", "TENX68", "TENX161", "TENX162", "TENX180"]
PROGRAMS = ["E2F_TARGETS", "INTERFERON_GAMMA_RESPONSE"]
SHORT = ["E2F", "IFNg"]
MAX_RANK = 1500
BLOCK_UM = 1000.0
FOV_UM = 112.0
BATCH = 4
PAUSE_SECONDS = 1.0


def as_str_list(values: np.ndarray) -> list[str]:
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    out: list[str] = []
    for value in array:
        if isinstance(value, bytes):
            out.append(value.decode())
        else:
            out.append(str(value))
    return out


def ucell_matrix(counts: np.ndarray, member_indexes: list[np.ndarray], max_rank: int = MAX_RANK) -> np.ndarray:
    """Same descending average-rank UCell as scripts/t4_targets_discovery.py."""
    n_background = counts.shape[1]
    scores = np.empty((counts.shape[0], len(member_indexes)), dtype=np.float64)
    for start in range(0, counts.shape[0], 64):
        block = counts[start : start + 64]
        rank = n_background + 1 - rankdata(block, axis=1, method="average")
        capped = np.minimum(rank, max_rank)
        for column, index in enumerate(member_indexes):
            n_members = len(index)
            numerator = capped[:, index].sum(1) - n_members * (n_members + 1) / 2
            denominator = max_rank * n_members - n_members * (n_members + 1) / 2
            scores[start : start + len(block), column] = 1 - numerator / denominator
    return scores


def coloc_pair(values: np.ndarray) -> float:
    """Pearson correlation of average ranks for two columns.

    This is element [0, 1] of scripts/t5_common.py `coloc`. Fewer than 3 spots,
    or a constant column, is NA, the same undefined rule as config/t4_frozen.json.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] < 3:
        return float("nan")
    ranks = rankdata(values, axis=0, method="average")
    centered = ranks - ranks.mean(0)
    denominator = np.sqrt((centered * centered).sum(0))
    if denominator[0] * denominator[1] <= 0:
        return float("nan")
    return float((centered[:, 0] @ centered[:, 1]) / (denominator[0] * denominator[1]))


def assert_definitions() -> None:
    top = np.zeros((1, 1600))
    top[0, :20] = np.arange(20, 0, -1)
    score = ucell_matrix(top, [np.arange(20)])[0, 0]
    if abs(score - 1) >= 1e-12:
        raise AssertionError(f"UCell top program {score}")
    zeros = ucell_matrix(np.zeros((1, 4000)), [np.arange(20)])[0, 0]
    if abs(zeros) >= 1e-12:
        raise AssertionError(f"UCell all-zero {zeros}")
    rng = np.random.default_rng(0)
    wide = rng.normal(size=(40, 4))
    ranks = rankdata(wide, axis=0, method="average")
    centered = ranks - ranks.mean(0)
    denominator = np.sqrt((centered * centered).sum(0))
    matrix = (centered.T @ centered) / (denominator[:, None] * denominator[None, :])
    got = coloc_pair(wide[:, :2])
    if abs(got - matrix[0, 1]) >= 1e-12:
        raise AssertionError("coloc_pair diverges from the four-column T5 formula")
    if coloc_pair(np.ones((5, 2))) == coloc_pair(np.ones((5, 2))):
        if np.isfinite(coloc_pair(np.ones((5, 2)))):
            raise AssertionError("constant coloc should be NA")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())
