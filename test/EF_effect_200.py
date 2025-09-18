# %% [markdown]
# Batch Quantization Evaluation — EXACTLY match your original metrics
# ------------------------------------------------------------------
# This script:
# - Scans `sample/` for `.mid` files
# - For each file, replicates your *original single-sample* logic:
#   * Quantize with EF (default) and w/o EF (sync_error_mitigation=False)
#   * Compute FAE as |sum(delta_orig) - sum(delta_quant)| (absolute final drift)
#   * Compute MAE as mean(|abs_times_orig - abs_times_quant|) (onset-based MAE)
# - Aggregates per-file results and prints mean/std/95% CI across files
# - Saves `eval_200_detailed.csv` (per-file) and `eval_200_summary.csv` (summary)
#
# Notes
# - This code does NOT change your quantization API or metric definitions.
# - All comments/strings are in English as requested.

# %%
from __future__ import annotations
import copy
import csv
import math
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ------------------------------
# Config (keep identical behavior)
# ------------------------------
SAMPLE_DIR = Path("sample")
UNIT = "32"  # same as your single-sample code
CSV_DETAILED = Path("eval_200_detailed.csv")
CSV_SUMMARY = Path("eval_200_summary.csv")
PRINT_TOPK = 5

# ------------------------------
# I/O — use your loader exactly
# ------------------------------


def load_midi(path: Path):
    import midii  # same as your code/environment

    return midii.MidiFile(
        str(path), convert_1_to_0=True, lyric_encoding="cp949"
    )


# ------------------------------
# Helpers — identical metric semantics
# ------------------------------


def calculate_absolute_times(delta_times) -> np.ndarray:
    """Cumulative sum of delta ticks (int64)."""
    return np.cumsum(np.array(delta_times, dtype=np.int64))


def mae_on_onsets(abs0: np.ndarray, absq: np.ndarray) -> float:
    """Onset-based MAE: mean absolute difference between absolute-time sequences."""
    diff = (abs0 - absq).astype(np.float64)
    return float(np.mean(np.abs(diff)))


def fae_absolute(delta0: np.ndarray, deltaq: np.ndarray) -> int:
    """FAE as absolute final accumulated error (|sum(original) - sum(quantized)|)."""
    return int(
        abs(
            int(np.sum(delta0, dtype=np.int64))
            - int(np.sum(deltaq, dtype=np.int64))
        )
    )


# ------------------------------
# Per-file evaluation — mirrors your single-sample code
# ------------------------------


def evaluate_one(path: Path) -> Dict[str, float]:
    mid = load_midi(path)

    # Original (no quantization)
    d0 = np.asarray(mid.times, dtype=np.int64)
    abs0 = calculate_absolute_times(d0)

    # w/o EF (explicit False)
    mid_naive = copy.deepcopy(mid)
    mid_naive.quantize(unit=UNIT, sync_error_mitigation=False)
    dq_naive = np.asarray(mid_naive.times, dtype=np.int64)
    abs_naive = calculate_absolute_times(dq_naive)

    # with EF (default behavior of your code)
    mid_ef = copy.deepcopy(mid)
    mid_ef.quantize(unit=UNIT)  # EF enabled by default in your code
    dq_ef = np.asarray(mid_ef.times, dtype=np.int64)
    abs_ef = calculate_absolute_times(dq_ef)

    # Metrics (exact semantics)
    mae_naive = mae_on_onsets(abs0, abs_naive)
    mae_ef = mae_on_onsets(abs0, abs_ef)
    fae_naive = fae_absolute(d0, dq_naive)
    fae_ef = fae_absolute(d0, dq_ef)

    return {
        "file": path.name,
        "N": int(d0.size),
        "MAE_naive": mae_naive,
        "MAE_EF": mae_ef,
        "ΔMAE": mae_naive - mae_ef,
        "FAE_naive": fae_naive,
        "FAE_EF": fae_ef,
        "ΔFAE": fae_naive - fae_ef,
    }


# ------------------------------
# Stats
# ------------------------------


def mean_std_ci(
    x: List[float], alpha: float = 0.05
) -> Tuple[float, float, float, float]:
    a = np.asarray(x, dtype=np.float64)
    n = int(a.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mu = float(np.mean(a))
    sd = float(np.std(a, ddof=1)) if n > 1 else 0.0
    z = 1.959963984540054  # 95% normal quantile
    half = z * (sd / math.sqrt(n)) if n > 1 else 0.0
    return mu, sd, mu - half, mu + half


# ------------------------------
# Main
# ------------------------------


def main() -> None:
    files = sorted(SAMPLE_DIR.glob("**/*.mid"))
    if not files:
        print(f"No .mid files under {SAMPLE_DIR.resolve()}")
        return

    rows: List[Dict[str, float]] = []
    failures: List[Tuple[Path, str]] = []

    for p in files:
        try:
            rows.append(evaluate_one(p))
        except Exception as e:
            failures.append((p, f"{type(e).__name__}: {e}"))
            traceback.print_exc()

    print(f"Processed: {len(rows)} files; Failed: {len(failures)}")
    if failures:
        print("-- Failures (first 3) --")
        for pf, msg in failures[:3]:
            print(f"{pf.name}: {msg}")

    # Quick sanity: top by ΔMAE (descending)
    rows_sorted = sorted(rows, key=lambda r: r["ΔMAE"], reverse=True)
    print("\nTop examples by ΔMAE (onset-based):")
    for r in rows_sorted[:PRINT_TOPK]:
        print(
            f"{r['file']}: MAE_n={r['MAE_naive']:.3f}, MAE_EF={r['MAE_EF']:.3f}, Δ={r['ΔMAE']:.3f}"
        )

    # Summary stats (across files)
    def col(name: str) -> List[float]:
        return [float(r[name]) for r in rows]

    for name in ["MAE_naive", "MAE_EF", "ΔMAE", "FAE_naive", "FAE_EF", "ΔFAE"]:
        mu, sd, lo, hi = mean_std_ci(col(name))
        print(
            f"{name:10s}: mean={mu:.3f}, std={sd:.3f}, 95% CI=[{lo:.3f}, {hi:.3f}]"
        )

    # Save CSVs
    fieldnames = (
        list(rows_sorted[0].keys())
        if rows_sorted
        else [
            "file",
            "N",
            "MAE_naive",
            "MAE_EF",
            "ΔMAE",
            "FAE_naive",
            "FAE_EF",
            "ΔFAE",
        ]
    )
    with open(CSV_DETAILED, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_sorted)

    # Small summary table
    summary = []
    for name in ["MAE_naive", "MAE_EF", "ΔMAE", "FAE_naive", "FAE_EF", "ΔFAE"]:
        mu, sd, lo, hi = mean_std_ci(col(name))
        summary.append(
            {"metric": name, "mean": mu, "std": sd, "ci_lo": lo, "ci_hi": hi}
        )
    with open(CSV_SUMMARY, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["metric", "mean", "std", "ci_lo", "ci_hi"]
        )
        w.writeheader()
        w.writerows(summary)

    print(f"\nSaved detailed results to: {CSV_DETAILED.resolve()}")
    print(f"Saved summary results  to: {CSV_SUMMARY.resolve()}")


if __name__ == "__main__":
    main()
