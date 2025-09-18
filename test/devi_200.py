#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp. 5.3 quantitative evaluation over 200 samples.

This script generalizes the user's single-sample MAE computation to a batch
of up to 200 samples found under the local `sample/` folder.

For each MIDI file:
  1) Load with midii.MidiFile(..., convert_1_to_0=True, lyric_encoding="cp949").
  2) Create quantized copies for units in {4,8,16,32,64,128,256}.
  3) Compute MAE between cumulative onset sequences (ticks) of the original vs. each quantized file.

Finally, aggregate the MAE values across files and report mean, std, and a 95% CI
(using normal approximation; with N≈200 the difference to Student-t is negligible).

Outputs:
  - Pretty-printed stats per quantization unit.
  - CSV file `exp53_mae_stats_200.csv` for easy import into LaTeX/analysis tools.
  - Optional LaTeX table string printed to stdout; copy-paste into the paper if desired.

Note:
  - All comments are in English as requested.
  - The script is defensive to potential length mismatches (takes the common prefix length).
"""

import sys
import math
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import midii
import copy

# ------------------------------
# Configurable parameters
# ------------------------------
SAMPLE_DIR = Path("sample")  # Folder that contains ~200 samples
MAX_SAMPLES = 200  # Use up to 200 samples (deterministic order)
UNITS = [
    "4",
    "8",
    "16",
    "32",
    "64",
    "128",
    "256",
]  # Quantization units to evaluate
CSV_PATH = Path("exp53_mae_stats_200.csv")
PRINT_LATEX = True  # If True, print a LaTeX table at the end

# ------------------------------
# Utilities
# ------------------------------


def iter_sample_paths(sample_dir: Path, max_samples: int) -> List[Path]:
    """Collect up to `max_samples` MIDI-like files from `sample_dir`.

    The function prefers common MIDI extensions. Adjust as needed if your
    dataset contains other file formats that midii can read.
    """
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    # Collect paths deterministically (sorted) for reproducibility.
    exts = {".mid", ".midi", ".MID", ".MIDI"}
    all_paths = sorted(
        [p for p in sample_dir.rglob("*") if p.is_file() and p.suffix in exts]
    )

    if not all_paths:
        raise RuntimeError(
            f"No MIDI files (.mid/.midi) were found under: {sample_dir.resolve()}"
        )

    return all_paths[:max_samples]


def cumulative_times(midi_obj: "midii.MidiFile") -> np.ndarray:
    """Return cumulative onsets (absolute ticks) as a 1D numpy array."""
    # midii.MidiFile exposes `times` as delta-times (ticks).
    # We convert to absolute onsets via cumulative sum.
    return np.cumsum(np.asarray(midi_obj.times, dtype=np.float64))


def mean_absolute_error(a: np.ndarray, b: np.ndarray) -> float:
    """Compute MAE between two 1D arrays using the common prefix length.

    This guards against potential length differences introduced by
    pre/post-processing or quantization edge-cases.
    """
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return float("nan")
    return float(np.mean(np.abs(a[:n] - b[:n])))


def per_file_mae_by_unit(
    midi_path: Path, units: List[str]
) -> Dict[str, float]:
    """Compute MAE for each quantization unit for a single file."""
    try:
        base = midii.MidiFile(
            str(midi_path), convert_1_to_0=True, lyric_encoding="cp949"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load MIDI: {midi_path} -> {e}") from e

    base_abs = cumulative_times(base)

    maes: Dict[str, float] = {}

    for u in units:
        # Deep-copy to avoid in-place modifications on the original object.
        q = copy.deepcopy(base)
        try:
            q.quantize(unit=u)
        except Exception as e:
            # If quantization fails for a specific unit, record NaN and continue.
            print(
                f"[WARN] quantize(unit={u}) failed for {midi_path.name}: {e}",
                file=sys.stderr,
            )
            maes[u] = float("nan")
            continue

        q_abs = cumulative_times(q)
        maes[u] = mean_absolute_error(base_abs, q_abs)

    return maes


def mean_std_ci(
    values: List[float], ci: float = 0.95
) -> Tuple[float, float, float, float, int]:
    """Return (mean, std, ci_low, ci_high, n) for finite values.

    CI uses normal approximation (z ≈ 1.96 for 95%). With n≈200 this is close
    to Student's t (df=n-1). Non-finite values (NaN/inf) are filtered out.
    """
    arr = np.array([v for v in values if np.isfinite(v)], dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"), 0)

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))  # sample std (ddof=1)

    # z for 95% two-sided normal approx
    z = (
        1.96 if abs(ci - 0.95) < 1e-9 else 1.96
    )  # extend if other CI levels are needed
    half = float(z * (std / math.sqrt(n)))
    return (mean, std, mean - half, mean + half, n)


# ------------------------------
# Main evaluation
# ------------------------------


def main() -> None:
    paths = iter_sample_paths(SAMPLE_DIR, MAX_SAMPLES)

    # Accumulator: unit -> list of MAEs over files
    acc: Dict[str, List[float]] = {u: [] for u in UNITS}

    # Process files
    for idx, p in enumerate(paths, 1):
        try:
            maes = per_file_mae_by_unit(p, UNITS)
        except Exception as e:
            print(f"[ERROR] Skipping {p.name}: {e}", file=sys.stderr)
            continue

        for u, v in maes.items():
            acc[u].append(v)

        # Optional heartbeat
        # print(f"Processed {idx:3d}/{len(paths)}: {p.name}")

    # Compute stats and print
    print(
        "\n=== Exp. 5.3 — MAE stats over up to 200 samples (unit: ticks) ==="
    )
    header = ["unit", "mean", "std", "ci_low", "ci_high", "n"]
    rows = []

    for u in UNITS:
        mean, std, lo, hi, n = mean_std_ci(acc[u], ci=0.95)
        print(
            f"unit=1/{u:<4s}  mean={mean:10.3f}  std={std:10.3f}  95%CI=[{lo:10.3f}, {hi:10.3f}]  n={n}"
        )
        rows.append(
            [
                f"1/{u}",
                f"{mean:.3f}",
                f"{std:.3f}",
                f"{lo:.3f}",
                f"{hi:.3f}",
                str(n),
            ]
        )

    # Save CSV for downstream LaTeX/table use
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"\nSaved: {CSV_PATH.resolve()}")


if __name__ == "__main__":
    main()
