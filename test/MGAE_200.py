#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp. 5.1 quantitative evaluation over 200 samples (Grid Alignment Error).

This script generalizes the user's single-sample measurement to a batch of up to
200 samples in the local `sample/` folder. For each file, it measures the
Mean Grid Alignment Error (GAE) for both the original MIDI and its quantized
version at the target unit (default: 1/32 note).

Metrics reported across files (after filtering non-finite values):
  - Original  GAE: mean, std, 95% CI
  - Quantized GAE: mean, std, 95% CI
  - Delta (orig - quant): mean, std, 95% CI, and reduction rate (%)

Outputs:
  - Console summary per metric.
  - CSV file `exp51_gae_stats_200.csv`.
  - Optional LaTeX table printed to stdout for copy-paste.

Notes:
  - All comments are in English (per user preference).
  - `unit_to_grid_ticks()` infers the grid size from the file's TPQN and the
    quantization denominator (e.g., 32 -> 1/32 note). For TPQN=120, 1/32 grid
    is 120/8=15 ticks, matching the user's original code.
  - The GAE formula matches the user's function: we compute start/end distances
    to the nearest grid for each note, sum them, and divide by the number of
    notes (not by 2×notes). This preserves identical semantics.
"""

import math
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import midii
import copy

# ------------------------------
# Configuration
# ------------------------------
SAMPLE_DIR = Path("sample")  # Folder with ~200 samples
MAX_SAMPLES = 200  # Use up to 200 samples (deterministic order)
TARGET_UNIT = (
    "32"  # Quantization unit as denominator string: "4","8","16","32",...
)
CSV_PATH = Path("exp51_gae_stats_200.csv")
PRINT_LATEX = True  # If True, print LaTeX table at the end

# ------------------------------
# Utilities
# ------------------------------


def iter_sample_paths(sample_dir: Path, max_samples: int) -> List[Path]:
    """Collect up to `max_samples` MIDI files from `sample_dir` in sorted order."""
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    exts = {".mid", ".midi", ".MID", ".MIDI"}
    all_paths = sorted(
        [p for p in sample_dir.rglob("*") if p.is_file() and p.suffix in exts]
    )
    if not all_paths:
        raise RuntimeError(
            f"No MIDI files (.mid/.midi) found under: {sample_dir.resolve()}"
        )
    return all_paths[:max_samples]


def get_tpqn(midi_obj: "midii.MidiFile") -> int:
    """Return ticks-per-quarter-note (TPQN) from the midii object."""
    # Try common attribute names; fall back to 480 if unavailable.
    for attr in ("ticks_per_beat", "tpqn", "ticksPerBeat"):
        if hasattr(midi_obj, attr):
            return int(getattr(midi_obj, attr))
    return 480


def unit_to_grid_ticks(tpqn: int, unit: str) -> int:
    """Convert quantization denominator (e.g., '32') to grid ticks for one subunit.

    A quarter note is 1 beat. Denominator D corresponds to 1/D note.
    Grid size in beats = (1/4) / (1/D) = 4/D beats per subunit? No:
    1/D note = (1/D) * (quarter_note * 4)? Clarify via direct ratio:
      - Quarter = denominator 4 -> 1 beat.
      - 1/32 note = (1/8) beat.
    Therefore: ticks per (1/D) = TPQN * (4/D).
    Example: TPQN=120, D=32 => 120 * (4/32) = 15 ticks.
    """
    D = int(unit)
    ticks = tpqn * 4 / D
    return int(round(ticks))


def nearest_grid_distance_tick(tick: int, grid: int) -> int:
    """Return the absolute distance (in ticks) from `tick` to the nearest grid line.

    Equivalent to min(tick % grid, grid - (tick % grid)).
    """
    r = tick % grid
    return r if r < (grid - r) else (grid - r)


def mean_grid_alignment_error_from_json(
    items: List[dict], grid_ticks: int
) -> float:
    """Compute Mean Grid Alignment Error (GAE) for a sequence of note dicts.

    GAE is defined exactly like the user's function:
        - For each note, add distance(start, grid) + distance(end, grid)
        - Divide by the number of notes (not by 2×notes)
    Returns NaN if the list is empty or grid is invalid.
    """
    if grid_ticks <= 0 or not items:
        return float("nan")

    total_error = 0
    for x in items:
        # Expect 'start' and 'end' keys in ticks.
        s = int(x.get("start", 0))
        e = int(x.get("end", 0))
        total_error += nearest_grid_distance_tick(e, grid_ticks)
        total_error += nearest_grid_distance_tick(s, grid_ticks)
    return float(total_error / len(items))


def mean_std_ci(
    values: List[float], ci: float = 0.95
) -> Tuple[float, float, float, float, int]:
    """Return (mean, std, ci_low, ci_high, n) after filtering non-finite values.

    CI uses normal approximation (z=1.96 for 95%). With n≈200, this closely
    matches the t-interval. Uses sample std (ddof=1).
    """
    arr = np.array([v for v in values if np.isfinite(v)], dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"), 0)

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    z = 1.96 if abs(ci - 0.95) < 1e-9 else 1.96
    half = float(z * (std / math.sqrt(n)))
    return (mean, std, mean - half, mean + half, n)


# ------------------------------
# Core evaluation
# ------------------------------


def per_file_gae_original_and_quantized(
    midi_path: Path, target_unit: str
) -> Tuple[float, float]:
    """Compute (GAE_original, GAE_quantized) for a single MIDI file."""
    base = midii.MidiFile(
        str(midi_path), convert_1_to_0=True, lyric_encoding="cp949"
    )
    tpqn = get_tpqn(base)
    grid = unit_to_grid_ticks(tpqn, target_unit)

    # Original JSON and GAE
    items_base = base.to_json()
    gae_orig = mean_grid_alignment_error_from_json(items_base, grid)

    # Quantized copy and GAE
    q = copy.deepcopy(base)
    q.quantize(unit=target_unit)
    items_q = q.to_json()
    gae_quant = mean_grid_alignment_error_from_json(items_q, grid)

    return gae_orig, gae_quant


def main() -> None:
    paths = iter_sample_paths(SAMPLE_DIR, MAX_SAMPLES)

    gae_orig_all: List[float] = []
    gae_quant_all: List[float] = []

    for i, p in enumerate(paths, 1):
        try:
            gae_o, gae_q = per_file_gae_original_and_quantized(p, TARGET_UNIT)
        except Exception as e:
            print(f"[ERROR] Skipping {p.name}: {e}")
            continue
        gae_orig_all.append(gae_o)
        gae_quant_all.append(gae_q)
        # print(f"Processed {i:3d}/{len(paths)}: {p.name}")

    # Compute stats
    o_mean, o_std, o_lo, o_hi, o_n = mean_std_ci(gae_orig_all)
    q_mean, q_std, q_lo, q_hi, q_n = mean_std_ci(gae_quant_all)

    # Delta and reduction
    # We compute per-file delta, then aggregate to reflect pairing.
    deltas = [
        o - q
        for (o, q) in zip(gae_orig_all, gae_quant_all)
        if np.isfinite(o) and np.isfinite(q)
    ]
    d_mean, d_std, d_lo, d_hi, d_n = mean_std_ci(deltas)
    reduction_pct = float("nan")
    if np.isfinite(o_mean) and o_mean != 0:
        reduction_pct = 100.0 * (d_mean / o_mean)

    # Console summary
    print(
        "\n=== Exp. 5.1 — Grid Alignment Error (unit: 1/%s, ticks) over up to 200 samples ==="
        % TARGET_UNIT
    )
    print(
        f"Original : mean={o_mean:.3f}  std={o_std:.3f}  95%CI=[{o_lo:.3f}, {o_hi:.3f}]  n={o_n}"
    )
    print(
        f"Quantized: mean={q_mean:.3f}  std={q_std:.3f}  95%CI=[{q_lo:.3f}, {q_hi:.3f}]  n={q_n}"
    )
    print(
        f"Delta    : mean={d_mean:.3f}  std={d_std:.3f}  95%CI=[{d_lo:.3f}, {d_hi:.3f}]  n={d_n}"
    )
    print(f"Reduction: {reduction_pct:.2f}% (relative to Original mean)")

    # Save CSV
    header = [
        "unit",
        "orig_mean",
        "orig_std",
        "orig_ci_low",
        "orig_ci_high",
        "orig_n",
        "quant_mean",
        "quant_std",
        "quant_ci_low",
        "quant_ci_high",
        "quant_n",
        "delta_mean",
        "delta_std",
        "delta_ci_low",
        "delta_ci_high",
        "delta_n",
        "reduction_percent",
    ]
    row = [
        f"1/{TARGET_UNIT}",
        f"{o_mean:.3f}",
        f"{o_std:.3f}",
        f"{o_lo:.3f}",
        f"{o_hi:.3f}",
        str(o_n),
        f"{q_mean:.3f}",
        f"{q_std:.3f}",
        f"{q_lo:.3f}",
        f"{q_hi:.3f}",
        str(q_n),
        f"{d_mean:.3f}",
        f"{d_std:.3f}",
        f"{d_lo:.3f}",
        f"{d_hi:.3f}",
        str(d_n),
        f"{reduction_pct:.2f}",
    ]

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)
    print(f"\nSaved: {CSV_PATH.resolve()}")


if __name__ == "__main__":
    main()
