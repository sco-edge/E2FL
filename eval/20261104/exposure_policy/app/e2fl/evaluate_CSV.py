#!/usr/bin/env python3
"""Compare two FL runs from CSV logs.

Metrics computed per CSV:

1) Total training time (s)
   - Consider rows where phase == 'train'
   - For each row, row_time = comp_time + comm_time
   - For each round, take the MAX row_time across devices (2 devices -> 2 rows)
   - Sum across rounds

2) Total energy per device
   - Consider rows where phase == 'train'
   - The CSV's `energy` column is assumed to be the *per-round average power-integrated energy* for that device.
   - Total energy per device = sum(energy) across train rounds.

Usage:
  python evaluate_CSV.py runA.csv runB.csv

Optional:
  python evaluate_CSV.py runA.csv runB.csv --phase train --round-col round --device-col client_id

Notes:
- Column names can vary; the script provides reasonable defaults and fallbacks.

If device IDs differ across runs, you can pass --infer-slow-fast to align devices as slow/fast by average comp_time.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -------------------------
# Utilities
# -------------------------

def _find_first_existing(columns: List[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _coerce_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    if s.isna().all():
        raise ValueError(f"Column '{col}' is not numeric or could not be parsed.")
    return s


@dataclass
class RunStats:
    path: str
    rounds: int
    total_train_time_s: float
    per_device_energy: Dict[str, float]
    per_device_comp_mean: Dict[str, float]
    total_energy: float


def compute_stats(
    csv_path: str,
    phase_value: str = "train",
    phase_col: Optional[str] = None,
    round_col: Optional[str] = None,
    device_col: Optional[str] = None,
    comp_col: Optional[str] = None,
    comm_col: Optional[str] = None,
    energy_col: Optional[str] = None,
) -> RunStats:
    df = pd.read_csv(csv_path)

    # Auto-detect columns if not specified
    cols = df.columns.tolist()

    if phase_col is None:
        phase_col = _find_first_existing(cols, ["phase", "mode", "stage"])
    if round_col is None:
        round_col = _find_first_existing(cols, ["round", "server_round", "rnd"])
    if device_col is None:
        device_col = _find_first_existing(cols, [
            "client_id", "cid", "node_id", "device_id", "client", "node", "partition_id"
        ])
    if comp_col is None:
        comp_col = _find_first_existing(cols, ["comp_time", "compute_time", "train_time", "compute_s", "comp_s"])
    if comm_col is None:
        comm_col = _find_first_existing(cols, ["comm_time", "communication_time", "comm_s", "upload_time", "download_time"])
    if energy_col is None:
        energy_col = _find_first_existing(cols, ["energy", "energy_j", "train_energy", "energy_joule", "energy_joules"])

    missing = [name for name, val in [
        ("phase_col", phase_col),
        ("round_col", round_col),
        ("device_col", device_col),
        ("comp_col", comp_col),
        ("comm_col", comm_col),
        ("energy_col", energy_col),
    ] if val is None]
    if missing:
        raise SystemExit(
            f"[evaluate_CSV] ERROR: Could not auto-detect required columns: {missing}.\n"
            f"Columns in file: {cols}\n"
            f"Please pass explicit args: --phase-col/--round-col/--device-col/--comp-col/--comm-col/--energy-col"
        )

    # Filter to requested phase
    phase_mask = df[phase_col].astype(str).str.lower() == phase_value.lower()
    train = df[phase_mask].copy()
    if train.empty:
        raise SystemExit(f"[evaluate_CSV] ERROR: No rows found for phase='{phase_value}' in {csv_path}")

    # Row time
    train["__comp"] = _coerce_numeric(train, comp_col)
    train["__comm"] = _coerce_numeric(train, comm_col)
    train["__row_time"] = train["__comp"] + train["__comm"]

    # Round -> max across devices
    train["__round"] = train[round_col]
    per_round_max = train.groupby("__round")["__row_time"].max()
    total_train_time_s = float(per_round_max.sum())

    # Energy per device (sum over rounds)
    train["__energy"] = _coerce_numeric(train, energy_col)
    train["__device"] = train[device_col].astype(str)
    per_device_energy = (
        train.groupby("__device")["__energy"].sum().sort_index().to_dict()
    )
    total_energy = float(sum(per_device_energy.values()))

    # Compute per-device mean comp_time for alignment heuristics
    per_device_comp_mean = (
        train.groupby("__device")["__comp"].mean().sort_index().to_dict()
    )

    return RunStats(
        path=csv_path,
        rounds=int(per_round_max.shape[0]),
        total_train_time_s=total_train_time_s,
        per_device_energy=per_device_energy,
        per_device_comp_mean=per_device_comp_mean,
        total_energy=total_energy,
    )


def pct_delta(new: float, old: float) -> float:
    if old == 0:
        return math.inf if new != 0 else 0.0
    return 100.0 * (new - old) / old


def format_seconds(s: float) -> str:
    # Show both seconds and h:m:s
    sec = int(round(s))
    h = sec // 3600
    m = (sec % 3600) // 60
    ss = sec % 60
    return f"{s:.3f}s ({h:d}:{m:02d}:{ss:02d})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_a", type=str, help="First CSV (baseline)")
    ap.add_argument("csv_b", type=str, help="Second CSV (comparison)")

    ap.add_argument("--phase", type=str, default="train", help="Phase value to filter (default: train)")
    ap.add_argument("--phase-col", type=str, default=None)
    ap.add_argument("--round-col", type=str, default=None)
    ap.add_argument("--device-col", type=str, default=None)
    ap.add_argument("--comp-col", type=str, default=None)
    ap.add_argument("--comm-col", type=str, default=None)
    ap.add_argument("--energy-col", type=str, default=None)
    ap.add_argument(
        "--infer-slow-fast",
        action="store_true",
        help="If device ids differ across runs, align devices by average comp_time: label the higher-comp_time device as 'slow' and the other as 'fast'.",
    )

    args = ap.parse_args()

    stats_a = compute_stats(
        args.csv_a,
        phase_value=args.phase,
        phase_col=args.phase_col,
        round_col=args.round_col,
        device_col=args.device_col,
        comp_col=args.comp_col,
        comm_col=args.comm_col,
        energy_col=args.energy_col,
    )
    stats_b = compute_stats(
        args.csv_b,
        phase_value=args.phase,
        phase_col=args.phase_col,
        round_col=args.round_col,
        device_col=args.device_col,
        comp_col=args.comp_col,
        comm_col=args.comm_col,
        energy_col=args.energy_col,
    )

    # Print summary
    print("\n=== Summary (phase=%s) ===" % args.phase)

    def _print_run(label: str, st: RunStats) -> None:
        print(f"\n[{label}] {st.path}")
        print(f"  rounds                  : {st.rounds}")
        print(f"  total_train_time        : {format_seconds(st.total_train_time_s)}")
        print(f"  total_energy (sum devs) : {st.total_energy:.6f}")
        print(f"  (note) total_energy is sum over devices across train rounds")
        print("  per_device_energy       :")
        for dev, e in st.per_device_energy.items():
            print(f"    - {dev}: {e:.6f}")

    _print_run("A", stats_a)
    _print_run("B", stats_b)

    print("\n=== Delta (B vs A) ===")
    dt = pct_delta(stats_b.total_train_time_s, stats_a.total_train_time_s)
    de = pct_delta(stats_b.total_energy, stats_a.total_energy)

    print(f"  train_time: {stats_a.total_train_time_s:.3f}s -> {stats_b.total_train_time_s:.3f}s  ({dt:+.2f}%)")
    print(f"  energy    : {stats_a.total_energy:.6f} -> {stats_b.total_energy:.6f}  ({de:+.2f}%)")

    # Per-device deltas are only meaningful if device identifiers match across runs.
    devs_a = set(stats_a.per_device_energy.keys())
    devs_b = set(stats_b.per_device_energy.keys())
    common = sorted(devs_a & devs_b)

    if common:
        print("\n  per-device energy delta (matched device ids):")
        for dev in common:
            ea = stats_a.per_device_energy.get(dev, 0.0)
            eb = stats_b.per_device_energy.get(dev, 0.0)
            d = pct_delta(eb, ea)
            print(f"    - {dev}: {ea:.6f} -> {eb:.6f}  ({d:+.2f}%)")

        only_a = sorted(devs_a - devs_b)
        only_b = sorted(devs_b - devs_a)
        if only_a or only_b:
            print("\n  [warn] Some device ids differ between runs; unmatched devices are omitted from deltas.")
            if only_a:
                print("    only in A:")
                for dev in only_a:
                    print(f"      - {dev}: {stats_a.per_device_energy[dev]:.6f}")
            if only_b:
                print("    only in B:")
                for dev in only_b:
                    print(f"      - {dev}: {stats_b.per_device_energy[dev]:.6f}")

    elif args.infer_slow_fast:
        # Heuristic alignment: classify devices by mean comp_time.
        # Higher mean comp_time => slower device (likely RPi5), lower => faster (likely Jetson).
        def label_by_speed(per_dev_comp: Dict[str, float]) -> Dict[str, str]:
            items = sorted(per_dev_comp.items(), key=lambda kv: kv[1], reverse=True)
            labels: Dict[str, str] = {}
            if not items:
                return labels
            labels[items[0][0]] = "slow"
            for dev, _ in items[1:]:
                labels[dev] = "fast"
            return labels

        labels_a = label_by_speed(stats_a.per_device_comp_mean)
        labels_b = label_by_speed(stats_b.per_device_comp_mean)

        # Aggregate energy by label
        def energy_by_label(per_dev_energy: Dict[str, float], labels: Dict[str, str]) -> Dict[str, float]:
            out: Dict[str, float] = {"slow": 0.0, "fast": 0.0}
            for dev, e in per_dev_energy.items():
                lab = labels.get(dev)
                if lab in out:
                    out[lab] += float(e)
            return out

        ea = energy_by_label(stats_a.per_device_energy, labels_a)
        eb = energy_by_label(stats_b.per_device_energy, labels_b)

        print("\n  per-device energy delta (inferred slow/fast by mean comp_time):")
        print("    (slow: larger mean comp_time; fast: smaller mean comp_time)")
        for lab in ["slow", "fast"]:
            d = pct_delta(eb[lab], ea[lab])
            print(f"    - {lab}: {ea[lab]:.6f} -> {eb[lab]:.6f}  ({d:+.2f}%)")

        print("\n  inferred mapping details:")
        print("    A labels (device_id -> label, mean_comp_time):")
        for dev, lab in sorted(labels_a.items(), key=lambda kv: kv[1]):
            print(f"      - {dev} -> {lab}, mean_comp={stats_a.per_device_comp_mean.get(dev, float('nan')):.6f}")
        print("    B labels (device_id -> label, mean_comp_time):")
        for dev, lab in sorted(labels_b.items(), key=lambda kv: kv[1]):
            print(f"      - {dev} -> {lab}, mean_comp={stats_b.per_device_comp_mean.get(dev, float('nan')):.6f}")

        print("\n  [note] This alignment is a heuristic. If comp_time distributions overlap (e.g., two similar devices), label inference may be unreliable.")

    else:
        print("\n  [warn] Device ids do not match across runs, so per-device deltas are not meaningful.")
        print("        Use --infer-slow-fast to align by mean comp_time (heuristic), or log a stable device label.")


if __name__ == "__main__":
    main()