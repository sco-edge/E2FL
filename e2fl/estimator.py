

"""E2FL Cost Estimator Utilities

This module contains two parts:

1) Offline profiling / coefficient fitting:
   Fit device-dependent coefficients (beta_sum, kappa, optional power baseline)
   from previously collected CSV logs.

2) (Optional) lightweight online helpers (EMA) that can be used by a Flower
   strategy at runtime to track per-round scaling (alpha) and power drift.

The fitting targets the user's compute-time model:

  T_obs ≈ tau * (beta_sum + kappa * (rank * n_modules))

Where:
  tau := local_steps * batch_size * seq_len

All time units inside the fitted coefficients are milliseconds per-token
(because that is usually convenient when comp_time is in seconds).

Usage example:
  python -m e2fl.estimator offline-fit \
    --csv /path/to/profile_runs.csv \
    --out /path/to/device_profiles.json \
    --device-col device_id \
    --time-col comp_time \
    --energy-col energy \
    --batch-col batch_size \
    --seqlen-col seq_len \
    --steps-col local_steps \
    --rank-col lora_rank \
    --modules-col lora_modules

Notes:
- This code is intentionally defensive: you can map column names via CLI args.
- If LoRA is fixed (rank/modules constant), kappa may be underdetermined; the
  fitter will still report beta_sum, but will warn if the regression is ill-conditioned.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover
    raise RuntimeError("pandas is required for offline profiling. pip install pandas") from e


# -------------------------------
# Data structures
# -------------------------------


@dataclass
class FitConfig:
    device_col: str = "device_id"
    phase_col: Optional[str] = None
    phase_train_value: str = "train"

    # Observations
    time_col: str = "comp_time"          # seconds or milliseconds (see time_unit)
    time_unit: str = "s"                 # "s" or "ms"
    energy_col: Optional[str] = None      # Joules (recommended) or Watt-seconds

    # Workload knobs
    batch_col: str = "batch_size"
    seqlen_col: str = "seq_len"
    steps_col: str = "local_steps"       # steps per round on that device

    # LoRA knobs (optional)
    rank_col: Optional[str] = "lora_rank"
    modules_col: Optional[str] = "lora_modules"  # n(M)

    # Optional: filter
    min_tau: int = 1


@dataclass
class DeviceProfile:
    device_id: str

    # Time coefficients (ms per token)
    beta_sum_ms_per_token: float
    kappa_ms_per_token_per_rank_module: float

    # Optional power baseline (W)
    p_comp_watt_base: Optional[float] = None

    # Basic fit stats
    n_samples: int = 0
    r2: Optional[float] = None
    warnings: Optional[List[str]] = None


# -------------------------------
# Helpers
# -------------------------------


def _ensure_cols(df: "pd.DataFrame", cols: Iterable[str]) -> None:
    missing = [c for c in cols if c and c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")


def _to_ms(time_values: np.ndarray, time_unit: str) -> np.ndarray:
    if time_unit == "ms":
        return time_values.astype(float)
    if time_unit == "s":
        return time_values.astype(float) * 1000.0
    raise ValueError(f"Unsupported time_unit={time_unit}. Use 's' or 'ms'.")


def _r2_score(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _fit_beta_kappa(
    tau: np.ndarray,
    x_rank_modules: np.ndarray,
    t_obs_ms: np.ndarray,
) -> Tuple[float, float, float, List[str]]:
    """Fit y = beta_sum + kappa * x, where y = T/tau.

    Returns: (beta_sum, kappa, r2, warnings)
    """
    warnings: List[str] = []

    # Guard against zero/negative tau
    valid = tau > 0
    tau = tau[valid]
    x = x_rank_modules[valid]
    t = t_obs_ms[valid]

    if len(t) < 2:
        raise ValueError("Not enough samples to fit. Need at least 2.")

    y = t / tau

    # Design matrix: [1, x]
    A = np.column_stack([np.ones_like(x, dtype=float), x.astype(float)])

    # Check conditioning: if x has near-zero variance, kappa is underdetermined.
    if float(np.std(x)) < 1e-9:
        warnings.append(
            "LoRA feature (rank*modules) has ~zero variance; kappa may be underdetermined. "
            "Collect runs with different LoRA ranks or target module sets."
        )

    # Least squares
    coef, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    beta_sum = float(coef[0])
    kappa = float(coef[1])

    y_hat = A @ coef
    r2 = _r2_score(y, y_hat)

    if rank < 2:
        warnings.append("Regression matrix is rank-deficient; fitted coefficients may be unstable.")

    # Sanity checks
    if beta_sum < 0:
        warnings.append("beta_sum fitted negative; check logs/units or add more samples.")
    if kappa < 0:
        warnings.append("kappa fitted negative; check LoRA feature definition or sample diversity.")

    return beta_sum, kappa, r2, warnings


def _estimate_power_watt(energy_j: np.ndarray, time_s: np.ndarray) -> Optional[float]:
    valid = (energy_j > 0) & (time_s > 0)
    if not np.any(valid):
        return None
    p = energy_j[valid] / time_s[valid]
    # Use median to reduce effect of spikes
    return float(np.median(p))


# -------------------------------
# Offline fitting entry
# -------------------------------


CID_LABELS: Dict[str, str] = {
    # 예시)
    "153099042351398445": "jetson-orin",
    "17182517327825067016": "rpi5",
}


def load_csvs(csv_paths: List[str]) -> "pd.DataFrame":
    frames = []
    for p in csv_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        frames.append(pd.read_csv(p))
    if not frames:
        raise ValueError("No CSVs provided")
    return pd.concat(frames, ignore_index=True)


def offline_fit(df: "pd.DataFrame", cfg: FitConfig) -> Dict[str, DeviceProfile]:
    # Optional phase filter
    if cfg.phase_col and cfg.phase_col in df.columns:
        df = df[df[cfg.phase_col].astype(str) == cfg.phase_train_value]

    # Required columns
    req = [cfg.device_col, cfg.time_col, cfg.batch_col, cfg.seqlen_col, cfg.steps_col]
    if cfg.rank_col:
        req.append(cfg.rank_col)
    if cfg.modules_col:
        req.append(cfg.modules_col)
    if cfg.energy_col:
        req.append(cfg.energy_col)
    _ensure_cols(df, req)

    profiles: Dict[str, DeviceProfile] = {}

    for device_id, gdf in df.groupby(cfg.device_col):
        # Extract arrays
        t_obs = gdf[cfg.time_col].to_numpy(dtype=float)
        t_obs_ms = _to_ms(t_obs, cfg.time_unit)

        batch = gdf[cfg.batch_col].to_numpy(dtype=float)
        seqlen = gdf[cfg.seqlen_col].to_numpy(dtype=float)
        steps = gdf[cfg.steps_col].to_numpy(dtype=float)

        tau = steps * batch * seqlen

        # Filter by tau
        valid_tau = tau >= cfg.min_tau
        gdf = gdf.loc[valid_tau]
        tau = tau[valid_tau]
        t_obs_ms = t_obs_ms[valid_tau]

        if len(gdf) < 2:
            raise ValueError(
                f"Device {device_id}: not enough valid rows after filtering (need >=2)."
            )

        # LoRA feature
        if cfg.rank_col and cfg.rank_col in gdf.columns:
            rank = gdf[cfg.rank_col].to_numpy(dtype=float)
        else:
            rank = np.zeros(len(gdf), dtype=float)

        if cfg.modules_col and cfg.modules_col in gdf.columns:
            nmods = gdf[cfg.modules_col].to_numpy(dtype=float)
        else:
            nmods = np.zeros(len(gdf), dtype=float)

        x = rank * nmods

        beta_sum, kappa, r2, warns = _fit_beta_kappa(tau=tau, x_rank_modules=x, t_obs_ms=t_obs_ms)

        # Optional power baseline
        p_base = None
        if cfg.energy_col and cfg.energy_col in gdf.columns:
            energy_j = gdf[cfg.energy_col].to_numpy(dtype=float)
            # time in seconds
            if cfg.time_unit == "s":
                time_s = gdf[cfg.time_col].to_numpy(dtype=float)
            else:
                time_s = gdf[cfg.time_col].to_numpy(dtype=float) / 1000.0
            p_base = _estimate_power_watt(energy_j, time_s)

        profiles[str(device_id)] = DeviceProfile(
            device_id=str(device_id),
            beta_sum_ms_per_token=float(beta_sum),
            kappa_ms_per_token_per_rank_module=float(kappa),
            p_comp_watt_base=p_base,
            n_samples=int(len(gdf)),
            r2=float(r2) if not (isinstance(r2, float) and math.isnan(r2)) else None,
            warnings=warns or None,
        )

    return profiles


def save_profiles_json(profiles: Dict[str, DeviceProfile], out_path: str) -> None:
    payload = {k: asdict(v) for k, v in profiles.items()}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

# -------------------------------
# Simple LLM profiling helper
# -------------------------------

def offline_fit_llm(csv_paths: List[str], out_path: Optional[str] = None) -> Dict[str, DeviceProfile]:
    """Convenience entry point for LLM profiling.

    Assumes CSV schema:
      - device id: 'cid'
      - phase column: 'phase', training rows have value 'train'
      - time: 'comp_time' (seconds)
      - batch: 'batch_size'
      - seq len: 'seq_len'
      - steps: 'local_steps'
    and ignores energy and LoRA variation.
    """

    df = load_csvs(csv_paths)

    cfg = FitConfig(
        device_col="cid",
        phase_col="phase",
        phase_train_value="train",
        time_col="comp_time",
        time_unit="s",
        energy_col=None,
        batch_col="batch_size",
        seqlen_col="seq_len",
        steps_col="local_steps",
        rank_col=None,
        modules_col=None,
        min_tau=1,
    )

    profiles = offline_fit(df, cfg)

    # Optional: map raw cids to human-readable labels (e.g., "jetson-orin", "rpi5")
    if CID_LABELS:
        mapped: Dict[str, DeviceProfile] = {}
        for raw_id, prof in profiles.items():
            dev_key = CID_LABELS.get(raw_id, raw_id)
            # profile 안의 device_id도 사람이 읽기 좋은 이름으로 바꿔줌
            prof.device_id = dev_key
            mapped[dev_key] = prof
        profiles = mapped

    # If out_path is not provided, save next to the first CSV with a default name
    if out_path is None:
        first = csv_paths[0]
        base_dir = os.path.dirname(first) or "."
        out_path = os.path.join(base_dir, "device_profiles_llm.json")

    save_profiles_json(profiles, out_path)

    return profiles


@dataclass
class EMAState:
    alpha_comp: float = 1.0
    p_comp_watt: Optional[float] = None


def ema_update(prev: float, obs: float, lam: float) -> float:
    """Standard EMA update."""
    lam = float(lam)
    if not (0.0 < lam <= 1.0):
        raise ValueError("lam must be in (0, 1]")
    return (1.0 - lam) * float(prev) + lam * float(obs)


# -------------------------------
# CLI
# -------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="e2fl.estimator", add_help=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("offline-fit", help="Fit device coefficients from CSV logs")
    fit.add_argument("--csv", nargs="+", required=True, help="One or more CSV paths")
    fit.add_argument("--out", required=True, help="Output JSON path")

    fit.add_argument("--device-col", default="device_id")
    fit.add_argument("--phase-col", default=None)
    fit.add_argument("--phase-train-value", default="train")

    fit.add_argument("--time-col", default="comp_time")
    fit.add_argument("--time-unit", choices=["s", "ms"], default="s")
    fit.add_argument("--energy-col", default=None)

    fit.add_argument("--batch-col", default="batch_size")
    fit.add_argument("--seqlen-col", default="seq_len")
    fit.add_argument("--steps-col", default="local_steps")

    fit.add_argument("--rank-col", default="lora_rank")
    fit.add_argument("--modules-col", default="lora_modules")

    fit.add_argument("--min-tau", type=int, default=1)

    # Simple LLM profiling mode: just pass CSV paths
    llm = sub.add_parser("llm", help="Simple LLM profiling: just pass CSV paths")
    llm.add_argument("csv", nargs="+", help="One or more LLM profiling CSV paths")
    llm.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: device_profiles_llm.json next to first CSV)",
    )

    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    if args.cmd == "offline-fit":
        cfg = FitConfig(
            device_col=args.device_col,
            phase_col=args.phase_col,
            phase_train_value=args.phase_train_value,
            time_col=args.time_col,
            time_unit=args.time_unit,
            energy_col=args.energy_col if args.energy_col not in ("", "None", None) else None,
            batch_col=args.batch_col,
            seqlen_col=args.seqlen_col,
            steps_col=args.steps_col,
            rank_col=args.rank_col if args.rank_col not in ("", "None", None) else None,
            modules_col=args.modules_col if args.modules_col not in ("", "None", None) else None,
            min_tau=args.min_tau,
        )

        df = load_csvs(args.csv)
        profiles = offline_fit(df, cfg)
        save_profiles_json(profiles, args.out)

        # Print a short summary to stdout
        for dev, prof in profiles.items():
            msg = (
                f"[{dev}] beta_sum={prof.beta_sum_ms_per_token:.6f} ms/token, "
                f"kappa={prof.kappa_ms_per_token_per_rank_module:.6f} ms/token/(rank*module), "
                f"n={prof.n_samples}, r2={prof.r2}"
            )
            if prof.p_comp_watt_base is not None:
                msg += f", P_base={prof.p_comp_watt_base:.3f} W"
            print(msg)
            if prof.warnings:
                for w in prof.warnings:
                    print(f"  - WARNING: {w}")

    elif args.cmd == "llm":
        profiles = offline_fit_llm(args.csv, out_path=args.out)

        # Print a short summary to stdout
        for dev, prof in profiles.items():
            msg = (
                f"[{dev}] beta_sum={prof.beta_sum_ms_per_token:.6f} ms/token, "
                f"n={prof.n_samples}, r2={prof.r2}"
            )
            print(msg)
            if prof.warnings:
                for w in prof.warnings:
                    print(f"  - WARNING: {w}")

    else:
        raise RuntimeError(f"Unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()