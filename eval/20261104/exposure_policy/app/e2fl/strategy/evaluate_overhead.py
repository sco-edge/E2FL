

"""Server-side decision overhead microbenchmark.

This script measures the time spent in the *server decision logic* (estimation +
workload allocation) as the number of participating clients N increases.

It intentionally does NOT run an end-to-end Flower simulation. Instead, it
constructs synthetic per-client states and repeatedly calls a pure-Python
allocation routine whose complexity is O(N).

Outputs:
- CSV/JSONL with per-trial timing
- A line plot: median decision time vs N (+ optional p95)

Example:
  python -m E2FL.e2fl.strategy.evaluate_overhead \
    --out-dir ../eval/evaluate/20260302/overhead \
    --ns 2 4 8 16 32 64 128 \
    --trials 200 \
    --p95

Notes for paper writing:
- This is a server-side microbenchmark, not an end-to-end FL throughput result.
- Even with only a few real devices, the decision overhead is measurable because
  the server decision depends only on per-client state inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Synthetic client state
# -------------------------
# Final paper-facing style for a full IEEE single-column figure.
OVERHEAD_FIGSIZE = (3.45, 1.90)

OVERHEAD_FONT_SIZE = 7.5
OVERHEAD_AXIS_LABEL_SIZE = 7.5
OVERHEAD_TICK_SIZE = 7.0
OVERHEAD_LEGEND_SIZE = 6.5

OVERHEAD_STYLES = {
    "median": {
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
        "label": "Median",
    },
    "p95": {
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
        "label": "95th percentile",
    },
}

@dataclass
class ClientState:
    """Minimal per-client state needed for prediction + allocation."""

    cid: int
    # "speed" is a proxy for compute capacity; higher => faster
    speed: float
    # optional comm bandwidth proxy (not strictly needed)
    bw_mbps: float
    # recent observed train tokens and train time (sec) to estimate beta
    tokens_hist: List[float]
    time_hist: List[float]


def _fit_beta_sec_per_token(tokens: np.ndarray, times: np.ndarray) -> float:
    """Fit beta in time = beta * tokens (through origin)."""
    den = float(np.sum(tokens * tokens))
    if den <= 0.0:
        return 0.0
    num = float(np.sum(tokens * times))
    beta = num / den
    return beta if beta > 0.0 else 0.0


def estimate_betas(states: List[ClientState], max_hist: int = 5) -> np.ndarray:
    """Estimate per-client beta (sec/token) using up to max_hist samples.

    Complexity: O(N * H).
    """
    betas = np.zeros(len(states), dtype=np.float64)
    for i, st in enumerate(states):
        h = min(len(st.tokens_hist), len(st.time_hist), max_hist)
        if h <= 0:
            betas[i] = 0.0
            continue
        t = np.asarray(st.tokens_hist[-h:], dtype=np.float64)
        y = np.asarray(st.time_hist[-h:], dtype=np.float64)
        betas[i] = _fit_beta_sec_per_token(t, y)
    return betas


def allocate_workload(
    states: List[ClientState],
    target_round_time_s: float = 800.0,
    min_quota: int = 64,
    max_quota: int = 4096,
    tokens_per_example: float = 256.0,
    max_hist: int = 5,
) -> Dict[int, int]:
    """O(N) server decision logic: estimate + allocate.

    The policy here is intentionally simple but realistic:
    - Estimate beta(sec/token) per client from history.
    - Predict tokens budget per client such that predicted time matches target.
    - Convert tokens -> examples quota and clamp to [min_quota, max_quota].

    Returns:
        dict cid -> max_samples_per_round
    """
    # (1) estimation
    betas = estimate_betas(states, max_hist=max_hist)

    # Replace missing betas with a conservative default derived from speed.
    # (Faster clients => smaller beta)
    for i, st in enumerate(states):
        if betas[i] <= 0.0:
            betas[i] = 1.0 / max(st.speed, 1e-6) * 1e-3  # sec/token

    # (2) allocation
    quotas: Dict[int, int] = {}
    for i, st in enumerate(states):
        beta = float(betas[i])
        # tokens budget to meet target time
        token_budget = target_round_time_s / max(beta, 1e-9)
        # convert to examples
        ex = int(token_budget / max(tokens_per_example, 1.0))
        ex = max(min_quota, min(max_quota, ex))
        quotas[st.cid] = ex

    return quotas


def make_synth_states(
    n: int,
    seed: int,
    hist_len: int = 5,
    tokens_mean: float = 2.0e5,
    tokens_jitter: float = 0.15,
    speed_log_mean: float = 0.0,
    speed_log_std: float = 0.6,
) -> List[ClientState]:
    """Create N synthetic clients with heterogeneous compute speeds."""
    rnd = random.Random(seed)

    states: List[ClientState] = []
    for cid in range(n):
        # lognormal speeds -> heavy tail heterogeneity
        speed = math.exp(rnd.gauss(speed_log_mean, speed_log_std))
        bw = math.exp(rnd.gauss(1.0, 0.5)) * 10.0  # Mbps-ish

        tokens_hist = []
        time_hist = []
        # generate a consistent beta per client (inverse speed)
        true_beta = 1.5e-3 / max(speed, 1e-6)  # sec/token

        for _ in range(hist_len):
            tks = tokens_mean * (1.0 + rnd.uniform(-tokens_jitter, tokens_jitter))
            # time = beta*tokens with small noise
            ts = true_beta * tks * (1.0 + rnd.uniform(-0.05, 0.05))
            tokens_hist.append(float(tks))
            time_hist.append(float(ts))

        states.append(
            ClientState(
                cid=cid,
                speed=float(speed),
                bw_mbps=float(bw),
                tokens_hist=tokens_hist,
                time_hist=time_hist,
            )
        )

    return states


def time_decision(
    n: int,
    trials: int,
    base_seed: int,
    **alloc_kwargs,
) -> np.ndarray:
    """Return per-trial decision times in milliseconds."""
    times_ms = np.zeros(trials, dtype=np.float64)

    # Hold states fixed for a given N to reduce variance; the decision time should
    # mostly depend on N, not on random values.
    states = make_synth_states(n=n, seed=base_seed)

    for k in range(trials):
        t0 = time.perf_counter_ns()
        _ = allocate_workload(states, **alloc_kwargs)
        t1 = time.perf_counter_ns()
        times_ms[k] = (t1 - t0) / 1e6

    return times_ms


def save_trials_jsonl(path: str, n: int, times_ms: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for t in times_ms.tolist():
            f.write(json.dumps({"n_clients": n, "decision_ms": float(t)}) + "\n")


def save_summary_csv(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = [
        "n_clients",
        "median_ms",
        "mean_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
        "trials",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def plot_overhead(
    out_path: str,
    ns: List[int],
    medians: List[float],
    p95s: List[float] | None = None,
    use_p95: bool = False,
    ylog: bool = False,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": OVERHEAD_FONT_SIZE,
            "axes.labelsize": OVERHEAD_AXIS_LABEL_SIZE,
            "xtick.labelsize": OVERHEAD_TICK_SIZE,
            "ytick.labelsize": OVERHEAD_TICK_SIZE,
            "legend.fontsize": OVERHEAD_LEGEND_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(
        figsize=OVERHEAD_FIGSIZE
    )

    median_style = OVERHEAD_STYLES[
        "median"
    ]

    ax.plot(
        ns,
        medians,
        color=median_style["color"],
        marker=median_style["marker"],
        linestyle=median_style["linestyle"],
        linewidth=1.15,
        markersize=2.8,
        markerfacecolor="white",
        markeredgewidth=0.7,
        label=median_style["label"],
    )

    if use_p95 and p95s is not None:
        p95_style = OVERHEAD_STYLES[
            "p95"
        ]

        ax.plot(
            ns,
            p95s,
            color=p95_style["color"],
            marker=p95_style["marker"],
            linestyle=p95_style["linestyle"],
            linewidth=1.0,
            markersize=2.6,
            markerfacecolor="white",
            markeredgewidth=0.7,
            label=p95_style["label"],
        )

        ax.legend(
            frameon=False,
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            fontsize=OVERHEAD_LEGEND_SIZE,
            columnspacing=0.9,
            handletextpad=0.4,
            handlelength=1.7,
            labelspacing=0.25,
        )

    ax.set_xlabel(
        "Number of participating clients",
        fontsize=OVERHEAD_AXIS_LABEL_SIZE,
    )

    ax.set_ylabel(
        "Decision time (ms)",
        fontsize=OVERHEAD_AXIS_LABEL_SIZE,
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=OVERHEAD_TICK_SIZE,
        width=0.6,
        length=2.2,
        pad=1.5,
    )

    if ylog:
        ax.set_yscale("log")

    ax.grid(
        True,
        which="both",
        linestyle=":",
        linewidth=0.45,
        alpha=0.65,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    top_margin = (
        0.76
        if use_p95 and p95s is not None
        else 0.96
    )

    fig.subplots_adjust(
        left=0.16,
        right=0.98,
        bottom=0.24,
        top=top_margin,
    )

    fig.savefig(
        out_path,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory (CSV/JSONL/plot).",
    )
    ap.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32, 64],
        help="List of N (# clients) to benchmark.",
    )
    ap.add_argument("--trials", type=int, default=200, help="# trials per N")
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed")

    # allocation knobs (kept close to your strategy defaults)
    ap.add_argument("--target-round-time-s", type=float, default=800.0)
    ap.add_argument("--min-quota", type=int, default=64)
    ap.add_argument("--max-quota", type=int, default=4096)
    ap.add_argument("--tokens-per-example", type=float, default=256.0)
    ap.add_argument("--max-hist", type=int, default=5)

    # plot toggles
    ap.add_argument("--p95", action="store_true", help="Plot p95 as dashed line")
    ap.add_argument("--ylog", action="store_true", help="Use log-scale for y")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    trials_path = os.path.join(out_dir, "server_decision_trials.jsonl")
    summary_path = os.path.join(out_dir, "server_decision_summary.csv")
    fig_path = os.path.join(out_dir, "fig_server_decision_overhead_vs_clients.pdf")

    # Overwrite trials file for a clean run
    if os.path.exists(trials_path):
        os.remove(trials_path)

    summary_rows: List[Dict] = []
    medians: List[float] = []
    p95s: List[float] = []

    for n in args.ns:
        times_ms = time_decision(
            n=n,
            trials=args.trials,
            base_seed=args.seed,
            target_round_time_s=args.target_round_time_s,
            min_quota=args.min_quota,
            max_quota=args.max_quota,
            tokens_per_example=args.tokens_per_example,
            max_hist=args.max_hist,
        )

        save_trials_jsonl(trials_path, n=n, times_ms=times_ms)

        med = float(np.median(times_ms))
        mean = float(np.mean(times_ms))
        p95 = float(np.percentile(times_ms, 95))

        summary_rows.append(
            {
                "n_clients": int(n),
                "median_ms": med,
                "mean_ms": mean,
                "p95_ms": p95,
                "min_ms": float(np.min(times_ms)),
                "max_ms": float(np.max(times_ms)),
                "trials": int(args.trials),
            }
        )

        medians.append(med)
        p95s.append(p95)

        print(
            f"[N={n:>4}] median={med:.4f}ms mean={mean:.4f}ms p95={p95:.4f}ms (trials={args.trials})"
        )

    save_summary_csv(summary_path, summary_rows)
    plot_overhead(
        fig_path,
        ns=list(args.ns),
        medians=medians,
        p95s=p95s,
        use_p95=bool(args.p95),
        ylog=bool(args.ylog),
    )

    print(f"\n[DONE] trials  -> {trials_path}")
    print(f"[DONE] summary -> {summary_path}")
    print(f"[DONE] fig     -> {fig_path}")


if __name__ == "__main__":
    main()