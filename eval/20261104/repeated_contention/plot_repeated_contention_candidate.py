#!/usr/bin/env python3
"""Render separate, reproducible candidate figures from the verified comparison.

Data inputs are only the summary JSON and round CSV produced by
compare_repeated_schedule.py. The comparison must have completed its full
R1--R100 source-chain and scorer gates before these files exist. This plotter
also checks their mutual consistency and rejects partial or mixed results.

Example:
  python plot_repeated_contention_candidate.py --input-dir comparison --check-only
  python plot_repeated_contention_candidate.py --input-dir comparison

Rendering requires Matplotlib; ``--check-only`` does not import it. No GPU
work, source-run access, or interpolation occurs here. The plotted
throughput and round time are trace-projected *local-compute* quantities;
balanced R100 NLL comes from actual RTX validation. Nano contention energy
is unsupported. The PyramidFL comparator is only its profile-slack component.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile


HERE = Path(__file__).resolve().parent
POLICIES = ("PyramidFL profile-slack", "LoRACast")
DISPLAY = {
    "PyramidFL profile-slack": "PyramidFL profile-slack component",
    "LoRACast": "LoRACast (Exposure-(2))",
}
EXPECTED_BRANCH_MANIFEST_SHA256 = (
    "c8a9b67c33ce900f58932d3a48267b3568499a7083c9b92780a8b5c4043b63de"
)
MEASUREMENT_BOUNDARY = (
    "RTX actual NLL; trace-projected edge local-compute time; no contended energy"
)
REQUIRED_SOURCE_KEYS = (
    "frozen_trace_sha256", "branch_manifest_sha256",
    "loracast_parent_rounds_sha256", "pyramid_rounds_sha256",
    "loracast_rounds_sha256",
)
REQUIRED_ROUND_COLUMNS = (
    "policy", "round", "phase", "fig9_reference_round", "nano_time_factor",
    "nano_executed_steps", "nx_executed_steps", "nano_executed_samples",
    "nx_executed_samples", "input_tokens", "nano_compute_s", "nx_compute_s",
    "round_compute_window_s", "cumulative_tokens",
    "cumulative_compute_window_s", "cumulative_tokens_per_hour",
    "nx_cumulative_exposure",
)
CAPTIONS = {
    "repeated_r1_r100_cumulative_tokens_per_hour": (
        "Seed 42, Qwen Strong DFull, Nano Super + Orin NX. Cumulative executed "
        "input tokens divided by the cumulative trace-projected edge "
        "local-compute window across R1–R100. Shading marks repeated Nano "
        "contention and recovery phases from R50; R1–R49 and unshaded R50+ "
        "rounds use the normal profile. PyramidFL is the profile-slack "
        "component only; LoRACast uses Exposure-(2) with NX floor 256. "
        "These are not measured end-to-end edge Tokens/h."
    ),
    "repeated_r100_balanced_validation_nll": (
        "Seed 42, Qwen Strong DFull, Nano Super + Orin NX. Actual RTX R100 "
        "balanced full-sequence validation NLL from the completed independent "
        "policy trajectories under the same R0 and scorer. Lower is better. "
        "PyramidFL denotes its profile-slack component only; LoRACast uses "
        "Exposure-(2) with NX floor 256."
    ),
    "repeated_r1_r100_round_compute_window": (
        "Seed 42, Qwen Strong DFull, Nano Super + Orin NX. Per-round "
        "trace-projected local-compute window, the maximum of Nano and NX "
        "projected compute times. Shading marks repeated Nano contention and "
        "recovery phases. This diagnostic preserves slow recovery rounds, "
        "including R58, rather than smoothing them. PyramidFL is the "
        "profile-slack component only; LoRACast uses Exposure-(2) with "
        "NX floor 256."
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value: object, label: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid number") from exc
    require(math.isfinite(result), f"{label}: nonfinite number")
    require(not positive or result > 0, f"{label}: expected positive number")
    return result


def integer(value: object, label: str, *, nonnegative: bool = True) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid integer") from exc
    require(str(result) == str(value), f"{label}: noncanonical integer")
    require(not nonnegative or result >= 0, f"{label}: negative integer")
    return result


def close(actual: float, expected: float, label: str) -> None:
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7),
            f"{label}: {actual} differs from {expected}")


def expected_schedule(round_number: int) -> tuple[str, int | None]:
    if round_number < 50:
        return "normal", None
    offset = (round_number - 50) % 12
    if offset < 4:
        return "normal", None
    if offset < 8:
        return "contention", 5 + offset - 4
    return "recovery", 5 + offset - 4


def load_verified_comparison(input_dir: Path) -> tuple[dict, dict[str, list[dict]], Path, Path]:
    summary_path = input_dir / "repeated_contention_summary.json"
    rounds_path = input_dir / "repeated_contention_round_values.csv"
    require(summary_path.is_file() and not summary_path.is_symlink(),
            f"missing verified comparison summary: {summary_path}")
    require(rounds_path.is_file() and not rounds_path.is_symlink(),
            f"missing verified comparison rounds: {rounds_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(isinstance(summary, dict), "comparison summary is not a JSON object")
    require(summary.get("measurement_boundary") == MEASUREMENT_BOUNDARY,
            "comparison measurement boundary changed")
    source = summary.get("source")
    require(isinstance(source, dict), "comparison source provenance is missing")
    for key in REQUIRED_SOURCE_KEYS:
        require(bool(re.fullmatch(r"[0-9a-f]{64}", str(source.get(key, "")))),
                f"missing or malformed comparison source hash: {key}")
    require(source["branch_manifest_sha256"] == EXPECTED_BRANCH_MANIFEST_SHA256,
            "comparison uses a different frozen branch manifest")
    policy_summary = summary.get("policies")
    require(isinstance(policy_summary, dict) and set(policy_summary) == set(POLICIES),
            "comparison policy set differs from the two completed trajectories")
    with rounds_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(set(REQUIRED_ROUND_COLUMNS).issubset(reader.fieldnames or []),
                "round CSV lacks required comparison columns")
        raw_rows = list(reader)
    require(len(raw_rows) == 200, "comparison must contain exactly 100 rounds per policy")
    require(all(row.get("policy") in POLICIES for row in raw_rows),
            "round CSV contains another policy")

    by_policy: dict[str, list[dict]] = {}
    for policy in POLICIES:
        rows = [row for row in raw_rows if row["policy"] == policy]
        require([integer(row["round"], f"{policy} round") for row in rows] ==
                list(range(1, 101)), f"{policy}: R1–R100 is incomplete or out of order")
        info = policy_summary[policy]
        require(isinstance(info, dict), f"{policy}: missing summary")
        require(info.get("loss_mode") == "full_sequence_64_token",
                f"{policy}: validation scorer differs")
        finite_number(info.get("r100_balanced_validation_nll"),
                      f"{policy} R100 balanced NLL", positive=True)
        parsed: list[dict] = []
        cumulative_tokens = 0
        cumulative_window = 0.0
        phase_totals = {p: {"tokens": 0, "time_s": 0.0, "rounds": 0}
                        for p in ("normal", "contention", "recovery")}
        for number, row in enumerate(rows, 1):
            label = f"{policy} R{number}"
            phase, reference = expected_schedule(number)
            require(row["phase"] == phase, f"{label}: phase differs from frozen schedule")
            require(row["fig9_reference_round"] == (str(reference) if reference else ""),
                    f"{label}: Fig. 9 reference differs")
            factor = finite_number(row["nano_time_factor"], f"{label} Nano factor", positive=True)
            if phase == "normal":
                close(factor, 1.0, f"{label} normal Nano factor")
            tokens = integer(row["input_tokens"], f"{label} executed input tokens")
            require(tokens > 0, f"{label}: no executed input tokens")
            nano_steps = integer(row["nano_executed_steps"], f"{label} Nano steps")
            nx_steps = integer(row["nx_executed_steps"], f"{label} NX steps")
            nano_samples = integer(row["nano_executed_samples"], f"{label} Nano samples")
            nx_samples = integer(row["nx_executed_samples"], f"{label} NX samples")
            require(nano_steps == nano_samples and nx_steps == nx_samples,
                    f"{label}: E=1 execution accounting differs")
            if policy == "PyramidFL profile-slack":
                require((nano_steps, nx_steps) == (320, 342),
                        f"{label}: profile-slack workload is not fixed 320/342")
            nano_time = finite_number(row["nano_compute_s"], f"{label} Nano compute")
            nx_time = finite_number(row["nx_compute_s"], f"{label} NX compute")
            require(min(nano_time, nx_time) >= 0, f"{label}: negative projected time")
            window = finite_number(row["round_compute_window_s"],
                                   f"{label} round compute window", positive=True)
            close(window, max(nano_time, nx_time), f"{label} compute window")
            cumulative_tokens += tokens
            cumulative_window += window
            close(float(row["cumulative_tokens"]), cumulative_tokens,
                  f"{label} cumulative input tokens")
            close(finite_number(row["cumulative_compute_window_s"],
                                f"{label} cumulative window"), cumulative_window,
                  f"{label} cumulative window")
            throughput = cumulative_tokens / cumulative_window * 3600
            close(finite_number(row["cumulative_tokens_per_hour"],
                                f"{label} cumulative Tokens/h", positive=True),
                  throughput, f"{label} cumulative Tokens/h")
            exposure = finite_number(row["nx_cumulative_exposure"],
                                     f"{label} NX exposure")
            require(exposure >= 0, f"{label}: negative NX exposure")
            if parsed:
                require(exposure + 1e-12 >= parsed[-1]["nx_exposure"],
                        f"{label}: NX exposure regressed")
            if policy == "LoRACast":
                require(exposure <= 2 + 1e-12, f"{label}: exposure cap exceeded")
            phase_totals[phase]["tokens"] += tokens
            phase_totals[phase]["time_s"] += window
            phase_totals[phase]["rounds"] += 1
            parsed.append({"round": number, "phase": phase, "reference": reference,
                           "factor": factor, "tokens": tokens, "nano_steps": nano_steps,
                           "nx_steps": nx_steps, "nano_samples": nano_samples,
                           "nx_samples": nx_samples, "nano_compute_s": nano_time,
                           "nx_compute_s": nx_time, "window_s": window,
                           "cumulative_tokens_per_hour": throughput,
                           "nx_exposure": exposure})
        require(set(info.get("phase_totals", {})) == set(phase_totals),
                f"{policy}: phase summary coverage differs")
        for phase, totals in phase_totals.items():
            reported = info["phase_totals"][phase]
            require(integer(reported["tokens"], f"{policy} {phase} tokens") == totals["tokens"],
                    f"{policy}: {phase} token total differs")
            require(integer(reported["rounds"], f"{policy} {phase} rounds") == totals["rounds"],
                    f"{policy}: {phase} round count differs")
            close(finite_number(reported["time_s"], f"{policy} {phase} time"),
                  totals["time_s"], f"{policy} {phase} time")
            close(finite_number(reported["tokens_per_hour"],
                                f"{policy} {phase} Tokens/h"),
                  totals["tokens"] / totals["time_s"] * 3600,
                  f"{policy} {phase} Tokens/h")
        require(integer(info.get("total_input_tokens"), f"{policy} total tokens") ==
                cumulative_tokens, f"{policy}: total input tokens differ")
        close(finite_number(info.get("total_projected_local_compute_window_s"),
                            f"{policy} total projected window"), cumulative_window,
              f"{policy} total projected window")
        close(finite_number(info.get("total_projected_tokens_per_hour"),
                            f"{policy} total projected Tokens/h"),
              parsed[-1]["cumulative_tokens_per_hour"],
              f"{policy} total projected Tokens/h")
        require(integer(info.get("nx_executed_samples"), f"{policy} NX samples") ==
                sum(row["nx_samples"] for row in parsed),
                f"{policy}: NX executed-sample total differs")
        require(integer(info.get("nx_omitted_rounds"), f"{policy} NX omitted rounds") ==
                sum(row["nx_samples"] == 0 for row in parsed),
                f"{policy}: NX omission count differs")
        close(finite_number(info.get("nx_final_exposure"), f"{policy} final NX exposure"),
              parsed[-1]["nx_exposure"], f"{policy} final NX exposure")
        by_policy[policy] = parsed
    for idx in range(100):
        a, b = (by_policy[policy][idx] for policy in POLICIES)
        close(a["factor"], b["factor"], f"R{idx + 1} common Nano factor")
    return summary, by_policy, summary_path, rounds_path


def apply_eefl_revision_style() -> None:
    """Match EEFL plot/revision paper_style without another data dependency."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.4,
        "savefig.bbox": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def phase_shading(ax: object) -> list[object]:
    from matplotlib.patches import Patch

    for first in (50, 62, 74, 86, 98):
        for offset, color in ((4, "#e6e6e6"), (8, "#f4f4f4")):
            left = first + offset - 0.5
            right = min(first + offset + 3.5, 100.5)
            if left <= 100.5:
                ax.axvspan(left, right, color=color, linewidth=0, zorder=0)
    ax.axvline(49.5, color="#777777", linestyle=":", linewidth=0.8, zorder=1)
    return [Patch(facecolor="#e6e6e6", edgecolor="none", label="Nano contention"),
            Patch(facecolor="#f4f4f4", edgecolor="none", label="Recovery")]


def common_axes(ax: object) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=0.8, length=3)


def save_three(fig: object, base: Path) -> list[Path]:
    outputs = []
    for suffix in ("pdf", "svg", "png"):
        path = base.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=600 if suffix == "png" else None, bbox_inches=None)
        outputs.append(path)
    return outputs


def plot_cumulative(by_policy: dict[str, list[dict]], out: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(17.8 / 2.54, 5.8 / 2.54))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.21, top=0.79)
    patches = phase_shading(ax)
    styles = (("#333333", "-"), ("#666666", "--"))
    handles = []
    for policy, (color, linestyle) in zip(POLICIES, styles):
        rows = by_policy[policy]
        line, = ax.plot([row["round"] for row in rows],
                        [row["cumulative_tokens_per_hour"] for row in rows],
                        color=color, linestyle=linestyle, linewidth=1.5,
                        label=DISPLAY[policy], zorder=3)
        handles.append(line)
        ax.annotate(f"{rows[-1]['cumulative_tokens_per_hour'] / 1_000_000:.3f}M",
                    (100, rows[-1]["cumulative_tokens_per_hour"]),
                    xytext=(-4, 6 if policy == POLICIES[0] else -8),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=7.1, color=color, zorder=4)
    ax.set_xlim(1, 100)
    ax.set_xticks((1, 20, 40, 50, 60, 80, 100))
    ax.set_xlabel("Round")
    ax.set_ylabel("Projected local-compute Tokens/h")
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda value, _position: f"{value / 1_000_000:g}M" if value >= 1_000_000
        else f"{value / 1_000:g}k"))
    common_axes(ax)
    ax.legend(handles=handles + patches, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=4, frameon=False, fontsize=7.5, columnspacing=1.2)
    paths = save_three(fig, out / "repeated_r1_r100_cumulative_tokens_per_hour")
    plt.close(fig)
    return paths


def plot_nll(summary: dict, out: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    values = [float(summary["policies"][p]["r100_balanced_validation_nll"])
              for p in POLICIES]
    fig, ax = plt.subplots(figsize=(8.6 / 2.54, 5.2 / 2.54))
    fig.subplots_adjust(left=0.38, right=0.95, bottom=0.27, top=0.88)
    gap = max(abs(values[0] - values[1]), 0.003)
    pad = max(gap * 0.7, 0.003)
    ax.set_xlim(min(values) - pad, max(values) + 2.3 * pad)
    ax.set_ylim(-0.6, 1.6)
    for y, value, color, marker in zip((1, 0), values,
                                        ("#333333", "#777777"), ("o", "D")):
        ax.scatter([value], [y], s=42, color=color, marker=marker, zorder=3)
        ax.annotate(f"{value:.4f}", (value, y), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks((1, 0))
    ax.set_yticklabels(("PyramidFL profile-slack\ncomponent",
                        "LoRACast\n(Exposure-(2))"), fontsize=7.5)
    ax.set_xlabel("Balanced validation NLL (RTX actual)")
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=0.8, length=3)
    paths = save_three(fig, out / "repeated_r100_balanced_validation_nll")
    plt.close(fig)
    return paths


def plot_window(by_policy: dict[str, list[dict]], out: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(17.8 / 2.54, 5.8 / 2.54))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.21, top=0.79)
    patches = phase_shading(ax)
    styles = (("#333333", "-"), ("#666666", "--"))
    handles = []
    for policy, (color, linestyle) in zip(POLICIES, styles):
        rows = by_policy[policy]
        line, = ax.plot([row["round"] for row in rows],
                        [row["window_s"] for row in rows],
                        color=color, linestyle=linestyle, linewidth=1.25,
                        label=DISPLAY[policy], zorder=3)
        handles.append(line)
    ax.set_xlim(1, 100)
    ax.set_xticks((1, 20, 40, 50, 60, 80, 100))
    ax.set_xlabel("Round")
    ax.set_ylabel("Projected round compute window (s)")
    common_axes(ax)
    ax.legend(handles=handles + patches, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=4, frameon=False, fontsize=7.5, columnspacing=1.2)
    paths = save_three(fig, out / "repeated_r1_r100_round_compute_window")
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=HERE,
                        help="directory holding compare_repeated_schedule.py output")
    parser.add_argument("--output-dir", type=Path, default=HERE / "candidate_figures")
    parser.add_argument("--check-only", action="store_true",
                        help="validate the completed comparison without writing figures")
    args = parser.parse_args()
    try:
        summary, by_policy, summary_path, rounds_path = load_verified_comparison(args.input_dir)
    except (OSError, ValueError, KeyError, TypeError, csv.Error, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "BLOCKED_INCOMPLETE_OR_INCONSISTENT_COMPARISON",
                          "reason": str(exc)}, indent=2), file=sys.stderr)
        return 2
    values = {policy: {
        "r100_balanced_validation_nll": summary["policies"][policy]["r100_balanced_validation_nll"],
        "r100_cumulative_projected_local_compute_tokens_per_hour":
            by_policy[policy][-1]["cumulative_tokens_per_hour"],
    } for policy in POLICIES}
    pyramid_rate = values[POLICIES[0]]["r100_cumulative_projected_local_compute_tokens_per_hour"]
    loracast_rate = values[POLICIES[1]]["r100_cumulative_projected_local_compute_tokens_per_hour"]
    rate_change_pct = 100 * (loracast_rate / pyramid_rate - 1)
    captions = dict(CAPTIONS)
    captions["repeated_r1_r100_cumulative_tokens_per_hour"] += (
        f" At R100, LoRACast is {loracast_rate:,.0f} versus "
        f"{pyramid_rate:,.0f} Tokens/h for the profile-slack component "
        f"({rate_change_pct:+.2f}%)."
    )
    if args.check_only:
        print(json.dumps({"status": "READY", "rounds_per_policy": 100,
                          "comparison_sha256": {summary_path.name: sha256(summary_path),
                                                rounds_path.name: sha256(rounds_path)},
                          "values": values}, indent=2))
        return 0

    apply_eefl_revision_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".repeated_figures_", dir=args.output_dir) as tmp:
        stage = Path(tmp)
        figures = (plot_cumulative(by_policy, stage) + plot_nll(summary, stage) +
                   plot_window(by_policy, stage))
        require(len(figures) == 9 and all(path.is_file() and path.stat().st_size > 0
                                         for path in figures),
                "one or more candidate figure formats were not rendered")
        caption_path = stage / "candidate_figure_captions.md"
        caption_path.write_text("\n\n".join(
            f"### {stem}\n\n{caption}" for stem, caption in captions.items()) + "\n",
            encoding="utf-8")
        manifest = {
            "status": "PASS",
            "figure_type": "separate_single_seed_repeated_contention_candidates",
            "rounds_per_policy": 100,
            "measurement_boundary": MEASUREMENT_BOUNDARY,
            "pyramidfl_scope": "profile-slack component only",
            "nano_contention_energy": "unsupported",
            "source_sha256": {summary_path.name: sha256(summary_path),
                              rounds_path.name: sha256(rounds_path)},
            "values": values,
            "loracast_relative_tokens_per_hour_pct": rate_change_pct,
            "output_sha256": {path.name: sha256(path) for path in figures + [caption_path]},
            "captions": captions,
        }
        manifest_path = stage / "candidate_figure_source_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
        for path in figures + [caption_path, manifest_path]:
            os.replace(path, args.output_dir / path.name)
    print(json.dumps({"status": "PASS", "output_dir": str(args.output_dir),
                      "figures": sorted({path.stem for path in figures}),
                      "values": values}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
