#!/usr/bin/env python3
"""Plot actual Flower server decision timings at ACM single-column size.

Input is the unmodified summary.csv from benchmark_actual_server.py.  This
script deliberately plots only server policies with a registered Flower
strategy.  Distributed-(2) is an RTX run_adaptive_training rule, not a Flower controller
path, so it has no latency series in this figure.

Example:
    /usr/local/bin/python3.7 plot_server_decision_overhead.py \
        --summary ../summary.csv --out-dir ../figures
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter


COUNTS = (2, 4, 8, 16, 32, 64, 128, 250)
MAIN_SERIES = (
    ("Unbounded", "normal", "Unbounded", "#777777", "o", "-"),
    ("Rebalancing", "normal", "Rebalancing", "#D55E00", "s", "--"),
    ("Exposure-(2)", "nonbinding", "Exposure-(2)", "#0072B2", "^", "-."),
)
ACTIVE_SERIES = (
    ("Unbounded", "normal", "Unbounded", "#777777", "o", "-"),
    ("Rebalancing", "trigger", "Rebalancing", "#D55E00", "s", "--"),
    ("Exposure-(2)", "partial", "Exposure-(2)", "#0072B2", "^", "-."),
)
FIGURES = (
    (
        "server_decision_overhead_matched_normal",
        MAIN_SERIES,
        "Server-side decision latency under matched normal client histories. "
        "Points show the mean of repeated calls to the current Flower strategy "
        "decision hook with the same measured NX anchor and Pi-profile client "
        "states. The Rebalancing trigger is inactive and the Exposure-(2) budget "
        "is nonbinding. The client-population expansion beyond two devices is "
        "synthetic. Training, aggregation, network transfer, and Flower message "
        "construction are outside the timed boundary.",
        "All three policies use the same normal synthetic client history; "
        "Rebalancing is not triggered and Exposure-(2) is nonbinding.",
    ),
    (
        "server_decision_overhead_active_paths",
        ACTIVE_SERIES,
        "Server-side decision latency for active policy branches. Unbounded "
        "uses a normal history, Rebalancing uses a synthetic trigger history, "
        "and Exposure-(2) has a partially remaining budget. These state histories "
        "differ, so differences cannot be attributed solely to policy code. "
        "Points are means of repeated calls to the current Flower decision hook; "
        "training, aggregation, network transfer, and Flower message construction "
        "are excluded.",
        "Unbounded is normal; Rebalancing has a synthetic trigger history; "
        "Exposure-(2) has a partially remaining budget. State histories differ.",
    ),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_summary(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"clients", "policy", "state", "mean_ms", "p95_ms"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError("summary CSV missing columns: " + ", ".join(sorted(missing)))
        result = {}
        for source in reader:
            try:
                n = int(source["clients"])
                mean_ms = float(source["mean_ms"])
                p95_ms = float(source["p95_ms"])
            except (ValueError, TypeError) as exc:
                raise ValueError("invalid numeric summary row: " + repr(source)) from exc
            key = (n, source["policy"].strip(), source["state"].strip())
            if key in result:
                raise ValueError("duplicate summary row: " + repr(key))
            if not (math.isfinite(mean_ms) and mean_ms > 0):
                raise ValueError("nonpositive or nonfinite mean latency: " + repr(key))
            if not (math.isfinite(p95_ms) and p95_ms > 0):
                raise ValueError("nonpositive or nonfinite p95 latency: " + repr(key))
            record = dict(source)
            record["clients"] = n
            record["mean_ms"] = mean_ms
            record["p95_ms"] = p95_ms
            result[key] = record
    return result


def select_rows(rows, series):
    selected = []
    for policy, state, label, _color, _marker, _line in series:
        for n in COUNTS:
            key = (n, policy, state)
            if key not in rows:
                raise ValueError("required measured condition missing: " + repr(key))
            item = rows[key]
            selected.append({
                "clients": n,
                "policy": policy,
                "state": state,
                "legend_label": label,
                "mean_ms": item["mean_ms"],
                "p95_ms": item["p95_ms"],
                "observations": item.get("observations", ""),
            })
    return selected


def set_style():
    # INFOCOM revision overhead style, resized from 3.25 x 1.40 in to the
    # paper's requested 8.6 x 5.2 cm single-column footprint.
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
    })


def plot(rows, series, stem, out_dir):
    set_style()
    fig, ax = plt.subplots(figsize=(8.6 / 2.54, 5.2 / 2.54))
    for policy, state, label, color, marker, line in series:
        values = [r for r in rows if r["policy"] == policy and r["state"] == state]
        ax.plot(
            [r["clients"] for r in values],
            [r["mean_ms"] for r in values],
            color=color, marker=marker, linestyle=line,
            linewidth=1.15, markersize=3.2, markerfacecolor="white",
            markeredgewidth=0.8, label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlim(1.8, 282)
    ax.set_xticks(COUNTS)
    ax.set_xticklabels([str(n) for n in COUNTS])
    ax.minorticks_off()
    ax.set_xlabel("Number of clients", labelpad=2.0)
    ax.set_ylabel("Decision latency (ms)", labelpad=2.0)
    # The per-client correction reaches hundreds of ms at N=250 while the
    # normal path stays near a few ms. Log scale keeps all measured policies
    # legible without clipping or inventing a broken axis.
    ax.set_yscale("log")
    ax.set_ylim(0.4, 1000)
    ax.set_yticks((0.5, 1, 10, 100, 1000))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    ax.minorticks_off()
    ax.tick_params(axis="both", which="major", width=0.6, length=2.2, pad=1.6)
    ax.grid(axis="y", which="major", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Three policy names fit in one row without covering the data.
    ax.legend(
        frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.005),
        columnspacing=0.65, handletextpad=0.32, handlelength=1.35,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.255, top=0.785)
    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(out_dir / (stem + "." + extension), dpi=600)
    plt.close(fig)


def write_provenance(rows, summary_path, out_dir, stem, series, caption, scope):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / (stem + "_plotted_values.csv")).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "figure": stem,
        "source_csv": str(summary_path.resolve()),
        "source_sha256": sha256(summary_path),
        "plotted_policy_states": [
            {"policy": policy, "state": state, "legend_label": label}
            for policy, state, label, *_ in series
        ],
        "not_plotted": {
            "policy": "Distributed-(2)",
            "reason": "No registered Flower server decision path exists for this RTX-only rule.",
        },
        "metric": "arithmetic mean of measured synchronous decision latency in milliseconds",
        "axis_scale": {"x": "log2", "y": "log10"},
        "figure_size_cm": [8.6, 5.2],
        "dpi_png": 600,
        "font": "Times New Roman",
        "comparison_scope": scope,
    }
    (out_dir / (stem + "_manifest.json")).write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / (stem + "_caption.md")).write_text(
        caption + " Distributed-(2) has no registered Flower server decision path "
        "and is not plotted. Both axes use logarithmic scales.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.summary.is_file():
        raise FileNotFoundError(args.summary)
    summary = load_summary(args.summary)
    for stem, series, caption, scope in FIGURES:
        selected = select_rows(summary, series)
        plot(selected, series, stem, args.out_dir)
        write_provenance(selected, args.summary, args.out_dir,
                         stem, series, caption, scope)
        print("wrote " + str(args.out_dir / stem) + ".{pdf,svg,png}")


if __name__ == "__main__":
    main()
