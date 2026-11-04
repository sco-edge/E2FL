#!/usr/bin/env python3
"""Export Figures 7 and 8 as separate, single-column subcaption panels.

Only existing checkpoint records are read. The source-verification functions
from the original figure scripts are reused so the split cannot silently
change the plotted metric, seed set, or policy implementation.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from PIL import Image


SCRIPT = Path(__file__).resolve()
OUT = SCRIPT.parents[1]
WIDTH_CM = 8.6
HEIGHT_CM = 5.2
EPOCH_STYLE = {
    1: {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    2: {"color": "#D55E00", "linestyle": "--", "marker": "s"},
    4: {"color": "#009E73", "linestyle": "-.", "marker": "D"},
}
POLICY_STYLE = {
    "nr": {"label": "Unbounded", "color": "#0072B2", "linestyle": "-", "marker": "^"},
    "full": {"label": "Hard Rebalancing", "color": "#D55E00", "linestyle": "--", "marker": "s"},
    "exposurecap2": {"label": "Exposure-2", "color": "#009E73", "linestyle": "-.", "marker": "D"},
}
# R41 and R42 remain connected but receive no markers at single-column size.
MARK_EVERY_TEN = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12]


def import_script(stem: str):
    path = SCRIPT.parent / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import verified plot source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[stem] = module
    spec.loader.exec_module(module)
    return module


def load_style() -> None:
    """Use audited EEFL paper typography when its iCloud source is dataless."""
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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": None,
    })


def prepare_axes(*, dot: bool = False):
    fig, ax = plt.subplots(figsize=(WIDTH_CM / 2.54, HEIGHT_CM / 2.54))
    fig.subplots_adjust(left=0.23 if dot else 0.215,
                        right=0.93 if dot else 0.955,
                        bottom=0.23, top=0.88 if dot else 0.79)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(width=0.8, labelsize=9)
    if not dot:
        ax.set_xlim(0, 100)
        ax.set_xticks((0, 20, 40, 60, 80, 100))
        ax.set_xlabel("FL round", fontsize=10)
        ax.set_ylabel("Balanced validation NLL", fontsize=10)
        ax.grid(axis="y", linewidth=0.4, color="#C8C8C8", alpha=0.55)
    return fig, ax


def add_legend(ax, *, policy: bool = False):
    if policy:
        return ax.legend(loc="upper center", bbox_to_anchor=(0.50, 1.35),
                         ncol=2, frameon=False, fontsize=8.5,
                         handlelength=1.55, handletextpad=0.35,
                         columnspacing=0.8, borderaxespad=0)
    return ax.legend(loc="upper center", bbox_to_anchor=(0.50, 1.36),
                     ncol=3, frameon=False, fontsize=8.5,
                     handlelength=1.55, handletextpad=0.35,
                     columnspacing=0.8, borderaxespad=0)


def validate_export(fig, ax, stem: str, expected_lines: int, *, legend: bool = True) -> dict:
    if ax.get_title() or ax.get_title(loc="left") or ax.get_title(loc="right"):
        raise ValueError(f"An in-graph subcaption remains in {stem}")
    if len(fig.axes) != 1 or len(ax.lines) != expected_lines:
        raise ValueError(f"Expected exactly one graph in {stem}: axes={len(fig.axes)}, lines={len(ax.lines)}")
    fig.canvas.draw()
    if any(not math.isclose(t.get_fontsize(), 9, abs_tol=0.01)
           for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels())):
        raise ValueError(f"Unexpected tick font size in {stem}")
    key = ax.get_legend()
    if legend:
        if key is None or len(key.get_texts()) != 3:
            raise ValueError(f"Missing three-series legend in {stem}")
        if any(not math.isclose(t.get_fontsize(), 8.5, abs_tol=0.01)
               for t in key.get_texts()):
            raise ValueError(f"Unexpected legend font size in {stem}")
    elif key is not None:
        raise ValueError(f"Unexpected legend in directly labeled dot plot {stem}")
    base = OUT / "main" / stem
    base.parent.mkdir(parents=True, exist_ok=True)
    for extension, options in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(base.with_suffix("." + extension), bbox_inches=None, **options)
    output = subprocess.check_output(["pdfinfo", str(base.with_suffix(".pdf"))], text=True)
    size_line = next(line for line in output.splitlines() if line.startswith("Page size:"))
    parts = size_line.split()
    measured_cm = [float(parts[2]) / 72 * 2.54, float(parts[4]) / 72 * 2.54]
    if abs(measured_cm[0] - WIDTH_CM) > 0.01 or abs(measured_cm[1] - HEIGHT_CM) > 0.01:
        raise ValueError(f"Wrong PDF MediaBox in {stem}: {measured_cm}")
    if "<text" not in base.with_suffix(".svg").read_text():
        raise ValueError(f"SVG text is not editable in {stem}")
    with Image.open(base.with_suffix(".png")) as image:
        dpi = image.info.get("dpi")
        if dpi is None or min(dpi) < 599:
            raise ValueError(f"PNG dpi check failed for {stem}: {dpi}")
        pixels = list(image.size)
    matched_font = font_manager.findfont("Times New Roman")
    if "Times New Roman" not in matched_font or matplotlib.rcParams["pdf.fonttype"] != 42:
        raise ValueError(f"Reference font or editable PDF text settings differ in {stem}")
    return {
        "stem": stem,
        "files": {ext: str(base.with_suffix("." + ext)) for ext in ("pdf", "svg", "png")},
        "graph_count": len(fig.axes),
        "internal_subcaption": False,
        "size_cm": measured_cm,
        "png_pixels": pixels,
        "png_dpi": list(dpi),
        "font_file": matched_font,
        "line_count": expected_lines,
        "status": "PASS",
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_epoch_line(ax, rows: list[dict]) -> None:
    for epoch in (1, 2, 4):
        trajectory = sorted((row for row in rows if row["epoch"] == epoch),
                            key=lambda row: row["round"])
        style = EPOCH_STYLE[epoch]
        ax.plot([row["round"] for row in trajectory],
                [row["balanced_validation_nll"] for row in trajectory],
                label=f"E{epoch}", color=style["color"],
                linestyle=style["linestyle"], marker=style["marker"],
                linewidth=1.65, markersize=4.2, markevery=MARK_EVERY_TEN,
                markerfacecolor="white", markeredgecolor=style["color"],
                markeredgewidth=1.0, zorder=3)
    values = [row["balanced_validation_nll"] for row in rows]
    span = max(values) - min(values)
    ax.set_ylim(min(values) - 0.07 * span, max(values) + 0.07 * span)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    add_legend(ax)


def draw_equal_work_dot(ax, rows: list[dict]) -> list[dict]:
    endpoints = sorted((row for row in rows if row["round"] == 100),
                       key=lambda row: row["epoch"])
    if len(endpoints) != 3:
        raise ValueError("Fig. 7(a) must have one R100 endpoint per epoch")
    ax.set_xlim(1.578, 1.600)
    ax.set_xticks((1.580, 1.585, 1.590, 1.595, 1.600))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_xlabel("R100 balanced validation NLL", fontsize=10)
    ax.set_ylim(0.7, 3.3)
    ax.set_yticks((3, 2, 1), ("E1", "E2", "E4"))
    ax.grid(axis="x", color="#C8C8C8", linewidth=0.4, alpha=0.55)
    for y, row in zip((3, 2, 1), endpoints):
        epoch = row["epoch"]
        value = row["balanced_validation_nll"]
        style = EPOCH_STYLE[epoch]
        ax.plot(value, y, color=style["color"], marker=style["marker"],
                linestyle="None", markersize=7.0,
                markerfacecolor="white", markeredgewidth=1.6, zorder=4)
        ax.text(value + 0.0008, y, f"{value:.4f}", ha="left", va="center",
                fontsize=9, color="#222222")
    return endpoints


def figure7() -> dict:
    source = import_script("plot_epoch_data_availability")
    load_style()
    _, all_rows, _, _ = source.load_runs()
    equal_rows = [row for row in all_rows if row["budget"] == "D128"
                  and row["regime"] == "equal_work"]
    window_rows = [row for row in all_rows if row["budget"] == "D128"
                   and row["regime"] == "heterogeneous"]
    if len(equal_rows) != 3 * len(source.CHECKPOINT_ROUNDS) or len(window_rows) != len(equal_rows):
        raise ValueError("Incomplete D128 epoch checkpoint series")

    # Compare full-range line and R100 dot views at the actual paper size.
    candidate_dir = OUT / "verification"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    line_fig, line_ax = prepare_axes()
    draw_epoch_line(line_ax, equal_rows)
    line_fig.savefig(candidate_dir / "fig7a_candidate_line.png", dpi=600, bbox_inches=None)
    plt.close(line_fig)

    dot_fig, dot_ax = prepare_axes(dot=True)
    endpoints = draw_equal_work_dot(dot_ax, equal_rows)
    dot_fig.savefig(candidate_dir / "fig7a_candidate_dot.png", dpi=600, bbox_inches=None)
    outputs = [validate_export(dot_fig, dot_ax, "fig7a_equal_work_local_epochs", 3, legend=False)]
    plt.close(dot_fig)

    fig, ax = prepare_axes()
    draw_epoch_line(ax, window_rows)
    outputs.append(validate_export(fig, ax, "fig7b_window_filling_local_epochs", 3))
    plt.close(fig)
    plotted = ([{"output_stem": "fig7a_equal_work_local_epochs", **row} for row in endpoints]
               + [{"output_stem": "fig7b_window_filling_local_epochs", **row} for row in window_rows])
    save_csv(candidate_dir / "fig7_single_panel_plotted_values.csv", plotted)
    save_csv(candidate_dir / "fig7a_equal_work_source_checkpoints.csv",
             [{"output_stem": "fig7a_equal_work_local_epochs", **row} for row in equal_rows])
    by_epoch = {e: {row["round"]: row["balanced_validation_nll"]
                    for row in equal_rows if row["epoch"] == e} for e in (1, 2, 4)}
    prior_rounds = [r for r in source.CHECKPOINT_ROUNDS if r not in (0, 100)]
    e1_best_prior = sum(by_epoch[1][r] == min(by_epoch[e][r] for e in (1, 2, 4))
                        for r in prior_rounds)
    comparison = {
        "selected": "R100 dot plot",
        "candidates": [str(candidate_dir / "fig7a_candidate_line.png"),
                       str(candidate_dir / "fig7a_candidate_dot.png")],
        "rationale": "Full-range line trajectories overlap at 8.6 cm. Directly labeled R100 dots resolve the small final NLL span; all R0-R100 checkpoints remain in the source-checkpoint CSV.",
        "limitation": "R100 ranking does not describe earlier learning or establish a multi-seed effect.",
        "seed": 42,
        "prior_noninitial_checkpoints": len(prior_rounds),
        "e1_lowest_prior_checkpoints": e1_best_prior,
        "endpoints": {f"E{row['epoch']}": row["balanced_validation_nll"] for row in endpoints},
    }
    (candidate_dir / "fig7a_candidate_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    return {
        "source_verification": str(OUT / "verification/fig7_source_manifest.json"),
        "source_script": str(source.SCRIPT),
        "source_script_sha256": source.sha256(source.SCRIPT),
        "run_count": 6,
        "source_checkpoint_values": len(equal_rows) + len(window_rows),
        "plotted_values": len(plotted),
        "measurement": "Actual RTX full-sequence 64-token balanced validation NLL; training seed 42",
        "caveat": "Equal work matches optimizer steps, not input tokens or edge time. E2/E4 window-filling edge feasibility is not validated.",
        "fig7a_comparison": comparison,
        "outputs": outputs,
    }


def draw_policy_lines(ax, subset, source, stem: str) -> list[dict]:
    means_out = []
    endpoints = {}
    for policy in source.POLICIES:
        policy_rows = subset.loc[subset.source_policy.eq(policy)]
        means = (policy_rows.groupby("round", as_index=False)
                 .balanced_nll.mean().sort_values("round"))
        if means["round"].tolist() != list(source.ROUNDS):
            raise ValueError(f"Incomplete mean trajectory for {policy}")
        for round_number in source.ROUNDS:
            actual_seeds = sorted(policy_rows.loc[policy_rows["round"].eq(round_number), "seed"].tolist())
            if actual_seeds != list(source.SEEDS):
                raise ValueError(f"Unexpected seed set for {policy} R{round_number}: {actual_seeds}")
        style = POLICY_STYLE[policy]
        ax.plot(means["round"], means.balanced_nll,
                label=style["label"], color=style["color"],
                linestyle=style["linestyle"], marker=style["marker"],
                linewidth=1.8 if policy == "nr" else 1.55,
                markersize=4.2, markevery=MARK_EVERY_TEN,
                markerfacecolor="white", markeredgewidth=1.0,
                zorder=3 if policy == "nr" else 4)
        endpoints[policy] = float(means.iloc[-1].balanced_nll)
        for _, row in means.iterrows():
            means_out.append({
                "output_stem": stem,
                "budget": subset.budget.iloc[0],
                "policy": style["label"], "source_policy": policy,
                "round": int(row["round"]),
                "balanced_nll_mean": float(row.balanced_nll),
                "seed_count": len(source.SEEDS), "seeds": "43,44,45",
                "metric": "heldout_full_sequence_64_token_balanced_client_nll",
                "provenance": "actual_RTX_training_and_scoring",
            })
    values = [row["balanced_nll_mean"] for row in means_out]
    span = max(values) - min(values)
    ax.set_xlim(0, 116)
    bottom_pad = 0.09 if subset.budget.iloc[0] == "D512" else 0.045
    ax.set_ylim(min(values) - bottom_pad * span, max(values) + 0.055 * span)
    ax.set_ylabel("Full-sequence validation NLL", fontsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    add_legend(ax, policy=True)

    # Stagger nearby D512 labels and point each back to its exact R100 mean.
    offsets = {"nr": 0.0, "full": 0.0, "exposurecap2": 0.0}
    if subset.budget.iloc[0] == "D512":
        offsets.update({"full": 0.045, "exposurecap2": 0.0})
    for policy in source.POLICIES:
        value = endpoints[policy]
        style = POLICY_STYLE[policy]
        ax.annotate(f"{value:.3f}", xy=(100, value), xycoords="data",
                    xytext=(104, value + offsets[policy]), textcoords="data",
                    ha="left", va="center", fontsize=8.5, color=style["color"],
                    arrowprops={"arrowstyle": "-", "color": style["color"],
                                "linewidth": 0.65, "shrinkA": 0, "shrinkB": 4})
    return means_out


def figure8() -> dict:
    source = import_script("plot_training_budget_policies")
    load_style()
    table, _, provenance = source.verify_sources()
    outputs = []
    means_rows = []
    for budget, stem in (
        ("d128", "fig8a_d128_training_budget_management"),
        ("d512", "fig8b_d512_training_budget_management"),
    ):
        subset = table.loc[table.budget.eq(budget.upper())]
        if len(subset) != 3 * len(source.SEEDS) * len(source.ROUNDS):
            raise ValueError(f"Incomplete policy trajectories: {budget}")
        fig, ax = prepare_axes()
        means_rows.extend(draw_policy_lines(ax, subset, source, stem))
        outputs.append(validate_export(fig, ax, stem, 3))
        plt.close(fig)
    save_csv(OUT / "verification/fig8_single_panel_plotted_values.csv", means_rows)
    save_csv(OUT / "verification/fig8_revised_data_mapping.csv", [{
        "panel": panel,
        "source": "raw/confirmation_runs/confirm_s{43,44,45}_" + budget +
                  "_{nr,full,exposurecap2}/validation.jsonl",
        "source_field": "balanced_client_loss",
        "plotted_field": "balanced_nll_mean",
        "transformation": "Arithmetic mean of all three independent seeds at each observed round; no seed trajectories shown",
        "measurement": "RTX actual held-out full-sequence 64-token balanced client NLL",
    } for panel, budget in (("8a", "d128"), ("8b", "d512"))])
    r100_means = {
        budget: {policy: next(row["balanced_nll_mean"] for row in means_rows
                              if row["budget"] == budget and row["source_policy"] == policy
                              and row["round"] == 100)
                 for policy in source.POLICIES}
        for budget in ("D128", "D512")
    }
    return {
        "source_verification": str(OUT / "verification/fig8_source_evidence.json"),
        "source_script": str(source.Path(source.__file__).resolve()),
        "source_script_sha256": source.sha256(source.Path(source.__file__).resolve()),
        "style_source": "Local rcParams matching the audited EEFL paper_style typography and vector font settings",
        "reference_style_source": str(source.STYLE_SOURCE),
        "reference_style_status": "iCloud dataless; not read or modified for this revision",
        "run_count": 18,
        "source_checkpoint_values": len(table.loc[table.budget.isin(["D128", "D512"])]),
        "plotted_mean_values": len(means_rows),
        "r100_mean_nll": r100_means,
        "measurement": "Actual RTX full-sequence 64-token balanced validation NLL; seeds 43–45",
        "policy_scope": provenance["policy_version"],
        "outputs": outputs,
    }


def main() -> None:
    checks = (("fig7_single_panel_validation.json", figure7()),
              ("fig8_single_panel_validation.json", figure8()))
    for name, record in checks:
        (OUT / "verification" / name).write_text(json.dumps(record, indent=2) + "\n")
    print("Exported four validated single-graph Fig. 7/8 files.")


if __name__ == "__main__":
    main()
