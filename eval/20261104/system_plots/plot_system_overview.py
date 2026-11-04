#!/usr/bin/env python3
"""WWW Fig. 4: conceptual training-budget policies (illustrative, not data).

Run from any directory. The script imports the byte-identical local copy of
the EEFL INFOCOM revision plotting helper and writes fig/plot_system_overview.png and PDF.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
STYLE_DIR = PROJECT_ROOT / "exposure_policy/system_design_figures"
sys.path.insert(0, str(STYLE_DIR))
from common_loracast_plot_v4 import (  # noqa: E402
    COLORS,
    FONT_SIZE,
    cm_to_inch,
    save_figure,
    setup_matplotlib,
)


# Deliberately schematic workload units. The two c policies have the same
# cumulative budget: 8+8+8+6 = 10*3 = 30 units. These are not observations.
ROUNDS = tuple(range(1, 11))
BUDGET_C = 30
POLICIES = (
    ("(a) Unbounded", (8, 8, 8, 8, 8, 8, 8, 8, 8, 8)),
    ("(b) Participation", (8, 8, 0, 8, 8, 8, 0, 8, 8, 8)),
    ("(c) Exposure-(c)", (8, 8, 8, 6, 0, 0, 0, 0, 0, 0)),
    ("(d) Distributed-(c)", (3, 3, 3, 3, 3, 3, 3, 3, 3, 3)),
)


def main() -> None:
    assert all(len(values) == len(ROUNDS) for _, values in POLICIES)
    assert all(0 <= value <= 8 for _, values in POLICIES for value in values)
    assert sum(POLICIES[2][1]) == sum(POLICIES[3][1]) == BUDGET_C

    with (HERE / "fig1_illustrative_values.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["round", "unbounded", "participation", "exposure_c",
                         "distributed_c", "provenance"])
        for index, round_number in enumerate(ROUNDS):
            writer.writerow([round_number, *(values[index] for _, values in POLICIES),
                             "schematic example; not measured"])

    setup_matplotlib()
    with plt.rc_context({
        "font.size": FONT_SIZE * 0.36,
        "axes.titlesize": 8.0,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.linewidth": 0.8,
    }):
        fig, axes = plt.subplots(2, 2, figsize=cm_to_inch(8.51, 7.9),
                                 sharex=True, sharey=True)
        fig.subplots_adjust(left=0.18, right=0.975, bottom=0.14, top=0.91,
                            wspace=0.17, hspace=0.30)
        for index, (ax, (title, values)) in enumerate(zip(axes.flat, POLICIES)):
            ax.bar(ROUNDS, values, width=0.72, color=COLORS["single"],
                   edgecolor=COLORS["black"], linewidth=0.48, zorder=3)
            ax.set_title(title, pad=4)
            ax.set_xlim(0.4, 10.6)
            ax.set_ylim(0, 9)
            ax.set_xticks((1, 5, 10))
            ax.set_yticks((0, 4, 8))
            ax.set_axisbelow(True)
            ax.grid(axis="y", linestyle=":", color=COLORS["gray"],
                    linewidth=0.55, alpha=0.46)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(direction="out", width=0.75, length=2.3, pad=1.5,
                           labelbottom=index >= 2, labelleft=index % 2 == 0)
        fig.supxlabel("Federated Round", y=0.035, fontsize=8.2)
        fig.supylabel("Allowed Training Workload", x=0.025, fontsize=8.2)
        save_figure(fig, HERE, "plot_system_overview")


if __name__ == "__main__":
    main()
