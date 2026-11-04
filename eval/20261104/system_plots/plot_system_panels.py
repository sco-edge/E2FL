#!/usr/bin/env python3
"""Export the four conceptual WWW Fig. 4 panels as separate figures.

The panel letters and policy names belong in LaTeX subcaptions. Values are
schematic workload units, shared with plot_system_overview.py, rather than measurements.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_system_overview import HERE, POLICIES, ROUNDS
from common_loracast_plot_v4 import (  # imported via plot_system_overview's style path
    COLORS,
    FONT_SIZE,
    cm_to_inch,
    save_figure,
    setup_matplotlib,
)


def main() -> None:
    setup_matplotlib()
    with plt.rc_context({
        "font.size": FONT_SIZE * 0.36,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.linewidth": 0.8,
    }):
        for letter, (_, values) in zip("abcd", POLICIES):
            fig, ax = plt.subplots(figsize=cm_to_inch(4.1, 3.2))
            fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.97)
            ax.bar(ROUNDS, values, width=0.72, color=COLORS["single"],
                   edgecolor=COLORS["black"], linewidth=0.48, zorder=3)
            ax.set_xlim(0.4, 10.6)
            ax.set_ylim(0, 9)
            ax.set_xticks((1, 5, 10))
            ax.set_yticks((0, 4, 8))
            ax.set_axisbelow(True)
            ax.grid(axis="y", linestyle=":", color=COLORS["gray"],
                    linewidth=0.55, alpha=0.46)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(direction="out", width=0.75, length=2.3, pad=1.5)
            save_figure(fig, HERE, f"plot_system_overview{letter}")


if __name__ == "__main__":
    main()
