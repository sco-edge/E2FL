#!/usr/bin/env python3
"""Rebuild every supplementary evaluation panel as one graph per file.

Uses the same EEFL INFOCOM plotting style and original, verified observations.
The suffix after each figure number is a suggested LaTeX subcaption title.
No FL training, model scoring, device run, or interpolation is performed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from PIL import Image

import plot_pi5_nx_estimator_ablation as s1
import plot_client_workload_variation as s2
import plot_learning_quality_targets as s3
import plot_additional_budget_policies as s5
import plot_historical_estimator_errors as s6
import plot_epoch_data_availability as e7
import plot_training_budget_policies as b8


ROOT = Path(__file__).resolve().parents[1]
SUPP = ROOT / "supplementary"
VERIFY = ROOT / "verification"
SIZE_CM = (8.6, 5.2)
CREATED: list[dict] = []


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def blank(left: float = .19, right: float = .97, bottom: float = .24,
          top: float = .76) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(SIZE_CM[0] / 2.54, SIZE_CM[1] / 2.54))
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    return fig, ax


def tidy(ax: plt.Axes, *, zero_round: bool = True) -> None:
    ax.set_xlim(-2 if zero_round else 8, 102)
    ax.set_xticks((0, 20, 40, 60, 80, 100) if zero_round else
                  (10, 20, 40, 60, 80, 100))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xlabel("FL round")
    ax.grid(axis="y", linestyle="--", color="black", linewidth=.5, alpha=.25)
    ax.set_axisbelow(True)
    ax.tick_params(width=.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig: plt.Figure, name: str, source_paths: list[Path], note: str) -> None:
    SUPP.mkdir(parents=True, exist_ok=True)
    path = SUPP / name
    fig.savefig(path.with_suffix(".pdf"), bbox_inches=None)
    fig.savefig(path.with_suffix(".svg"), bbox_inches=None)
    fig.savefig(path.with_suffix(".png"), bbox_inches=None, dpi=600)
    width, height = e7.pdf_dimensions_cm(path.with_suffix(".pdf"))
    if abs(width - SIZE_CM[0]) > .01 or abs(height - SIZE_CM[1]) > .01:
        raise ValueError(f"Wrong PDF MediaBox: {name}: {width} x {height}")
    if "<text" not in path.with_suffix(".svg").read_text():
        raise ValueError(f"SVG text was outlined: {name}")
    with Image.open(path.with_suffix(".png")) as image:
        dpi = image.info.get("dpi")
        if dpi is None or min(dpi) < 599:
            raise ValueError(f"Wrong PNG DPI: {name}: {dpi}")
    if len(fig.axes) != 1:
        raise ValueError(f"Figure contains {len(fig.axes)} panels: {name}")
    if any(text.get_text().lstrip().startswith(("(a)", "(b)"))
           for ax in fig.axes for text in ax.texts):
        raise ValueError(f"Subcaption title remains inside: {name}")
    CREATED.append({
        "name": name, "suggested_subcaption": name.split("_", 1)[1].replace("_", " ").title(),
        "pdf": str(path.with_suffix(".pdf")),
        "svg": str(path.with_suffix(".svg")),
        "png_600dpi": str(path.with_suffix(".png")),
        "size_cm": list(SIZE_CM), "note": note,
        "sources_sha256": {str(p): digest(p) for p in sorted(set(source_paths))},
    })
    plt.close(fig)


def series_legend(fig: plt.Figure, ax: plt.Axes, columns: int, top: float = .99) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=columns, loc="upper center",
               bbox_to_anchor=(.55, top), frameon=False, handlelength=1.3,
               handletextpad=.35, columnspacing=.65)


def make_s1() -> None:
    values_path = VERIFY / "fig5_s1_pi5_nx_ablation_values.csv"
    records = [{**row, "seed": int(row["seed"])} for row in read_csv(values_path)]
    if len(records) != 8 or {(int(r["seed"]), r["policy"]) for r in records} != {
        (seed, policy) for seed in (43, 44) for policy in s1.LABEL
    }:
        raise ValueError("Incomplete S1 admitted run values")
    summary_path = Path(records[0]["source_policy_summary_csv"])
    if any(r["source_policy_summary_sha256"] != digest(summary_path) for r in records):
        raise ValueError("S1 underlying campaign summary changed")
    for name, field, label in (
        ("figS1a_input_tokens_per_active_second", "tokens_per_active_s", "Input tokens / active s"),
        ("figS1b_energy_per_1k_input_tokens", "j_per_1k_input_tokens", "Energy (J / 1K input tokens)"),
    ):
        fig, ax = blank(left=.18, top=.91)
        s1.plot_panel(ax, records, field, label)
        save(fig, name, [values_path, summary_path],
             "Pi5/NX physical 16-round paired seed-43/44 runs; bar=two-seed mean, dots=individual seeds")


def make_s2() -> None:
    point, provenance = s2.validate()
    if len(point) != 64 or len(provenance["run_ids"]) != 4:
        raise ValueError("S2 pilot incomplete")
    fields = (
        ("figS2a_pi_local_compute_time", "pi_local_compute_s", "Pi local compute (s)"),
        ("figS2b_nx_executed_optimizer_steps", "jetson_executed_optimizer_steps", "NX executed steps"),
        ("figS2c_input_tokens_per_active_second", "round_fl_tokens_per_active_s", "FL input tokens / active s"),
        ("figS2d_server_reply_barrier", "server_dispatch_to_all_replies_s", "Reply barrier (s)"),
    )
    for name, field, ylabel in fields:
        fig, ax = blank(left=.19, right=.965, top=.73)
        s2.plot_panel(ax, point, field=field, ylabel=ylabel)
        series_legend(fig, ax, columns=2)
        save(fig, name, [s2.ROUNDS_FILE, s2.REPLIES_FILE, s2.MATRIX_FILE, s2.MANIFEST_FILE],
             "Pi5/NX single-seed sequential development pilot; four fixed policy variants")


def make_s3() -> None:
    target, _ = s3.validated_targets()
    handles = [Patch(facecolor=s3.TARGET_STYLE[t]["facecolor"], edgecolor="black",
                     linewidth=.9, hatch=s3.TARGET_STYLE[t]["hatch"],
                     label=s3.TARGET_STYLE[t]["label"]) for t in s3.TARGETS]
    for cost, name in (("time", "figS3a_measured_time_to_validation_target"),
                       ("energy", "figS3b_measured_energy_to_validation_target")):
        fig, ax = blank(left=.32, right=.98, top=.75)
        s3.panel(ax, target, cost=cost)
        fig.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(.57, .99),
                   frameon=False, handlelength=1.4, columnspacing=.8)
        save(fig, name, [s3.TIME_SOURCE, s3.ENERGY_SOURCE, s3.ALIGNMENT_SOURCE],
             "Nano+NX seeds 43-45; first observed checkpoint reaching each frozen NLL target; unreached censored")


def make_s4() -> None:
    support_path = VERIFY / "figS4_source_evidence.json"
    support = json.loads(support_path.read_text())
    raw_path = Path(support["source"])
    if digest(raw_path) != support["source_sha256"]:
        raise ValueError("S4 raw 100-round log changed")
    table_path = ROOT / "tables" / "figS4_quota_omission_trace.csv"
    rows = read_csv(table_path)
    if len(rows) != 100 or [int(r["round"]) for r in rows] != list(range(1, 101)):
        raise ValueError("S4 plotted table incomplete")
    samples = [int(r["nx_executed_samples"]) for r in rows]
    assert sum(samples) == 1024
    partial = [r for r in rows if "partial" in r["action"]]
    assert len(partial) == 1 and int(partial[0]["round"]) == 33 and int(partial[0]["nx_executed_samples"]) == 24
    for name, kind in (("figS4a_executed_nx_samples_per_round", "work"),
                       ("figS4b_cumulative_nx_sample_exposure", "exposure")):
        fig, ax = blank(left=.20, right=.97, top=.88)
        xs = [int(r["round"]) for r in rows]
        if kind == "work":
            ax.bar(xs, samples, width=.85, facecolor="white", edgecolor="black", linewidth=.9)
            ax.bar([33], [24], width=.85, facecolor="white", edgecolor="black", linewidth=.9,
                   hatch="///", zorder=3)
            ax.annotate("partial", xy=(33, 24), xytext=(48, 27),
                        arrowprops={"arrowstyle": "->", "linewidth": .8}, ha="left")
            ax.set_ylabel("Executed NX samples")
            ax.set_ylim(0, 39)
        else:
            ax.plot(xs, [float(r["exposure_after"]) for r in rows],
                    color="black", linewidth=1.4, label="Used")
            ax.plot(xs, [float(r["remaining_exposure"]) for r in rows],
                    color="#666666", linestyle="--", linewidth=1.4, label="Remaining")
            ax.axhline(2, color="black", linestyle=":", linewidth=.9)
            ax.set_ylabel("Cumulative exposure")
            ax.set_ylim(0, 2.12)
            ax.legend(loc="center right", frameon=False)
        ax.axvline(33.5, color="black", linestyle=":", linewidth=1.0)
        tidy(ax)
        ax.set_xlim(0, 100)
        save(fig, name, [raw_path, table_path, support_path],
             "D512 Exposure-2, actual seed-43 RTX executed work; partial R33, omitted R34-R100")


def make_s5() -> None:
    source_paths: set[Path] = set()
    static, arrival = s5.load_all(source_paths)
    s5.equivalence_checks(static, source_paths)
    for budget, suffix in (("d128", "a"), ("d512", "b")):
        fig, ax = blank(left=.20, right=.965, top=.70)
        all_values = []
        for policy in s5.POLICIES:
            run = static[(budget, policy)]
            observations = [x for x in run["evaluations"] if x["round"] >= 10]
            s5.add_line(ax, [x["round"] for x in observations],
                        [x["balanced_client_loss"] for x in observations], policy)
            all_values.extend(x["balanced_client_loss"] for x in observations)
        tidy(ax, zero_round=False)
        span = max(all_values) - min(all_values)
        ax.set_ylim(min(all_values) - span * .08, max(all_values) + span * .08)
        ax.set_ylabel("Balanced validation NLL")
        series_legend(fig, ax, columns=2)
        save(fig, f"figS5{suffix}_{budget}_exposure_cap_sensitivity",
             [run["validation_path"] for policy in s5.POLICIES
              for run in [static[(budget, policy)]]],
             f"Actual seed-42 RTX FL; {budget.upper()} Cap-2/4/8/Distributed-2; D512 Cap-8 exact nonbinding alias")
    for budget, denominator, suffix in (("d128", 128, "c"), ("d512", 512, "d")):
        fig, ax = blank(left=.20, right=.965, top=.77)
        for policy in ("Exposure-2", "Distributed-2"):
            run = static[(budget, policy)]
            cumulative = 0
            xs, ys = [0], [0.]
            for row in run["rounds"]:
                cumulative += row["clients"]["nx"]["samples"]
                exposure = cumulative / denominator
                assert math.isclose(exposure, row["nx_exposure"], abs_tol=1e-12)
                xs.append(row["round"])
                ys.append(exposure)
            s5.add_line(ax, xs, ys, policy, marker=False)
        ax.axhline(2, color=".5", linestyle=":", linewidth=.8)
        tidy(ax)
        ax.set_ylim(-.08, 2.15)
        ax.set_ylabel("NX cumulative exposure")
        series_legend(fig, ax, columns=2)
        save(fig, f"figS5{suffix}_{budget}_frontloaded_vs_distributed_exposure",
             [static[(budget, policy)]["round_path"] for policy in ("Exposure-2", "Distributed-2")],
             f"Actual seed-42 RTX completed NX sample exposure; matched total budget={2*denominator} samples")
    fig, ax = blank(left=.20, right=.965, top=.76)
    for policy in ("Unbounded", "Exposure-2"):
        run = arrival[policy]
        xs = [x["round"] for x in run["evaluations"]]
        ys = [x["balanced_client_loss"] for x in run["evaluations"]]
        line = s5.add_line(ax, xs, ys, policy)
        if policy == "Exposure-2":
            line.set_linestyle("--")
            line.set_marker("s")
    ax.axvline(41, color=".45", linestyle=":", linewidth=.8)
    tidy(ax)
    ax.set_ylim(1.5, 2.15)
    ax.set_ylabel("Balanced validation NLL")
    series_legend(fig, ax, columns=2)
    save(fig, "figS5e_data_arrival_validation_loss",
         [arrival[p]["validation_path"] for p in ("Unbounded", "Exposure-2")],
         "Actual seed-46 RTX data-volume arrival at R41; stationary task distribution")
    fig, ax = blank(left=.20, right=.965, top=.75)
    for policy in ("Unbounded", "Exposure-2"):
        run = arrival[policy]
        cumulative = 0
        xs, ys = [0], [0]
        for row in run["rounds"]:
            cumulative += row["clients"]["nx"]["samples"]
            xs.append(row["round"])
            ys.append(cumulative)
        line = s5.add_line(ax, xs, ys, policy, marker=False)
        if policy == "Exposure-2":
            line.set_linestyle("--")
    cap_run = arrival["Exposure-2"]
    ax.step([0] + [r["round"] for r in cap_run["rounds"]],
            [2*cap_run["rounds"][0]["nx_eligible_unique_samples"]] +
            [2*r["nx_eligible_unique_samples"] for r in cap_run["rounds"]],
            where="post", color=".55", linestyle="--", linewidth=.9,
            label="Exposure-2 limit")
    ax.axvline(41, color=".45", linestyle=":", linewidth=.8)
    tidy(ax)
    ax.set_ylim(-70, 3400)
    ax.set_ylabel("Cumulative NX samples")
    series_legend(fig, ax, columns=2)
    save(fig, "figS5f_data_arrival_cumulative_nx_samples",
         [arrival[p]["round_path"] for p in ("Unbounded", "Exposure-2")],
         "Actual seed-46 RTX NX executed samples; eligibility expands 128 to 512 examples at R41")
    fig, ax = blank(left=.20, right=.965, top=.88)
    run = static[("dfull", "Unbounded")]
    s5.add_line(ax, [x["round"] for x in run["evaluations"]],
                [x["balanced_client_loss"] for x in run["evaluations"]], "Unbounded")
    tidy(ax)
    ax.set_ylabel("Balanced validation NLL")
    fig.text(.58, .79, "Unbounded = Exposure-2 = Exposure-4", ha="center", fontsize=8.5)
    save(fig, "figS5g_dfull_inactive_exposure_caps",
         [static[("dfull", policy)]["validation_path"] for policy in
          ("Unbounded", "Exposure-2", "Exposure-4")],
         "Actual seed-42 RTX FL; all three DFull trajectories have equal checkpoints and workloads")


def make_s6() -> None:
    records, raw_sources, _ = s6.validated_samples()
    for name, metric in (
        ("figS6a_historical_compute_time_prediction_error", "compute_time_error_pct"),
        ("figS6b_historical_compute_energy_prediction_error", "compute_energy_error_pct"),
    ):
        fig, ax = blank(left=.18, right=.97, top=.75)
        s6.ecdf_panel(ax, records, metric=metric)
        ax.set_ylabel("Empirical CDF")
        series_legend(fig, ax, columns=1)
        save(fig, name, [s6.SAMPLES, s6.SUMMARY, *map(Path, raw_sources.keys())],
             "Historical July Pi5/NX estimator, post-warm-up local compute error only")


def make_s7() -> None:
    _, plotted, _, inputs = e7.load_runs()
    for budget, letter in (("D512", "a"), ("DFull", "c")):
        for regime, title, index in (("equal_work", "equal_work_epochs", 0),
                                     ("heterogeneous", "window_filling_epochs", 1)):
            fig, ax = blank(left=.20, right=.965, top=.78)
            subset = [r for r in plotted if r["budget"] == budget and r["regime"] == regime]
            if len(subset) != 39:
                raise ValueError(f"Incomplete epoch series: {budget}/{regime}")
            values = []
            for epoch in e7.EPOCHS:
                series = sorted((r for r in subset if r["epoch"] == epoch), key=lambda x: x["round"])
                ys = [r["balanced_validation_nll"] for r in series]
                ax.plot([r["round"] for r in series], ys, color="black", linewidth=1.4,
                        markerfacecolor="white", markeredgecolor="black",
                        markersize=3.8, markeredgewidth=.75, label=f"E{epoch}",
                        **e7.STYLES[epoch])
                values.extend(ys)
            span = max(values) - min(values)
            ax.set_ylim(min(values) - span*.07, max(values) + span*.07)
            tidy(ax)
            ax.set_ylabel("Balanced validation NLL")
            series_legend(fig, ax, columns=3)
            file_letter = chr(ord(letter) + index)
            save(fig, f"figS7{file_letter}_{budget.lower()}_{title}",
                 [Path(r["validation_log"]) for r in subset],
                 f"Actual seed-42 RTX FL, {budget}, {regime}; E1/E2/E4; E2/E4 edge timing feasibility unverified")


def make_s8() -> None:
    table = read_csv(ROOT / "tables" / "fig8_plotted_values.csv")
    rows = [r for r in table if r["budget"] == "DFULL" and r["source_policy"] in ("nr", "full")]
    if len(rows) != 78:
        raise ValueError("DFull seed-43/44/45 checkpoint table incomplete")
    fig, ax = blank(left=.20, right=.93, top=.80)
    for policy in ("nr", "full"):
        group = [r for r in rows if r["source_policy"] == policy]
        for seed in b8.SEEDS:
            run = sorted((r for r in group if int(r["seed"]) == seed), key=lambda r: int(r["round"]))
            ax.plot([int(r["round"]) for r in run], [float(r["balanced_nll"]) for r in run],
                    color=b8.LINE[policy]["color"], linestyle=b8.LINE[policy]["linestyle"],
                    linewidth=.55, alpha=.28)
        means = []
        for round_no in b8.ROUNDS:
            vals = [float(r["balanced_nll"]) for r in group if int(r["round"]) == round_no]
            assert len(vals) == 3
            means.append(sum(vals)/3)
        ax.plot(b8.ROUNDS, means, label=b8.DISPLAY[policy],
                color=b8.LINE[policy]["color"], linestyle=b8.LINE[policy]["linestyle"],
                marker=b8.LINE[policy]["marker"], linewidth=1.4, markersize=3.3)
    tidy(ax)
    ax.set_xlim(0, 100)
    ax.set_ylabel("Full-sequence validation NLL")
    series_legend(fig, ax, columns=2)
    save(fig, "figS8_dfull_confirmation_exposure_cap_inactive",
         [ROOT / "tables" / "fig8_plotted_values.csv",
          ROOT / "verification" / "fig8_source_evidence.json"],
         "RTX independent seeds 43-45; DFull Cap-2 exact nonbinding alias of NR, not independent fourth run")


def main() -> None:
    e7.load_style()
    for action in (make_s1, make_s2, make_s3, make_s4, make_s5, make_s6, make_s7, make_s8):
        action()
    result = {
        "status": "PASS", "panel_count": len(CREATED), "one_axes_per_file": True,
        "physical_size_cm": list(SIZE_CM), "png_dpi": 600,
        "font_file": font_manager.findfont("Times New Roman"),
        "style_source": str(e7.STYLE), "style_sha256": digest(e7.STYLE),
        "plot_script": str(Path(__file__).resolve()),
        "plot_script_sha256": digest(Path(__file__)),
        "figures": CREATED,
    }
    VERIFY.mkdir(parents=True, exist_ok=True)
    (VERIFY / "supplementary_single_panel_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "single_panel_files": len(CREATED)}, indent=2))


if __name__ == "__main__":
    main()
