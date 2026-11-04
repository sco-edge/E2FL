#!/usr/bin/env python3
"""Supplementary S5 from existing RTX training/validation records only."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from PIL import Image

from plot_epoch_data_availability import (
    WORKSPACE, OUTPUT, STYLE, load_style, sha256, save_csv, pdf_dimensions_cm,
)


PARENT = WORKSPACE / "outputs" / "loracast_long_horizon_100r"
CAP = WORKSPACE / "outputs" / "loracast_exposure_cap_regulation_100r"
CONFIRM = WORKSPACE / "outputs" / "exposure_policy"
FULL_CHECKPOINTS = (0, 10, 20, 30, 40, 41, 42, 50, 60, 70, 80, 90, 100)
POLICIES = ("Exposure-2", "Exposure-4", "Exposure-8", "Distributed-2")
LINES = {
    "Exposure-2": dict(color="black", linestyle="-", marker="o"),
    "Exposure-4": dict(color="black", linestyle="--", marker="s"),
    "Exposure-8": dict(color="black", linestyle="-.", marker="^"),
    "Distributed-2": dict(color="black", linestyle=":", marker="D"),
    "Unbounded": dict(color="black", linestyle="-", marker="o"),
}
DOUBLE = (17.8, 5.1)
SINGLE = (8.6, 5.2)


def jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verified_ids(path: Path) -> set[str]:
    report = json.loads(path.read_text())
    if report.get("errors"):
        raise ValueError(f"Source integrity report has errors: {path}")
    return {item["run_id"] for item in report["run_results"]
            if item.get("status") == "verified"}


def load_run(root: Path, run_id: str, source_paths: set[Path]) -> dict:
    directory = root / "runs" / run_id
    if root == CONFIRM:
        directory = root / "raw" / "confirmation_runs" / run_id
    round_path = directory / "rounds.jsonl"
    validation_path = directory / "validation.jsonl"
    state_path = directory / "state.json"
    rounds = jsonl(round_path)
    evaluations = jsonl(validation_path)
    state = json.loads(state_path.read_text())
    source_paths.update((round_path, validation_path, state_path))
    if [x["round"] for x in rounds] != list(range(1, 101)):
        raise ValueError(f"Incomplete 100-round trajectory: {run_id}")
    if tuple(x["round"] for x in evaluations) != FULL_CHECKPOINTS:
        raise ValueError(f"Incomplete checkpoint sequence: {run_id}")
    if state["round"] != 100 or state["checkpoint_sha256"] != evaluations[-1]["checkpoint_sha256"]:
        raise ValueError(f"Committed final adapter mismatches evaluation: {run_id}")
    if any(x["run_id"] != run_id for x in rounds + evaluations):
        raise ValueError(f"Run ID mismatch: {run_id}")
    if any(x["loss_mode"] != "full_sequence_64_token" or x["seed"] != 777
           for x in evaluations):
        raise ValueError(f"Validation protocol mismatch: {run_id}")
    if any(not math.isclose(x["balanced_client_loss"],
                                (x["pi_loss"] + x["nx_loss"]) / 2,
                                abs_tol=1e-10) for x in evaluations):
        raise ValueError(f"Balanced-client metric mismatch: {run_id}")
    for x in rounds:
        if x["clients"]["nx"]["optimizer_steps"] != x["clients"]["nx"]["samples"]:
            raise ValueError(f"NX samples and executed steps mismatch: {run_id}")
    return {
        "run_id": run_id,
        "rounds": rounds,
        "evaluations": evaluations,
        "round_path": round_path,
        "validation_path": validation_path,
        "state_path": state_path,
        "nx_samples": sum(x["clients"]["nx"]["samples"] for x in rounds),
        "nx_input_tokens": sum(x["clients"]["nx"]["input_tokens"] for x in rounds),
        "pi_input_tokens": sum(x["clients"]["pi"]["input_tokens"] for x in rounds),
        "r100_loss": evaluations[-1]["balanced_client_loss"],
        "r100_sha256": evaluations[-1]["checkpoint_sha256"],
    }


def equivalence_checks(static: dict, source_paths: set[Path]) -> dict:
    # D512 Cap-8 was intentionally not retrained. The frozen normal schedule
    # executes 3,176 samples, below the 8*512 limit at every monotonic prefix.
    nr512 = static[("d512", "Unbounded")]
    if nr512["nx_samples"] != 3176 or nr512["nx_samples"] >= 8 * 512:
        raise ValueError("D512 Cap-8 nonbinding equivalence is not established")
    previous = 0
    for row in nr512["rounds"]:
        current = previous + row["clients"]["nx"]["samples"]
        if current > 8 * 512 or row["clients"]["nx"]["samples"] != row["normal_nx_samples"]:
            raise ValueError("D512 NR exceeds a nonbinding Cap-8 choice")
        previous = current

    # DFull Cap-2/4 were separately run. Compare actual workload, factors
    # (via adapter digests), and validation across every round/checkpoint.
    base = static[("dfull", "Unbounded")]
    dfull = {}
    for policy in ("Exposure-2", "Exposure-4"):
        candidate = static[("dfull", policy)]
        for before, after in zip(base["rounds"], candidate["rounds"]):
            fields = ("normal_nx_samples", "actual_nx_samples_cap", "normal_nx_epochs",
                      "actual_nx_epochs_cap", "normal_nx_steps_cap", "actual_nx_steps_cap",
                      "aggregation_weights", "nx_exposure")
            if any(before[field] != after[field] for field in fields):
                raise ValueError(f"DFull allocation differs in {policy}")
            for client in ("pi", "nx"):
                for field in ("samples", "optimizer_steps", "input_tokens",
                              "selected_sample_ids", "visited_sample_ids"):
                    if before["clients"][client][field] != after["clients"][client][field]:
                        raise ValueError(f"DFull executed work differs in {policy}")
            if after["action"] != "none":
                raise ValueError(f"DFull cap unexpectedly activated in {policy}")
        for before, after in zip(base["evaluations"], candidate["evaluations"]):
            if before["checkpoint_sha256"] != after["checkpoint_sha256"] \
                    or before["balanced_client_loss"] != after["balanced_client_loss"]:
                raise ValueError(f"DFull checkpoint differs in {policy}")
        dfull[policy] = {
            "all_100_round_workloads_equal": True,
            "all_13_checkpoint_sha256_equal": True,
            "r100_nll": candidate["r100_loss"],
            "r100_nx_exposure": candidate["rounds"][-1]["nx_exposure"],
        }
    return {
        "d512_exposure8": {
            "status": "exact_nonbinding_alias_to_Unbounded; no separate RTX run",
            "underlying_run_id": nr512["run_id"],
            "executed_nx_samples": nr512["nx_samples"],
            "limit_samples": 8 * 512,
            "every_prefix_within_limit": True,
        },
        "dfull": dfull,
    }


def load_all(source_paths: set[Path]) -> tuple[dict, dict]:
    parent_stage1 = PARENT / "stage1_integrity.json"
    parent_stage2 = PARENT / "stage2_integrity.json"
    cap_stage1 = CAP / "stage1_integrity.json"
    parent_verified = verified_ids(parent_stage1) | verified_ids(parent_stage2)
    cap_verified = verified_ids(cap_stage1)
    source_paths.update((parent_stage1, parent_stage2, cap_stage1,
                         PARENT / "run_manifest.json", CAP / "run_manifest.json",
                         CAP / "reused_baselines.json", CONFIRM / "experiment_manifest.json",
                         CONFIRM / "run_manifest.json", CONFIRM / "independent_seed_results.csv",
                         CONFIRM / "controller_equivalence.json",
                         PARENT / "source_manifest.json", CAP / "source_manifest.json",
                         CONFIRM / "source_manifest.json"))
    frozen_parent = {x["run_id"]: x for x in json.loads((PARENT / "run_manifest.json").read_text())["runs"]}
    frozen_cap = {x["run_id"]: x for x in json.loads((CAP / "run_manifest.json").read_text())["runs"]}
    static = {}
    for budget in ("d128", "d512"):
        for policy in POLICIES:
            key = {"Exposure-2": "exposurecap2", "Exposure-4": "exposurecap4",
                   "Exposure-8": "exposurecap8", "Distributed-2": "distributedcap2"}[policy]
            if budget == "d512" and policy == "Exposure-8":
                continue
            run_id = f"{budget}_stationary_e1_heterogeneous__{key}"
            root = PARENT if budget == "d128" and policy in ("Exposure-2", "Exposure-4") else CAP
            verified = parent_verified if root == PARENT else cap_verified
            if run_id not in verified:
                raise ValueError(f"Run missing verified integrity: {run_id}")
            planned = frozen_parent if root == PARENT else frozen_cap
            if run_id not in planned or planned[run_id]["seed"] != 42 or planned[run_id]["rounds"] != 100:
                raise ValueError(f"Run not frozen as seed-42/100-round condition: {run_id}")
            static[(budget, policy)] = load_run(root, run_id, source_paths)
        nr_id = f"{budget}_stationary_e1_heterogeneous__nr"
        if nr_id not in parent_verified or nr_id not in frozen_parent:
            raise ValueError(f"NR baseline missing integrity: {nr_id}")
        static[(budget, "Unbounded")] = load_run(PARENT, nr_id, source_paths)
    static[("d512", "Exposure-8")] = static[("d512", "Unbounded")]
    for policy in ("Unbounded", "Exposure-2", "Exposure-4"):
        # Build explicitly to keep the displayed aliases separate from run IDs.
        key = {"Unbounded": "nr", "Exposure-2": "exposurecap2", "Exposure-4": "exposurecap4"}[policy]
        run_id = f"dfull_stationary_e1_heterogeneous__{key}"
        if run_id not in parent_verified or run_id not in frozen_parent:
            raise ValueError(f"DFull run missing integrity: {run_id}")
        static[("dfull", policy)] = load_run(PARENT, run_id, source_paths)
    arrival = {}
    experiment_manifest = json.loads((CONFIRM / "experiment_manifest.json").read_text())
    selector_audit = json.loads((CONFIRM / "controller_equivalence.json").read_text())
    if selector_audit.get("mismatches") or selector_audit.get("checked_rounds", 0) < 100:
        raise ValueError("Final E1 selector / static-cap equivalence audit is unavailable")
    frozen = {x["run_id"]: x for x in experiment_manifest["runs"]}
    with (CONFIRM / "independent_seed_results.csv").open(newline="") as stream:
        summary = {x["run_id"]: x for x in csv.DictReader(stream)}
    for policy, code in (("Unbounded", "nr"), ("Exposure-2", "exposurecap2")):
        run_id = f"arrival_s46_{code}"
        if run_id not in frozen or frozen[run_id]["seed"] != 46:
            raise ValueError(f"Data-arrival scenario not frozen as seed 46: {run_id}")
        run = load_run(CONFIRM, run_id, source_paths)
        if run_id not in summary or summary[run_id]["trajectory_status"] != "actual_independent_RTX_FL":
            raise ValueError(f"Data-arrival result has no actual-trajectory record: {run_id}")
        if summary[run_id]["r100_checkpoint_sha256"] != run["r100_sha256"] or not math.isclose(
            float(summary[run_id]["r100_balanced_nll"]), run["r100_loss"], abs_tol=1e-12
        ):
            raise ValueError(f"Data-arrival summary disagrees with raw run: {run_id}")
        for row in run["rounds"]:
            expected = 128 if row["round"] <= 40 else 512
            if row["nx_eligible_unique_samples"] != expected or row["experiment_seed"] != 46:
                raise ValueError(f"Incorrect logged data-arrival schedule: {run_id}")
        arrival[policy] = run
    if len({run["evaluations"][0]["checkpoint_sha256"] for run in
            list(static.values()) + list(arrival.values())}) != 1:
        raise ValueError("Initial adapter digest differs among S5 runs")
    for budget, denominator in (("d128", 128), ("d512", 512)):
        front, distributed = static[(budget, "Exposure-2")], static[(budget, "Distributed-2")]
        if front["nx_samples"] != distributed["nx_samples"] or front["nx_samples"] != 2 * denominator:
            raise ValueError(f"Scheduled exposure totals differ: {budget}")
        if front["nx_input_tokens"] != distributed["nx_input_tokens"]:
            raise ValueError(f"Scheduled input token totals differ: {budget}")
    return static, arrival


def base_figure(size_cm, cm_to_inch):
    fig, axes = plt.subplots(1, 2, figsize=cm_to_inch(*size_cm))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.25, top=0.76, wspace=0.28)
    return fig, axes


def add_line(ax, xs, ys, label, marker=True):
    style = dict(LINES[label])
    if not marker:
        style.pop("marker")
    return ax.plot(xs, ys, label=label, linewidth=1.4, markersize=3.6,
                   markerfacecolor="white", markeredgecolor="black",
                   markeredgewidth=0.75, **style)[0]


def standard_axes(ax, title, *, x0=0, x1=100):
    ax.set_title(title, loc="left", fontsize=10, pad=3)
    ax.set_xticks((0, 20, 40, 60, 80, 100) if x0 == 0 else (20, 40, 60, 80, 100))
    ax.set_xlim(x0 - 2, x1 + 2)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(axis="y", linestyle="--", color="black", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figure_cap_sensitivity(static, cm_to_inch, plotted):
    fig, axes = base_figure(DOUBLE, cm_to_inch)
    handles = []
    for ax, budget, title in zip(axes, ("d128", "d512"), ("(a) D128", "(b) D512")):
        values = []
        for policy in POLICIES:
            run = static[(budget, policy)]
            # The common R0 is validated and retained in the raw table, while
            # this sensitivity view shows only the post-training checkpoints.
            observations = [x for x in run["evaluations"] if x["round"] >= 10]
            handle = add_line(ax, [x["round"] for x in observations],
                              [x["balanced_client_loss"] for x in observations], policy)
            if budget == "d128":
                handles.append(handle)
            for x in observations:
                plotted.append({
                    "figure": "figS5_cap_sensitivity", "panel": budget, "policy": policy,
                    "round": x["round"], "balanced_validation_nll": x["balanced_client_loss"],
                    "run_id": run["run_id"], "checkpoint_sha256": x["checkpoint_sha256"],
                    "source_status": "exact_nonbinding_alias_to_Unbounded" if
                                     (budget, policy) == ("d512", "Exposure-8") else "actual_RTX_FL",
                })
                values.append(x["balanced_client_loss"])
        standard_axes(ax, title, x0=10)
        span = max(values) - min(values)
        ax.set_ylim(min(values) - span * 0.08, max(values) + span * 0.08)
    fig.supxlabel("FL round", fontsize=10, y=0.06)
    fig.supylabel("Balanced validation NLL", fontsize=10, x=0.01)
    legend = fig.legend(handles, list(POLICIES), loc="upper center",
                        bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False)
    return fig, axes, legend


def figure_schedule(static, cm_to_inch, schedule_rows):
    fig, axes = base_figure(DOUBLE, cm_to_inch)
    handles = []
    for ax, budget, denominator, title in zip(
        axes, ("d128", "d512"), (128, 512), ("(a) D128", "(b) D512")
    ):
        for policy in ("Exposure-2", "Distributed-2"):
            run = static[(budget, policy)]
            cumulative = 0
            xs, ys = [0], [0.0]
            for row in run["rounds"]:
                cumulative += row["clients"]["nx"]["samples"]
                exposure = cumulative / denominator
                if not math.isclose(exposure, row["nx_exposure"], abs_tol=1e-12):
                    raise ValueError(f"Exposure does not match actual processed samples: {run['run_id']}")
                xs.append(row["round"])
                ys.append(exposure)
            handle = add_line(ax, xs, ys, policy, marker=False)
            if budget == "d128":
                handles.append(handle)
            schedule_rows.extend({
                "figure": "figS5_equal_budget_timing", "budget": budget.upper(),
                "policy": policy, "round": r, "cumulative_executed_nx_samples": int(round(v * denominator)),
                "nx_exposure": v, "run_id": run["run_id"], "round_log": str(run["round_path"]),
            } for r, v in zip(xs, ys))
        ax.axhline(2, color="0.5", linestyle=":", linewidth=0.8)
        standard_axes(ax, title)
        ax.set_ylim(-0.08, 2.15)
    fig.supxlabel("FL round", fontsize=10, y=0.06)
    fig.supylabel("NX cumulative exposure", fontsize=10, x=0.01)
    legend = fig.legend(handles, ("Exposure-2", "Distributed-2"), loc="upper center",
                        bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False)
    return fig, axes, legend


def figure_arrival(arrival, cm_to_inch, plotted, schedule_rows, arrival_budget_rows):
    fig, axes = base_figure(DOUBLE, cm_to_inch)
    quality, work = axes
    handles = []
    for policy in ("Unbounded", "Exposure-2"):
        run = arrival[policy]
        evals = run["evaluations"]
        handle = add_line(quality, [x["round"] for x in evals],
                          [x["balanced_client_loss"] for x in evals], policy)
        if policy == "Exposure-2":
            handle.set_linestyle("--")
            handle.set_marker("s")
        handles.append(handle)
        for x in evals:
            plotted.append({
                "figure": "figS5_data_volume_arrival", "panel": "quality", "policy": policy,
                "round": x["round"], "balanced_validation_nll": x["balanced_client_loss"],
                "run_id": run["run_id"], "checkpoint_sha256": x["checkpoint_sha256"],
                "source_status": "actual_RTX_FL_seed46_volume_arrival",
            })
        cumulative = 0
        xs, ys = [0], [0]
        for row in run["rounds"]:
            cumulative += row["clients"]["nx"]["samples"]
            xs.append(row["round"])
            ys.append(cumulative)
            expected_exposure = cumulative / row["nx_eligible_unique_samples"]
            if not math.isclose(expected_exposure, row["nx_exposure"], abs_tol=1e-12):
                raise ValueError(f"Arrival denominator not updated correctly: {run['run_id']}")
        work_line = add_line(work, xs, ys, policy, marker=False)
        if policy == "Exposure-2":
            work_line.set_linestyle("--")
        schedule_rows.extend({
            "figure": "figS5_data_volume_arrival", "budget": "D128_to_D512_R41",
            "policy": policy, "round": r, "cumulative_executed_nx_samples": n,
            "nx_exposure": 0 if r == 0 else n / (128 if r <= 40 else 512),
            "run_id": run["run_id"], "round_log": str(run["round_path"]),
        } for r, n in zip(xs, ys))
    for ax in axes:
        ax.axvline(41, color="0.45", linestyle=":", linewidth=0.8)
    # The ceiling is computed only from the eligible size recorded at each
    # completed round. It jumps when 384 new examples arrive at R41.
    cap_rounds = arrival["Exposure-2"]["rounds"]
    limit_x = [0] + [row["round"] for row in cap_rounds]
    limit_y = [2 * cap_rounds[0]["nx_eligible_unique_samples"]] + [
        2 * row["nx_eligible_unique_samples"] for row in cap_rounds
    ]
    arrival_budget_rows.extend({
        "round": r,
        "eligible_unique_nx_examples": limit // 2,
        "exposure_2_cumulative_sample_limit": limit,
        "source_run_id": arrival["Exposure-2"]["run_id"],
        "source_round_log": str(arrival["Exposure-2"]["round_path"]),
    } for r, limit in zip(limit_x, limit_y))
    budget_handle = work.step(limit_x, limit_y, where="post",
                              color="0.55", linestyle="--", linewidth=0.9,
                              label="Exposure-2 limit")[0]
    standard_axes(quality, "(a) Validation quality")
    standard_axes(work, "(b) NX completed samples")
    quality.set_ylim(1.5, 2.15)
    work.set_ylim(-70, 3400)
    quality.set_ylabel("Balanced validation NLL", fontsize=10)
    work.set_ylabel("Cumulative NX samples", fontsize=10)
    fig.supxlabel("FL round", fontsize=10, y=0.06)
    legend = fig.legend(handles + [budget_handle],
                        ("Unbounded", "Exposure-2", "Exposure-2 limit"), loc="upper center",
                        bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False)
    return fig, axes, legend


def figure_dfull(static, cm_to_inch, plotted):
    run = static[("dfull", "Unbounded")]
    fig, ax = plt.subplots(figsize=cm_to_inch(*SINGLE))
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.22, top=0.88)
    evaluations = run["evaluations"]
    add_line(ax, [x["round"] for x in evaluations],
             [x["balanced_client_loss"] for x in evaluations], "Unbounded")
    for x in evaluations:
        plotted.append({
            "figure": "figS5_dfull_inactive", "panel": "DFull", "policy":
            "Unbounded = Exposure-2 = Exposure-4",
            "round": x["round"], "balanced_validation_nll": x["balanced_client_loss"],
            "run_id": run["run_id"], "checkpoint_sha256": x["checkpoint_sha256"],
            "source_status": "three_actual_RTX_runs_exactly_equal",
        })
    standard_axes(ax, "DFull: exposure caps inactive")
    ax.set_xlabel("FL round", fontsize=10)
    ax.set_ylabel("Balanced validation NLL", fontsize=10)
    fig.text(0.53, 0.81, "Unbounded = Exposure-2 = Exposure-4",
             ha="center", va="center", fontsize=8.5)
    return fig, (ax,), None


def export_and_check(fig, axes, legend, name, size_cm):
    folder = OUTPUT / "supplementary"
    folder.mkdir(parents=True, exist_ok=True)
    paths = {suffix: folder / f"{name}.{suffix}" for suffix in ("pdf", "svg", "png")}
    fig.canvas.draw()
    for ax in axes:
        if not all(math.isclose(t.get_fontsize(), 9, abs_tol=0.01)
                   for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels())):
            raise ValueError(f"Incorrect INFOCOM tick font: {name}")
    if legend is not None and not all(math.isclose(t.get_fontsize(), 8.5, abs_tol=0.01)
                                      for t in legend.get_texts()):
        raise ValueError(f"Incorrect INFOCOM legend font: {name}")
    fig.savefig(paths["pdf"], bbox_inches=None)
    fig.savefig(paths["svg"], bbox_inches=None)
    fig.savefig(paths["png"], bbox_inches=None, dpi=600)
    width, height = pdf_dimensions_cm(paths["pdf"])
    if abs(width - size_cm[0]) > 0.01 or abs(height - size_cm[1]) > 0.01:
        raise ValueError(f"PDF physical size mismatch: {name}")
    svg_text = paths["svg"].read_text()
    if "<text" not in svg_text:
        raise ValueError(f"SVG text is not editable: {name}")
    with Image.open(paths["png"]) as image:
        dpi = image.info.get("dpi")
        if dpi is None or min(dpi) < 599:
            raise ValueError(f"PNG dpi mismatch: {name}")
        pixels = list(image.size)
    plt.close(fig)
    return {
        "figure": name,
        "formats": {x: str(path) for x, path in paths.items()},
        "intended_size_cm": list(size_cm),
        "actual_pdf_size_cm": [width, height],
        "png_dpi": list(dpi),
        "png_pixels": pixels,
        "font_file": font_manager.findfont("Times New Roman"),
        "axis_label_font_pt": 10,
        "tick_font_pt": 9,
        "legend_font_pt": 8.5 if legend is not None else None,
        "line_width_pt": 1.4,
    }


def main():
    cm_to_inch = load_style()
    inputs: set[Path] = {STYLE, OUTPUT / "scripts" / "plot_epoch_data_availability.py"}
    static, arrival = load_all(inputs)
    equivalence = equivalence_checks(static, inputs)
    plotted: list[dict] = []
    schedule_rows: list[dict] = []
    arrival_budget_rows: list[dict] = []
    figures = []
    figure_specs = (
        ("figS5_cap_sensitivity", *figure_cap_sensitivity(static, cm_to_inch, plotted), DOUBLE),
        ("figS5_equal_budget_timing", *figure_schedule(static, cm_to_inch, schedule_rows), DOUBLE),
        ("figS5_data_volume_arrival", *figure_arrival(arrival, cm_to_inch, plotted,
                                                      schedule_rows, arrival_budget_rows), DOUBLE),
        ("figS5_dfull_inactive", *figure_dfull(static, cm_to_inch, plotted), SINGLE),
    )
    for name, fig, axes, legend, size_cm in figure_specs:
        figures.append(export_and_check(fig, axes, legend, name, size_cm))
    if len(plotted) != 8 * 12 + 2 * 13 + 13:
        raise ValueError("Wrong checkpoint count in S5 plotted data")
    if len(schedule_rows) != (4 + 2) * 101:
        raise ValueError("Wrong cumulative executed-work sample count")
    save_csv(OUTPUT / "verification" / "figS5_plot_values.csv", plotted)
    save_csv(OUTPUT / "verification" / "figS5_schedule_values.csv", schedule_rows)
    save_csv(OUTPUT / "verification" / "figS5_arrival_budget_values.csv", arrival_budget_rows)
    summary = []
    for (budget, policy), run in static.items():
        if budget not in ("d128", "d512", "dfull"):
            continue
        if budget == "dfull" or policy in POLICIES:
            summary.append({
                "budget": "DFull" if budget == "dfull" else budget.upper(),
                "policy": policy, "source_run_id": run["run_id"],
                "source_status": "exact_nonbinding_alias_not_new_run" if
                                 (budget, policy) == ("d512", "Exposure-8") else "actual_RTX_FL",
                "training_seed": 42, "r100_balanced_validation_nll": run["r100_loss"],
                "nx_executed_samples_r1_r100": run["nx_samples"],
                "nx_input_tokens_r1_r100": run["nx_input_tokens"],
                "r100_checkpoint_sha256": run["r100_sha256"],
            })
    for policy, run in arrival.items():
        summary.append({
            "budget": "D128_to_D512_R41", "policy": policy, "source_run_id": run["run_id"],
            "source_status": "actual_RTX_FL_data_volume_arrival", "training_seed": 46,
            "r100_balanced_validation_nll": run["r100_loss"],
            "nx_executed_samples_r1_r100": run["nx_samples"],
            "nx_input_tokens_r1_r100": run["nx_input_tokens"],
            "r100_checkpoint_sha256": run["r100_sha256"],
        })
    save_csv(OUTPUT / "verification" / "figS5_r100_summary.csv", summary)
    audit = {
        "status": "PASS for plotted measured RTX trajectories; D512 Exposure-8 is an exact nonbinding alias",
        "equivalence": equivalence,
        "figures": figures,
        "plotted_checkpoint_rows": len(plotted),
        "plotted_cumulative_work_rows": len(schedule_rows),
        "data_arrival": {
            "experiment_seed": 46,
            "availability": "D128 R1-40; D512 R41-100",
            "first_returned_NX_work_after_arrival": next(x["round"] for x in arrival["Exposure-2"]["rounds"]
                                                       if x["round"] >= 41 and x["clients"]["nx"]["samples"] > 0),
            "physical_edge_arrival_validation": "not performed",
            "r100_balanced_nll": {policy: run["r100_loss"] for policy, run in arrival.items()},
        },
        "source_sha256": {str(p): sha256(p) for p in sorted(inputs)},
        "plot_script": str(Path(__file__).resolve()),
        "plot_script_sha256": sha256(Path(__file__).resolve()),
        "measurement": "actual RTX local training and held-out full-sequence 64-token checkpoint NLL",
        "no_new_training_or_scoring": True,
    }
    (OUTPUT / "verification" / "figS5_source_manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    save_csv(OUTPUT / "verification" / "figS5_size_font_audit.csv", [
        {"figure": x["figure"], "intended_width_cm": x["intended_size_cm"][0],
         "intended_height_cm": x["intended_size_cm"][1],
         "actual_pdf_width_cm": x["actual_pdf_size_cm"][0],
         "actual_pdf_height_cm": x["actual_pdf_size_cm"][1],
         "axis_label_font_pt": x["axis_label_font_pt"], "tick_font_pt": x["tick_font_pt"],
         "legend_font_pt": x["legend_font_pt"], "line_width_pt": x["line_width_pt"],
         "font_file": x["font_file"], "png_dpi": round(min(x["png_dpi"])),
         "status": "PASS"}
        for x in figures
    ])


if __name__ == "__main__":
    main()
