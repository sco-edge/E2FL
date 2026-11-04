#!/usr/bin/env python3
"""Rebuild WWW Fig. 7 from the committed RTX run logs, without rescoring.

The original EEFL INFOCOM plotting module supplies the exact paper style.
This script reads only the frozen Stage-1 stationary NR runs and exports the
observed 64-token full-sequence balanced-client validation checkpoints.
"""

from __future__ import annotations

import csv
import hashlib
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
from matplotlib.ticker import MaxNLocator


SCRIPT = Path(__file__).resolve()
WORKSPACE = SCRIPT.parents[3]
OUTPUT = SCRIPT.parents[1]
SOURCE = WORKSPACE / "outputs" / "loracast_long_horizon_100r"
DERIVED = WORKSPACE / "outputs" / "exposure_policy" / "epoch_data_sensitivity.csv"
STYLE = Path("/Users/jang-won-u/Documents/GitHub/EEFL/plot/revision/plot_loracast_revision.py")
FIGURE_WIDTH_CM = 17.8
FIGURE_HEIGHT_CM = 5.1
EPOCHS = (1, 2, 4)
BUDGETS = ("d128", "d512", "dfull")
REGIMES = ("equal_work", "heterogeneous")
CHECKPOINT_ROUNDS = (0, 10, 20, 30, 40, 41, 42, 50, 60, 70, 80, 90, 100)
STYLES = {
    1: dict(linestyle="-", marker="o"),
    2: dict(linestyle="--", marker="s"),
    4: dict(linestyle="-.", marker="^"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_style():
    if not STYLE.is_file():
        raise FileNotFoundError(f"INFOCOM reference style is missing: {STYLE}")
    spec = importlib.util.spec_from_file_location("plot_loracast_revision", STYLE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the EEFL style module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.paper_style()
    # The reference savefig helper emits PDF/PNG only. Preserve its rcParams,
    # add editable SVG, and keep the explicitly requested PDF MediaBox size.
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["savefig.bbox"] = None
    return module.cm_to_inch


def jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_existing_csv() -> dict[tuple[str, int], dict]:
    with DERIVED.open(newline="") as stream:
        entries = list(csv.DictReader(stream))
    selected = {
        (row["run_id"], int(row["round"])): row
        for row in entries
        if row["pi_distribution"] == "stationary"
        and row["allocation_regime"] in REGIMES
        and row["nx_budget"].lower() in BUDGETS
        and row["run_id"].endswith("__nr")
    }
    if len(selected) != 18 * len(CHECKPOINT_ROUNDS):
        raise ValueError(f"Derived checkpoint table has {len(selected)} selected rows, expected 234")
    return selected


def load_runs() -> tuple[dict[str, dict], list[dict], list[dict], list[Path]]:
    integrity_path = SOURCE / "stage1_integrity.json"
    integrity = json.loads(integrity_path.read_text())
    verified = {x["run_id"] for x in integrity["run_results"] if x["status"] == "verified"}
    derived = load_existing_csv()
    runs: dict[str, dict] = {}
    plotted: list[dict] = []
    totals: list[dict] = []
    inputs = [STYLE, DERIVED, integrity_path, SOURCE / "run_manifest.json",
              SOURCE / "dataset_budget_manifest.csv", SOURCE / "long_horizon_experiment_plan.md"]
    manifest = json.loads((SOURCE / "run_manifest.json").read_text())
    frozen = {run["run_id"]: run for run in manifest["runs"]}
    first_hash = None
    for budget in BUDGETS:
        for regime in REGIMES:
            for epoch in EPOCHS:
                run_id = f"{budget}_stationary_e{epoch}_{regime}__nr"
                if run_id not in verified or run_id not in frozen:
                    raise ValueError(f"Run lacks frozen and verified provenance: {run_id}")
                if frozen[run_id]["seed"] != 42 or frozen[run_id]["rounds"] != 100:
                    raise ValueError(f"Unexpected seed or planned horizon: {run_id}")
                directory = SOURCE / "runs" / run_id
                round_path = directory / "rounds.jsonl"
                validation_path = directory / "validation.jsonl"
                state_path = directory / "state.json"
                rounds = jsonl(round_path)
                evaluations = jsonl(validation_path)
                state = json.loads(state_path.read_text())
                inputs.extend([round_path, validation_path, state_path])
                if [row["round"] for row in rounds] != list(range(1, 101)):
                    raise ValueError(f"Incomplete or out-of-order round trace: {run_id}")
                if tuple(row["round"] for row in evaluations) != CHECKPOINT_ROUNDS:
                    raise ValueError(f"Unexpected checkpoint sequence: {run_id}")
                if state["round"] != 100 or state["checkpoint_sha256"] != evaluations[-1]["checkpoint_sha256"]:
                    raise ValueError(f"R100 committed state and validation disagree: {run_id}")
                if first_hash is None:
                    first_hash = evaluations[0]["checkpoint_sha256"]
                elif first_hash != evaluations[0]["checkpoint_sha256"]:
                    raise ValueError(f"Mismatched initial adapter: {run_id}")
                if any(x["run_id"] != run_id for x in rounds + evaluations):
                    raise ValueError(f"Run ID mismatch: {run_id}")
                if any(x["loss_mode"] != "full_sequence_64_token" or x["seed"] != 777
                       for x in evaluations):
                    raise ValueError(f"Validation protocol mismatch: {run_id}")
                nx_steps = sum(x["clients"]["nx"]["optimizer_steps"] for x in rounds)
                nx_samples = sum(x["clients"]["nx"]["samples"] for x in rounds)
                nx_tokens = sum(x["clients"]["nx"]["input_tokens"] for x in rounds)
                pi_steps = sum(x["clients"]["pi"]["optimizer_steps"] for x in rounds)
                if any(x["clients"]["nx"]["optimizer_steps"] != x["clients"]["nx"]["samples"]
                       for x in rounds):
                    raise ValueError(f"Executed steps and samples disagree at B=1: {run_id}")
                if regime == "equal_work":
                    if nx_steps != 400 or pi_steps != 400 or any(
                        x["clients"]["nx"]["optimizer_steps"] != 4
                        or x["clients"]["nx"]["completed_passes"] != epoch
                        or x["normal_nx_samples"] != 4 // epoch
                        or x["normal_nx_steps_cap"] != 4
                        for x in rounds
                    ):
                        raise ValueError(f"Equal-work/epoch execution invariant fails: {run_id}")
                elif any(x["clients"]["nx"]["optimizer_steps"] <= 0 for x in rounds):
                    raise ValueError(f"Unexpected missing NX work in window-filling: {run_id}")
                run_info = {
                    "run_id": run_id,
                    "budget": "DFull" if budget == "dfull" else budget.upper(),
                    "regime": regime,
                    "epoch": epoch,
                    "seed": 42,
                    "nx_optimizer_steps_r1_r100": nx_steps,
                    "nx_executed_samples_r1_r100": nx_samples,
                    "nx_input_tokens_r1_r100": nx_tokens,
                    "pi_optimizer_steps_r1_r100": pi_steps,
                    "validation_checkpoints": len(evaluations),
                    "r0_adapter_sha256": evaluations[0]["checkpoint_sha256"],
                    "r100_adapter_sha256": evaluations[-1]["checkpoint_sha256"],
                    "r100_balanced_validation_nll": evaluations[-1]["balanced_client_loss"],
                    "allocator_sources": sorted({x["allocator_source"] for x in rounds}),
                    "round_log": str(round_path),
                    "validation_log": str(validation_path),
                    "state_file": str(state_path),
                }
                totals.append(run_info)
                runs[run_id] = {"info": run_info, "rounds": rounds, "evaluations": evaluations}
                for checkpoint in evaluations:
                    round_number = checkpoint["round"]
                    derived_row = derived.get((run_id, round_number))
                    if derived_row is None:
                        raise ValueError(f"Checkpoint absent from derived table: {run_id}, R{round_number}")
                    measured = checkpoint["balanced_client_loss"]
                    if not math.isclose(measured, float(derived_row["balanced_client_loss"]), abs_tol=1e-12):
                        raise ValueError(f"Raw/derived balanced loss mismatch: {run_id}, R{round_number}")
                    if checkpoint["checkpoint_sha256"] != derived_row["checkpoint_sha256"]:
                        raise ValueError(f"Raw/derived checkpoint mismatch: {run_id}, R{round_number}")
                    if not math.isclose(measured,
                                        (checkpoint["pi_loss"] + checkpoint["nx_loss"]) / 2,
                                        abs_tol=1e-10):
                        raise ValueError(f"Balanced-client definition mismatch: {run_id}, R{round_number}")
                    plotted.append({
                        "figure": "plot_epoch_data_availability" if budget == "d128"
                                  else f"figS_epoch_{budget}",
                        "panel": "a_equal_work" if regime == "equal_work" else "b_window_filling",
                        "budget": "DFull" if budget == "dfull" else budget.upper(),
                        "regime": regime,
                        "epoch": epoch,
                        "run_id": run_id,
                        "training_seed": 42,
                        "evaluation_seed": checkpoint["seed"],
                        "round": round_number,
                        "balanced_validation_nll": measured,
                        "pi_validation_nll": checkpoint["pi_loss"],
                        "nx_validation_nll": checkpoint["nx_loss"],
                        "loss_mode": checkpoint["loss_mode"],
                        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                        "nx_optimizer_steps_r1_r100": nx_steps,
                        "nx_executed_samples_r1_r100": nx_samples,
                        "validation_log": str(validation_path),
                    })
    if len(plotted) != 234 or len(totals) != 18:
        raise ValueError("Incorrect source row counts")
    return runs, plotted, totals, inputs


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_figure(budget: str, plotted: list[dict], cm_to_inch):
    fig, axes = plt.subplots(1, 2, figsize=cm_to_inch(FIGURE_WIDTH_CM, FIGURE_HEIGHT_CM))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.25, top=0.76, wspace=0.25)
    line_handles = []
    for ax, regime, title in zip(axes, REGIMES, ("(a) Equal work", "(b) Window filling")):
        subset = [x for x in plotted if x["budget"].lower() == budget and x["regime"] == regime]
        if len(subset) != 3 * len(CHECKPOINT_ROUNDS):
            raise ValueError(f"Missing plotted checkpoint rows: {budget}, {regime}")
        for epoch in EPOCHS:
            trajectory = sorted((x for x in subset if x["epoch"] == epoch), key=lambda x: x["round"])
            line, = ax.plot(
                [x["round"] for x in trajectory],
                [x["balanced_validation_nll"] for x in trajectory],
                color="black", linewidth=1.4, markersize=3.8,
                markerfacecolor="white", markeredgecolor="black",
                markeredgewidth=0.75, label=f"E{epoch}", **STYLES[epoch],
            )
            if regime == "equal_work":
                line_handles.append(line)
        values = [x["balanced_validation_nll"] for x in subset]
        span = max(values) - min(values)
        ax.set_ylim(min(values) - 0.07 * span, max(values) + 0.07 * span)
        # Leave half a marker of white space at R0/R100 so measured endpoints
        # are not visually clipped by the axes boundary.
        ax.set_xlim(-2, 102)
        ax.set_xticks((0, 20, 40, 60, 80, 100))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(title, loc="left", fontsize=10, pad=3)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, color="black", alpha=0.25)
        ax.tick_params(width=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.supxlabel("FL round", fontsize=10, y=0.06)
    fig.supylabel("Balanced validation NLL", fontsize=10, x=0.01)
    legend = fig.legend(line_handles, [f"E{x}" for x in EPOCHS], loc="upper center",
                        bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False)
    return fig, axes, legend


def export(fig, name: str, folder: Path) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = {ext: folder / f"{name}.{ext}" for ext in ("pdf", "svg", "png")}
    fig.savefig(paths["pdf"], bbox_inches=None)
    fig.savefig(paths["svg"], bbox_inches=None)
    fig.savefig(paths["png"], bbox_inches=None, dpi=600)
    return paths


def pdf_dimensions_cm(path: Path) -> tuple[float, float]:
    output = subprocess.check_output(["pdfinfo", str(path)], text=True)
    for line in output.splitlines():
        if line.startswith("Page size:"):
            parts = line.split()
            return float(parts[2]) / 72 * 2.54, float(parts[4]) / 72 * 2.54
    raise RuntimeError(f"PDF page size missing: {path}")


def main() -> None:
    cm_to_inch = load_style()
    runs, plotted, totals, inputs = load_runs()
    save_csv(OUTPUT / "verification" / "fig7_plot_values.csv", plotted)
    save_csv(OUTPUT / "verification" / "fig7_executed_work.csv",
             [{key: value for key, value in row.items() if key != "allocator_sources"}
              for row in totals])
    save_csv(OUTPUT / "verification" / "fig7_r100_endpoints.csv", [
        {
            "budget": row["budget"],
            "regime": row["regime"],
            "epoch": row["epoch"],
            "seed": row["seed"],
            "run_id": row["run_id"],
            "r100_balanced_validation_nll": row["r100_balanced_validation_nll"],
            "nx_optimizer_steps_r1_r100": row["nx_optimizer_steps_r1_r100"],
            "nx_executed_samples_r1_r100": row["nx_executed_samples_r1_r100"],
            "nx_input_tokens_r1_r100": row["nx_input_tokens_r1_r100"],
            "r100_adapter_sha256": row["r100_adapter_sha256"],
        }
        for row in totals
    ])
    figures = []
    for budget in BUDGETS:
        name = "plot_epoch_data_availability" if budget == "d128" else f"figS_epoch_{budget}"
        folder = OUTPUT / ("main" if budget == "d128" else "supplementary")
        fig, axes, legend = build_figure(budget, plotted, cm_to_inch)
        fig.canvas.draw()
        if not math.isclose(fig.get_size_inches()[0] * 2.54, FIGURE_WIDTH_CM, abs_tol=0.001):
            raise ValueError("Unexpected figure width")
        if not math.isclose(fig.get_size_inches()[1] * 2.54, FIGURE_HEIGHT_CM, abs_tol=0.001):
            raise ValueError("Unexpected figure height")
        if not all(math.isclose(t.get_fontsize(), 9, abs_tol=0.01)
                   for ax in axes for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels())):
            raise ValueError("Tick font differs from the EEFL reference")
        if not all(math.isclose(t.get_fontsize(), 8.5, abs_tol=0.01) for t in legend.get_texts()):
            raise ValueError("Legend font differs from the EEFL reference")
        paths = export(fig, name, folder)
        width, height = pdf_dimensions_cm(paths["pdf"])
        if abs(width - FIGURE_WIDTH_CM) > 0.01 or abs(height - FIGURE_HEIGHT_CM) > 0.01:
            raise ValueError(f"Incorrect actual PDF MediaBox: {width:.3f} × {height:.3f} cm")
        svg = paths["svg"].read_text()
        if "<text" not in svg or 'Balanced validation NLL' not in svg:
            raise ValueError("SVG text was converted to outlines or labels are missing")
        from PIL import Image
        with Image.open(paths["png"]) as image:
            dpi = image.info.get("dpi")
            if dpi is None or min(dpi) < 599:
                raise ValueError(f"PNG export is not 600 dpi: {dpi}")
            pixels = image.size
        font_path = font_manager.findfont("Times New Roman")
        figures.append({
            "figure": name,
            "budget": budget.upper(),
            "formats": {ext: str(path) for ext, path in paths.items()},
            "intended_size_cm": [FIGURE_WIDTH_CM, FIGURE_HEIGHT_CM],
            "actual_pdf_size_cm": [width, height],
            "png_pixels": pixels,
            "png_dpi": list(dpi),
            "font_file": font_path,
            "axis_font_pt": 10,
            "tick_font_pt": 9,
            "legend_font_pt": 8.5,
            "line_width_pt": 1.4,
            "panel_y_axes": "independently scaled; numeric ticks shown",
            "checkpoint_points": 3 * 2 * len(CHECKPOINT_ROUNDS),
        })
        plt.close(fig)
    source_manifest = {
        "scope": "Fig. 7 D128 plus supplementary D512/DFull stationary epochs",
        "measurement": "actual RTX LoRA learning and held-out validation; no edge cost shown",
        "training_seed": 42,
        "evaluation_seed": 777,
        "model": "Qwen2.5-0.5B-Instruct",
        "optimizer": "torch.optim.Adam, lr=5e-5",
        "metric": "balanced_client_loss = (Pi-associated full-sequence 64-token NLL + NX-associated full-sequence 64-token NLL)/2",
        "initial_adapter_sha256": totals[0]["r0_adapter_sha256"],
        "run_ids": [row["run_id"] for row in totals],
        "inputs_sha256": {str(path): sha256(path) for path in inputs},
        "plot_script": str(SCRIPT),
        "plot_script_sha256": sha256(SCRIPT),
        "plots": figures,
        "caveat": "Equal work matches actual optimizer steps (four per client per round), not exact input-token counts or edge time. E2/E4 window-filling edge feasibility uses an unvalidated E1 timing proxy; these are RTX training variants, not observed online edge allocation.",
    }
    manifest_path = OUTPUT / "verification" / "fig7_source_manifest.json"
    manifest_path.write_text(json.dumps(source_manifest, indent=2) + "\n")
    audit = {
        "status": "PASS",
        "runs_checked": len(runs),
        "round_records_checked": 100 * len(runs),
        "validation_records_plotted": len(plotted),
        "checkpoints_per_run": len(CHECKPOINT_ROUNDS),
        "equal_work_nx_step_totals": {f"E{e}": next(x["nx_optimizer_steps_r1_r100"] for x in totals
                                                       if x["budget"] == "D128" and x["regime"] == "equal_work" and x["epoch"] == e)
                                      for e in EPOCHS},
        "window_filling_nx_step_totals": {f"E{e}": next(x["nx_optimizer_steps_r1_r100"] for x in totals
                                                          if x["budget"] == "D128" and x["regime"] == "heterogeneous" and x["epoch"] == e)
                                         for e in EPOCHS},
        "r100_nll": {x["run_id"]: x["r100_balanced_validation_nll"] for x in totals},
        "figures": figures,
        "validation": "Exact raw/derived checkpoint values and hashes, common initial adapter, 100-round traces, executed optimizer steps, 13 checkpoint counts, Figure dimensions, font settings, editable SVG text and 600-dpi PNG were checked.",
    }
    (OUTPUT / "verification" / "fig7_validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    save_csv(OUTPUT / "verification" / "fig7_size_font_audit.csv", [
        {
            "figure": entry["figure"],
            "intended_width_cm": FIGURE_WIDTH_CM,
            "intended_height_cm": FIGURE_HEIGHT_CM,
            "actual_pdf_width_cm": entry["actual_pdf_size_cm"][0],
            "actual_pdf_height_cm": entry["actual_pdf_size_cm"][1],
            "axis_label_font_pt": entry["axis_font_pt"],
            "tick_font_pt": entry["tick_font_pt"],
            "legend_font_pt": entry["legend_font_pt"],
            "line_width_pt": entry["line_width_pt"],
            "font_file": entry["font_file"],
            "png_dpi": round(min(entry["png_dpi"])),
            "vector_text": "Times New Roman PDF Type-42; SVG editable text",
            "status": "PASS",
        }
        for entry in figures
    ])


if __name__ == "__main__":
    main()
