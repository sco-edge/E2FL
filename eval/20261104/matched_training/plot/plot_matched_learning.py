#!/usr/bin/env python3
"""Gate and plot matched Nano Super + NX R0–R100 Qwen validation NLL.

This script does not train or evaluate a model. It accepts three *new* run
directories only after an independent full-chain integrity report passes. A
failed gate produces no figure. No historical paths or numerical values are
embedded in the plotter.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

POLICIES = ("Baseline", "PyramidFL-style", "LoRACast")
STYLES = {
    "Baseline": ("#7F7F7F", "-", "o", 1.2),
    "PyramidFL-style": ("#CC79A7", "-.", "D", 1.2),
    "LoRACast": ("#0072B2", "-", "^", 1.6),
}
POINT_ROUNDS = list(range(0, 101, 10))
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SCORER_FIELDS = ("loss_mode", "cap_per_view", "scored_examples", "seed",
                 "nano_partition_origin")
ATTESTED_HASH_FIELDS = ("rounds_sha256", "validation_sha256", "state_sha256")
SHARED_HASH_FIELDS = ("initial_checkpoint_sha256", "partition_manifest_sha256",
                      "validation_manifest_sha256", "schedule_sha256")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path: Path) -> list[dict]:
    require(path.is_file(), f"Missing input: {path}")
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                require(isinstance(row, dict), f"{path}:{line_number}: expected object")
                rows.append(row)
    return rows


def checked_sha(value: object, description: str) -> str:
    require(isinstance(value, str) and HEX64.fullmatch(value) is not None,
            f"Invalid SHA-256 for {description}")
    return value


def check_attestation(report: dict, policy: str, path: Path) -> dict:
    runs = report.get("runs", {})
    require(isinstance(runs, dict) and isinstance(runs.get(policy), dict),
            f"Integrity report lacks {policy}")
    entry = runs[policy]
    require(entry.get("status") == "PASS" and entry.get("checkpoint_chain") == "PASS",
            f"{policy}: full checkpoint chain is not verified")
    require(entry.get("round") == 100 and entry.get("round_count") == 100,
            f"{policy}: report does not attest 100 committed rounds")
    require(entry.get("checkpoint_count") == 101,
            f"{policy}: report does not attest R0–R100 checkpoints")
    require(entry.get("run_id") and isinstance(entry["run_id"], str),
            f"{policy}: missing attested run ID")
    for filename, field in (("rounds.jsonl", "rounds_sha256"),
                            ("validation.jsonl", "validation_sha256"),
                            ("state.json", "state_sha256")):
        expected = checked_sha(entry.get(field), f"{policy}/{field}")
        require(digest(path / filename) == expected,
                f"{policy}: {filename} differs from the verified report")
    shared = report.get("shared", {})
    require(isinstance(shared, dict), "Missing shared protocol attestation")
    for field in SHARED_HASH_FIELDS:
        expected = checked_sha(shared.get(field), f"shared/{field}")
        require(checked_sha(entry.get(field), f"{policy}/{field}") == expected,
                f"{policy}: {field} differs from the shared protocol")
    return entry


def check_run(policy: str, path: Path, entry: dict) -> tuple[list[dict], list[dict], dict]:
    rounds = jsonl(path / "rounds.jsonl")
    evaluations = jsonl(path / "validation.jsonl")
    state = json.loads((path / "state.json").read_text())
    require(len(rounds) == 100 and [x.get("round") for x in rounds] == list(range(1, 101)),
            f"{policy}: rounds.jsonl must contain exactly R1–R100 in order")
    require(len(evaluations) == 11 and
            [x.get("round") for x in evaluations] == POINT_ROUNDS,
            f"{policy}: validation.jsonl must contain actual R0,R10,…,R100 rows")
    require(state.get("round") == 100 and
            state.get("checkpoint_sha256") == rounds[-1].get("checkpoint_sha256"),
            f"{policy}: final state/checkpoint mismatch")
    run_id = entry["run_id"]
    require(state.get("run_id") == run_id and
            all(row.get("run_id") == run_id for row in rounds + evaluations),
            f"{policy}: mixed or incorrect run IDs")
    r0_sha = checked_sha(evaluations[0].get("checkpoint_sha256"), f"{policy}/R0")
    require(r0_sha == entry["initial_checkpoint_sha256"],
            f"{policy}: R0 checkpoint differs from shared initialization")
    by_round = {0: r0_sha}
    previous_sha = r0_sha
    for row in rounds:
        number = row["round"]
        start_sha = checked_sha(row.get("global_start_checkpoint_sha256"),
                                f"{policy}/R{number} start")
        end_sha = checked_sha(row.get("checkpoint_sha256"),
                              f"{policy}/R{number} checkpoint")
        require(start_sha == previous_sha,
                f"{policy}: broken global checkpoint chain at R{number}")
        by_round[number] = end_sha
        previous_sha = end_sha
    for row in evaluations:
        number = row["round"]
        require(checked_sha(row.get("checkpoint_sha256"),
                            f"{policy}/validation R{number}") == by_round[number],
                f"{policy}: validation R{number} uses the wrong checkpoint")
        require(row.get("measurement") == "RTX_actual_model_validation",
                f"{policy}: validation R{number} is not an actual checkpoint score")
        require(row.get("loss_mode") == "full_sequence_64_token" and
                row.get("cap_per_view") == 512 and row.get("scored_examples") == 1888 and
                row.get("seed") == 777,
                f"{policy}: validation protocol differs at R{number}")
        for field in ("balanced_client_loss", "nano_associated_loss", "nx_loss"):
            value = row.get(field)
            require(isinstance(value, (float, int)) and math.isfinite(value),
                    f"{policy}: invalid {field} at R{number}")
        average = (row["nano_associated_loss"] + row["nx_loss"]) / 2
        require(math.isclose(row["balanced_client_loss"], average,
                             rel_tol=0, abs_tol=1e-10),
                f"{policy}: R{number} is not equally balanced across Nano and NX")
    require(len({tuple(row.get(k) for k in SCORER_FIELDS) for row in evaluations}) == 1,
            f"{policy}: validation protocol changes between checkpoints")
    return rounds, evaluations, state


def configure_style(matplotlib) -> None:
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


def render(validations: dict[str, list[dict]], stage: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_style(matplotlib)
    fig, ax = plt.subplots(figsize=(8.6 / 2.54, 5.5 / 2.54))
    fig.subplots_adjust(left=.205, right=.95, bottom=.23, top=.76)
    visible_losses = []
    for policy in POLICIES:
        points = validations[policy][1:]  # Preserve R0 in CSV; same R10–R100 visual boundary.
        color, linestyle, marker, linewidth = STYLES[policy]
        y = [float(r["balanced_client_loss"]) for r in points]
        visible_losses.extend(y)
        ax.plot([r["round"] for r in points], y, color=color,
                linestyle=linestyle, linewidth=linewidth, marker=marker,
                markersize=3.6, markerfacecolor="white", markeredgewidth=.85,
                clip_on=False, label=policy)
    # Retain the approved range when all new points fit; otherwise expand just
    # enough to avoid silently clipping a genuine matched-run result.
    low, high = 1.495, 1.545
    if min(visible_losses) < low or max(visible_losses) > high:
        span = max(visible_losses) - min(visible_losses)
        pad = max(.002, span * .08)
        low, high = min(visible_losses) - pad, max(visible_losses) + pad
    ax.set_xlim(0, 100)
    ax.set_ylim(low, high)
    if (low, high) == (1.495, 1.545):
        ax.set_yticks((1.50, 1.51, 1.52, 1.53, 1.54))
    ax.set_xticks((0, 20, 40, 60, 80, 100))
    ax.set_xlabel("Round")
    ax.set_ylabel("Balanced validation NLL")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=.8, length=3)
    ax.grid(axis="y", color="#C8C8C8", alpha=.5, linewidth=.4)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.55, .98),
               ncol=2, frameon=False, handlelength=1.6,
               handletextpad=.35, columnspacing=.8, borderaxespad=0)
    outputs = []
    stem = "fig10b_nll_by_round_nano_super_nx"
    for suffix, options in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        destination = stage / f"{stem}.{suffix}"
        fig.savefig(destination, bbox_inches=None, **options)
        outputs.append(destination)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pyramid", type=Path, required=True)
    parser.add_argument("--loracast", type=Path, required=True)
    parser.add_argument("--integrity-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    paths = dict(zip(POLICIES, (args.baseline, args.pyramid, args.loracast)))
    report = json.loads(args.integrity_report.read_text())
    require(report.get("status") == "PASS" and
            report.get("matched_schedule_status") == "PASS",
            "Campaign integrity and matched schedule must both be PASS")
    require(report.get("shared", {}).get("model") == "Qwen2.5-0.5B-Instruct",
            "This Fig. 10b gate requires the same Qwen model")
    for field in SHARED_HASH_FIELDS:
        checked_sha(report["shared"].get(field), f"shared/{field}")
    require(len({path.resolve() for path in paths.values()}) == 3,
            "Three distinct policy trajectories are required")
    validations = {}
    attested_ids = {}
    input_hashes = {}
    for policy, path in paths.items():
        entry = check_attestation(report, policy, path)
        _, evaluations, _ = check_run(policy, path, entry)
        validations[policy] = evaluations
        attested_ids[policy] = entry["run_id"]
        input_hashes[policy] = {filename: digest(path / filename)
                                for filename in ("rounds.jsonl", "validation.jsonl",
                                                 "state.json")}
    require(len(set(attested_ids.values())) == 3,
            "Each method needs an independent training trajectory")
    r0_shas = {r[0]["checkpoint_sha256"] for r in validations.values()}
    r0_losses = [float(r[0]["balanced_client_loss"]) for r in validations.values()]
    require(len(r0_shas) == 1 and max(r0_losses) - min(r0_losses) < 1e-9,
            "Policies do not share the same R0 adapter and validation value")
    protocols = {tuple(r[0].get(k) for k in SCORER_FIELDS)
                 for r in validations.values()}
    require(len(protocols) == 1, "Policies use different NLL scorer/view protocols")
    output_dir = args.output_dir.resolve()
    targets = [output_dir / f"fig10b_nll_by_round_nano_super_nx.{ext}"
               for ext in ("pdf", "svg", "png")]
    targets += [output_dir / "fig10b_nll_plotted_values.csv",
                output_dir / "fig10b_nll_source_manifest.json"]
    require(not any(path.exists() for path in targets),
            "Refusing to overwrite an existing Fig. 10b result")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fig10b_stage_", dir=output_dir) as temporary:
        stage = Path(temporary)
        figures = render(validations, stage)
        values_file = stage / "fig10b_nll_plotted_values.csv"
        with values_file.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("policy", "round",
                "balanced_validation_nll", "nano_associated_nll", "nx_nll",
                "checkpoint_sha256", "visible_in_figure", "source_validation"))
            writer.writeheader()
            for policy in POLICIES:
                for row in validations[policy]:
                    writer.writerow({"policy": policy, "round": row["round"],
                        "balanced_validation_nll": row["balanced_client_loss"],
                        "nano_associated_nll": row["nano_associated_loss"],
                        "nx_nll": row["nx_loss"],
                        "checkpoint_sha256": row["checkpoint_sha256"],
                        "visible_in_figure": row["round"] > 0,
                        "source_validation": str(paths[policy].resolve() / "validation.jsonl")})
        manifest = {
            "status": "PASS",
            "figure": "Fig. 10(b)",
            "model": "Qwen2.5-0.5B-Instruct",
            "device_pair": "Nano Super + Orin NX",
            "training": "Three independent matched R0–R100 RTX LoRA trajectories",
            "comparison_scope": "One seed and one fixed dataset/protocol; R50–R100 matched 4 normal/4 contention/4 recovery schedule attested separately",
            "validation": "Actual RTX checkpoint NLL, full sequence 64 token, cap 512 per view, 1888 scored examples; equal mean of two client-associated losses",
            "plot_boundary": "R10–R100 points shown on Round 0–100 axis; R0 actual evaluation retained in CSV and omitted from the narrow NLL axis",
            "integrity_report": str(args.integrity_report.resolve()),
            "integrity_report_sha256": digest(args.integrity_report),
            "runs": {policy: {"path": str(paths[policy].resolve()),
                              "run_id": attested_ids[policy],
                              "inputs_sha256": input_hashes[policy]}
                     for policy in POLICIES},
            "outputs_sha256": {file.name: digest(file) for file in figures + [values_file]},
        }
        manifest_file = stage / "fig10b_nll_source_manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
        for file in figures + [values_file, manifest_file]:
            os.replace(file, output_dir / file.name)
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir),
                      "r100_nll": {p: validations[p][-1]["balanced_client_loss"]
                                   for p in POLICIES}}, indent=2))


if __name__ == "__main__":
    main()
