#!/usr/bin/env python3
"""Render paper Table 1 from verified Nano+NX seed-level measurements.

This reads the 12 seed-policy records generated from the Nano+NX confirmation
campaign. It does not train, score checkpoints, or substitute a cost estimate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
SEED_FILE = ROOT / "verification/fig6_seed_run_summary.csv"
FULL_SOURCE = ROOT.parent / "exposure_policy/nano_nx_independent_primary_loss.csv"
OLD_TABLE = ROOT / "tables/table1_real_device_performance.csv"
OUT_CSV = ROOT / "tables/table1_mean_sd_paper.csv"
OUT_TEX = ROOT / "tables/table1_mean_sd_paper.tex"
OUT_AUDIT = ROOT / "verification/table1_mean_sd_validation.json"
METHODS = ("FedAvg", "Fixed-Hetero", "PyramidFL-style", "LoRACast")
METRICS = (
    ("R12_validation_NLL", "R12_validation_NLL", 5),
    ("total_active_time_min", "total_active_time_min", 2),
    ("total_measured_client_energy_kJ", "total_measured_client_energy_kJ", 2),
    ("tokens_per_active_s", "tokens_per_active_s", 2),
    ("J_per_1K_input_tokens", "J_per_1K_input_tokens", 2),
)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> None:
    seed_rows = read(SEED_FILE)
    previous = {row["method"]: row for row in read(OLD_TABLE)}
    if len(seed_rows) != 12 or len(previous) != 4:
        raise AssertionError("Expected exactly 12 seed runs and four previous table rows")
    pairs = {(row["method"], int(row["seed"])) for row in seed_rows}
    if pairs != {(method, seed) for method in METHODS for seed in (43, 44, 45)}:
        raise AssertionError("The paired seed/policy matrix is incomplete or duplicated")
    if any(not row["run_id"].startswith("nano-nx-learning-") for row in seed_rows):
        raise AssertionError("Unexpected experiment run ID")

    full_rows = read(FULL_SOURCE)
    full_keys = {(row["policy"], int(row["seed"]), int(row["round"])) for row in full_rows}
    if len(full_rows) != 156 or full_keys != {
        (policy, seed, round_number)
        for policy in ("U", "F", "P", "C")
        for seed in (43, 44, 45)
        for round_number in range(13)
    }:
        raise AssertionError("Complete 4-policy × 3-seed × R0–R12 source matrix required")
    source_r12 = {(row["policy"], int(row["seed"])): row
                  for row in full_rows if int(row["round"]) == 12}
    for row in seed_rows:
        raw = source_r12[(row["policy_code"], int(row["seed"]))]
        if raw["run_id"] != row["run_id"] or not raw["checkpoint_sha256"]:
            raise AssertionError("R12 run/checkpoint identity differs from seed summary")
        seconds = float(raw["cumulative_measured_active_s"])
        joules = float(raw["cumulative_measured_client_j"])
        tokens = float(raw["cumulative_executed_input_tokens"])
        matched = {
            "R12_validation_NLL": float(raw["full_sequence_nll_bf16_l64"]),
            "total_active_time_min": seconds / 60,
            "total_measured_client_energy_kJ": joules / 1000,
            "tokens_per_active_s": tokens / seconds,
            "J_per_1K_input_tokens": 1000 * joules / tokens,
        }
        for field, value in matched.items():
            if not math.isclose(value, float(row[field]), abs_tol=1e-8, rel_tol=0):
                raise AssertionError(f"R12 source/summary mismatch: {row['run_id']} {field}")

    formatted = []
    numerical = []
    for method in METHODS:
        group = [row for row in seed_rows if row["method"] == method]
        output = {"Policy": method, "Independent seeds": "43, 44, 45"}
        result = {"method": method, "n": 3}
        for field, old_prefix, decimals in METRICS:
            values = [float(row[field]) for row in group]
            avg, sd = mean(values), stdev(values)
            result[field + "_mean"] = avg
            result[field + "_sample_sd"] = sd
            tolerance = 0.5 * 10 ** (-decimals) + 1e-9
            for suffix, calculated in (("mean", avg), ("SD", sd)):
                listed = float(previous[method][old_prefix + "_" + suffix])
                if not math.isclose(calculated, listed, abs_tol=tolerance, rel_tol=0):
                    raise AssertionError(f"Old Table 1 mismatch: {method}/{field}/{suffix}")
            output[field] = f"{avg:.{decimals}f} ± {sd:.{decimals}f}"
        formatted.append(output)
        numerical.append(result)

    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(formatted[0]))
        writer.writeheader()
        writer.writerows(formatted)

    def tex_value(text: str) -> str:
        return text.replace(" ± ", r" $\pm$ ")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Real-device learning efficiency over independent seeds 43--45. Values are mean $\pm$ sample standard deviation across seed-level FL runs. Validation NLL is scored on RTX from the device-generated R12 checkpoints; active time and two-client VDD\_IN energy are measured on Nano Super and Orin NX during server-active FL rounds.}",
        r"\label{tab:real_device_efficiency}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Policy & R12 NLL & Active time (min) & Client energy (kJ) & Input tokens/s & J/1K tokens \\",
        r"\midrule",
    ]
    for row in formatted:
        cols = [row["Policy"], *(tex_value(row[field]) for field, _, _ in METRICS)]
        lines.append(" & ".join(cols) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}"])
    OUT_TEX.write_text("\n".join(lines) + "\n")
    OUT_AUDIT.write_text(json.dumps({
        "status": "PASS",
        "independent_unit": "seed-level FL run",
        "seed_ids": [43, 44, 45],
        "policy_order": list(METHODS),
        "source": str(SEED_FILE),
        "source_sha256": digest(SEED_FILE),
        "crosschecked_previous_table": str(OLD_TABLE),
        "previous_table_sha256": digest(OLD_TABLE),
        "complete_R0_R12_source": str(FULL_SOURCE),
        "complete_R0_R12_source_sha256": digest(FULL_SOURCE),
        "complete_R0_R12_rows": len(full_rows),
        "source_note": "R12 seed summary and all five table metrics were crosschecked against the intact 156-row primary Nano+NX loss/cost table; /Documents campaign copies are iCloud dataless.",
        "sd_definition": "statistics.stdev: sample SD over three independent seeds",
        "R12_numerical_values": numerical,
        "output_csv": str(OUT_CSV),
        "output_tex": str(OUT_TEX),
    }, indent=2) + "\n")
    print(f"Wrote {OUT_CSV.name} and {OUT_TEX.name}: 4 policies, seeds 43--45")


if __name__ == "__main__":
    run()
