#!/usr/bin/env python3
"""Rebuild Fig. 9a/b with a white background and the measured Baseline.

The reviewed hatched-bar layout is retained.  The first bars in the attached
older images were Fixed-Hetero *measurements*, so this script replaces those
measurements with policy U from the same Nano Super + Orin NX campaign instead
of merely changing the tick label.  No training or trace projection is run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev


OUT = Path(__file__).resolve().parent
SOURCE = Path(
    "/Users/jang-won-u/Documents/Codex/2026-10-10/new-chat/outputs/"
    "learning_efficiency_confirmation"
)
ROUNDS = SOURCE / "per_round_hardware_results.csv"
SUMMARY = SOURCE / "measurements/policy_summary.csv"
QC = SOURCE / "analysis_qc.json"
POLICY_AUDIT = SOURCE / "policy_and_seed_audit.md"
SEEDS = (43, 44, 45)
POLICIES = ("U", "P", "C")
NAMES = {"U": "Baseline", "P": "PyramidFL-style", "C": "LoRACast"}
ACTUAL_NAMES = {
    "U": "uniform FedAvg", "P": "bounded PyramidFL-style adaptive local training",
    "C": "LoRACast-NR",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-6):
        raise AssertionError(f"{label}: {actual} != {expected}")


def derive() -> tuple[list[dict], list[dict]]:
    assert json.loads(QC.read_text())["status"] == "verified"
    policy_audit = POLICY_AUDIT.read_text()
    assert "U is the existing uniform FedAvg" in policy_audit
    assert "F is the existing fixed heterogeneous quota" in policy_audit
    summary = read_csv(SUMMARY)
    rounds = read_csv(ROUNDS)
    by_summary = {
        (int(r["scenario"].split("_")[0].removeprefix("seed")), r["policy"]): r
        for r in summary
    }
    by_round = {(int(r["seed"]), r["policy"], int(r["round"])): r for r in rounds}
    assert len(summary) == 12 and len(rounds) == 144
    assert len(by_summary) == 12 and len(by_round) == 144

    seed_values = []
    for policy in POLICIES:
        for seed in SEEDS:
            entries = [by_round[(seed, policy, round_no)] for round_no in range(1, 13)]
            assert [r["phase"] for r in entries] == (
                ["normal"] * 4 + ["contention"] * 4 + ["recovery"] * 4
            )
            run_ids = {r["run_id"] for r in entries}
            assert len(run_ids) == 1
            tokens = sum(int(r["aggregate_input_tokens"]) for r in entries)
            active_s = sum(float(r["server_active_round_s"]) for r in entries)
            energy_j = sum(float(r["client_device_active_energy_j"]) for r in entries)
            assert tokens > 0 and active_s > 0 and energy_j > 0
            recorded = by_summary[(seed, policy)]
            assert int(recorded["rounds"]) == 12
            close(tokens, float(recorded["aggregate_input_tokens"]), f"tokens {seed}/{policy}")
            close(active_s, float(recorded["server_active_time_s"]), f"time {seed}/{policy}")
            close(energy_j, float(recorded["client_device_active_energy_j"]), f"energy {seed}/{policy}")
            close(tokens / active_s, float(recorded["tokens_per_active_s"]), f"rate {seed}/{policy}")
            close(1000 * energy_j / tokens, float(recorded["j_per_1k_tokens"]), f"energy intensity {seed}/{policy}")
            seed_values.append({
                "policy_code": policy, "policy": NAMES[policy], "seed": seed,
                "run_id": next(iter(run_ids)), "rounds": 12,
                "executed_input_tokens": tokens,
                "server_active_s": active_s,
                "measured_client_energy_j": energy_j,
                "throughput_tokens_per_h": 3600 * tokens / active_s,
                "energy_j_per_1k_tokens": 1000 * energy_j / tokens,
            })
    plotted = []
    for policy in POLICIES:
        rows = [r for r in seed_values if r["policy_code"] == policy]
        assert len(rows) == 3
        plotted.append({
            "policy_code": policy, "policy": NAMES[policy], "n_seeds": len(rows),
            "throughput_tokens_per_h_mean": mean(r["throughput_tokens_per_h"] for r in rows),
            "throughput_tokens_per_h_sample_sd": stdev(r["throughput_tokens_per_h"] for r in rows),
            "energy_j_per_1k_tokens_mean": mean(r["energy_j_per_1k_tokens"] for r in rows),
            "energy_j_per_1k_tokens_sample_sd": stdev(r["energy_j_per_1k_tokens"] for r in rows),
        })
    return seed_values, plotted


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw(rows: list[dict], metric: str, stem: str, ylabel: str,
         lower: int, upper: int, tick_step: int, value_scale: float,
         suffix: str, decimals: int) -> None:
    """Render the reviewed hatched-bar style on Fig. 10's white background."""
    assert [r["policy_code"] for r in rows] == list(POLICIES)
    values = [float(r[f"{metric}_mean"]) for r in rows]
    assert lower > 0 and all(lower < v < upper for v in values)
    x0, x1, y0, y1 = 155, 842, 385, 58
    centers, width = (270, 498, 726), 135
    to_y = lambda v: y0 - (v - lower) / (upper - lower) * (y0 - y1)
    # Match the Nano Super bar in the supplied Fig. 8(a) at its PNG size:
    # #cbcbcb fill, rising #343434 diagonal hatch, and #343434 outline.
    # The source Fig. 8 PNG has approximately 44 px hatch pitch and 5 px
    # hatch strokes; this 2031 px export uses 13.3-unit tiles and 2.1-unit
    # strokes for the same visible pattern.
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="8.6cm" height="5.2cm" viewBox="0 0 860 520">',
        # Explicit white page and plot backgrounds avoid transparency/viewer differences.
        '<rect x="0" y="0" width="860" height="520" fill="#FFFFFF"/>',
        f'<rect x="{x0}" y="{y1}" width="{x1-x0}" height="{y0-y1}" fill="#FFFFFF"/>',
        '<defs><pattern id="barHatch" width="13.3" height="13.3" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(45)"><rect width="13.3" height="13.3" fill="#CBCBCB"/>'
        '<path d="M 0 0 V 13.3" stroke="#343434" stroke-width="2.1"/></pattern></defs>',
        '<g font-family="Times New Roman, Times, DejaVu Serif" fill="#111111">',
    ]
    for tick in range(lower, upper + 1, tick_step):
        y = to_y(tick)
        if tick != lower:
            parts.append(f'<line x1="{x0}" y1="{y:.3f}" x2="{x1}" y2="{y:.3f}" '
                         'stroke="#DEDEDE" stroke-width="1.2"/>')
        parts.append(f'<line x1="{x0-7}" y1="{y:.3f}" x2="{x0}" y2="{y:.3f}" '
                     'stroke="#333333" stroke-width="2.8"/>')
        parts.append(f'<text x="{x0-17}" y="{y+10:.3f}" font-size="29" text-anchor="end">'
                     f'{tick/value_scale:.0f}{suffix}</text>')
    parts.extend([
        f'<line x1="{x0}" y1="{y1}" x2="{x0}" y2="{y0}" stroke="#333333" stroke-width="2.8"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#333333" stroke-width="2.8"/>',
    ])
    labels = (("Baseline", ""), ("PyramidFL-", "style"), ("LoRACast", ""))
    for center, value, label in zip(centers, values, labels):
        top = to_y(value)
        parts.append(f'<rect x="{center-width/2:.3f}" y="{top:.3f}" width="{width}" '
                     f'height="{y0-top:.3f}" fill="url(#barHatch)" stroke="#343434" stroke-width="2.46"/>')
        parts.append(f'<text x="{center}" y="{top-13:.3f}" font-size="29" text-anchor="middle">'
                     f'{value/value_scale:.{decimals}f}{suffix}</text>')
        parts.append(f'<line x1="{center}" y1="{y0}" x2="{center}" y2="{y0+7}" '
                     'stroke="#333333" stroke-width="2.8"/>')
        if label[1]:
            parts.append(f'<text x="{center}" y="{y0+38}" font-size="28" text-anchor="middle">{label[0]}</text>')
            parts.append(f'<text x="{center}" y="{y0+70}" font-size="28" text-anchor="middle">{label[1]}</text>')
        else:
            parts.append(f'<text x="{center}" y="{y0+54}" font-size="28" text-anchor="middle">{label[0]}</text>')
    mid = (y0 + y1) / 2
    parts.append(f'<text x="40" y="{mid:.3f}" transform="rotate(-90 40 {mid:.3f})" '
                 f'font-size="31" text-anchor="middle">{ylabel}</text>')
    parts.extend(['</g>', '</svg>'])
    svg = OUT / f"{stem}.svg"
    svg.write_text("\n".join(parts) + "\n", encoding="utf-8")
    subprocess.run(["rsvg-convert", "-f", "pdf", "-d", "72", "-p", "72",
                    "-o", str(OUT / f"{stem}.pdf"), str(svg)], check=True)
    subprocess.run(["rsvg-convert", "-f", "png", "-d", "600", "-p", "600",
                    "-o", str(OUT / f"{stem}.png"), str(svg)], check=True)
    subprocess.run(["sips", "-s", "dpiWidth", "600", "-s", "dpiHeight", "600",
                    str(OUT / f"{stem}.png")], check=True, capture_output=True)


def main(*, render_only: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if render_only:
        # Preserve the already verified source-derived coordinates when the
        # external experiment volume is temporarily unavailable.
        plotted = read_csv(OUT / "fig9_plotted_values.csv")
        assert [r["policy_code"] for r in plotted] == list(POLICIES)
        manifest = json.loads((OUT / "fig9_source_manifest.json").read_text())
    else:
        seed_values, plotted = derive()
        write_csv(OUT / "fig9_seed_values.csv", seed_values)
        write_csv(OUT / "fig9_plotted_values.csv", plotted)
        manifest = {
            "campaign": "Nano Super + Orin NX, measured 12-round CPU-contention campaign",
            "rounds": "R1–4 normal, R5–8 contention, R9–12 recovery",
            "seeds": list(SEEDS),
            "display_policies": [NAMES[p] for p in POLICIES],
            "actual_policy_C": ACTUAL_NAMES["C"],
            "measurement": "per-seed tokens/active server time and Nano+NX measured client energy/tokens; mean over seeds",
            "source_sha256": {str(p): sha256(p) for p in (ROUNDS, SUMMARY, QC, POLICY_AUDIT)},
            "attached_fig9_first_bar": "Fixed-Hetero (F); values not reused for Baseline (U)",
        }
    draw(plotted, "throughput_tokens_per_h", "fig9a_contention_training_throughput_baseline_white",
         "Tokens/h", 720000, 860000, 20000, 1000, "K", 1)
    draw(plotted, "energy_j_per_1k_tokens", "fig9b_contention_client_energy_baseline_white",
         "J/1K Tokens", 105, 113, 1, 1, "", 1)
    manifest.update({
        "changes": [
            "white page and axes", "first bar U actual data and Baseline tick",
            "axes expanded to contain U", "all Nano bars use the Fig. 8(a) gray diagonal hatch",
        ],
        "hatch_reference": "attached Fig. 8(a): #cbcbcb Nano fill, rising #343434 diagonals; Pi is white and unhatched",
        "pi5_status": "no matched completed Pi5+NX Fig. 9 physical-contention policy runs; no Pi bars plotted",
        "plotted_files_sha256": {
            str(p.name): sha256(p) for p in OUT.glob("fig9*.pdf")
        },
    })
    (OUT / "fig9_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["--render-only"]):
        raise SystemExit("usage: plot_contention_comparison.py [--render-only]")
    main(render_only=sys.argv[1:] == ["--render-only"])
