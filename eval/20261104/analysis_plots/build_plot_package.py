#!/usr/bin/env python3
"""Assemble the paper-figure provenance and data-column map from verified plots.

This script only reads previously completed experiment records and figure audits.
It does not train, score, or execute any client workload.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import stat
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "verification"
STYLE = Path("/Users/jang-won-u/Documents/GitHub/EEFL/plot/revision/plot_loracast_revision.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(name: str) -> dict:
    return json.loads((VERIFY / name).read_text())


GROUPS = {
    "fig5new": {
        "audit": "fig5_estimator_contention_validation.json",
        "script": "plot_estimator_contention.py",
        "seeds": "July: not recorded; October: independent 43–45",
        "rounds": "July after R2; October R1–R12 (APE after R2)",
        "version": "July Pi 5/NX estimator and October Nano/NX online residual-corrected LoRACast-NR",
        "metric": "device-local compute-time MARE/APE, derived power-times-duration energy-proxy MARE, and executed NX optimizer steps",
        "boundary": "Pi5/NX logged local compute and client-logged power proxy; Nano local compute and externally measured server-active VDD_IN mean-power proxy; actual NX executed steps",
        "value_files": ["fig5a_normal_prediction_values.csv", "fig5a_nano_normal_prediction_values.csv",
                        "fig5a_normal_prediction_mare.csv",
                        "fig5b_contention_prediction_by_seed.csv", "fig5b_contention_prediction_ape.csv",
                        "fig5b_contention_phase_summary.csv", "fig5c_executed_steps_by_seed.csv",
                        "fig5c_runtime_workload_adaptation.csv"],
    },
    "fig5": {
        "audit": "fig5_source_and_plot_validation.json",
        "script": "plot_cost_prediction_adaptation.py",
        "seeds": "Fig. 5(a): July unrecorded, October Nano 42; Fig. 5(b): 43–45",
        "rounds": "Fig. 5(a): July/Nano after R2; Fig. 5(b): R1–R12",
        "version": "July Pi5/NX and October Nano Super logged predictor settings in (a); October Nano/NX LoRACast-NR versus Fixed-Hetero in (b)",
        "metric": "(a) pre-execution local compute-time MARE; (b) actually executed NX optimizer steps",
        "boundary": "July Pi5/NX and October Nano Super device-local compute predictions; separately measured Nano Super/NX execution in panel (b)",
        "value_files": ["fig5a_historical_prediction_values.csv", "fig5a_historical_mare.csv", "fig5a_nano_prediction_values.csv", "fig5a_nano_phase_mare.csv", "fig5a_three_device_mare.csv", "fig5a_plotted_phase_aware_mare.csv", "fig5b_executed_steps_by_seed.csv", "fig5b_executed_steps_mean.csv"],
    },
    "fig5_alt": {
        "audit": "fig5a_alternative_reference_window_validation.json",
        "script": "plot_reference_window.py",
        "seeds": "43",
        "rounds": "R3 pre-dispatch",
        "version": "October Pi5/NX LoRACast-NR online-cost allocator",
        "metric": "logged predicted candidate local compute time and selected workload",
        "boundary": "pre-dispatch prediction, not executed device duration",
        "value_files": ["fig5a_alternative_reference_window_values.csv"],
    },
    "fig6": {
        "audit": "fig6_validation.json",
        "script": "plot_device_learning_efficiency.py",
        "seeds": "43–45 independent",
        "rounds": "observed global checkpoints R0–R12",
        "version": "Nano Super/NX FedAvg, Fixed-Hetero, restricted PyramidFL-style, LoRACast-NR",
        "metric": "balanced held-out BF16 full-sequence 64-token validation NLL",
        "boundary": "server-active time and measured Nano+NX client VDD_IN; RTX checkpoint scoring",
        "value_files": ["fig6_plot_data.csv", "fig6_seed_run_summary.csv"],
    },
    "fig7": {
        "audit": "fig7_source_manifest.json",
        "script": "plot_epoch_data_availability.py",
        "seeds": "42 exploratory training; 777 evaluation",
        "rounds": "100 FL rounds, preserved checkpoints",
        "version": "October RTX NR, D128/D512/DFull, equal-work or heterogeneous epoch E1/E2/E4",
        "metric": "balanced Pi/NX held-out full-sequence 64-token validation NLL",
        "boundary": "actual RTX LoRA learning; no measured edge cost; E2/E4 edge-time feasibility unvalidated",
        "value_files": ["fig7_plot_values.csv", "fig7_executed_work.csv", "fig7_r100_endpoints.csv"],
    },
    "fig8": {
        "audit": "fig8_source_evidence.json",
        "script": "plot_training_budget_policies.py",
        "seeds": "43–45 independent training; 777 evaluation",
        "rounds": "100 FL rounds, 13 observed checkpoints per run",
        "version": "October RTX NR/participation-triggered hard omission/pretraining static Exposure-2",
        "metric": "balanced Pi/NX held-out full-sequence 64-token validation NLL",
        "boundary": "actual RTX LoRA learning and validation; no edge cost plotted",
        "value_files": ["../tables/fig8_plotted_values.csv", "../tables/fig8_mean_values.csv"],
    },
    "s1": {
        "audit": "fig5_s1_pi5_nx_ablation_validation.json",
        "script": "plot_pi5_nx_estimator_ablation.py",
        "seeds": "43–44 independent paired",
        "rounds": "16 FL rounds",
        "version": "Pi5/NX F fixed, A frozen, B last rate, C original LoRACast-NR",
        "metric": "executed input tokens/server-active second; measured active-client J/1K input tokens",
        "boundary": "Pi5+NX active client-device energy; server/AP and intentional cooldown excluded",
        "value_files": ["fig5_s1_pi5_nx_ablation_values.csv"],
    },
    "s2": {
        "audit": "figS2_validation.json",
        "script": "plot_client_workload_variation.py",
        "seeds": "42, single development seed",
        "rounds": "16 rounds; Pi quota blocks 4/3/5/4",
        "version": "Pi5/NX development quota-variation trace",
        "metric": "measured reference compute, NX executed steps, active efficiency, reply barrier",
        "boundary": "measured Pi5/NX run; server observation and runtime overhead may confound",
        "value_files": ["figS2_plot_data.csv"],
    },
    "s3": {
        "audit": "figS3_validation.json",
        "script": "plot_learning_quality_targets.py",
        "seeds": "43–45 independent",
        "rounds": "first observed R5–R12 checkpoint reaching frozen target",
        "version": "Nano Super/NX external policies as Figure 6",
        "metric": "first reached BF16 full-sequence 64-token NLL ≤1.535 or ≤1.533",
        "boundary": "measured server-active time and Nano+NX client VDD_IN among attained seeds only",
        "value_files": ["figS3_target_summary.csv"],
    },
    "s4": {
        "audit": "figS4_source_evidence.json",
        "script": "plot_quota_omission.py",
        "seeds": "43",
        "rounds": "100 FL rounds",
        "version": "October RTX D512 E1 stationary Exposure-2",
        "metric": "actual NX executed samples and cumulative sample exposure",
        "boundary": "RTX virtual-client execution records, not measured edge energy",
        "value_files": ["../tables/figS4_quota_omission_trace.csv"],
    },
    "s5": {
        "audit": "figS5_source_manifest.json",
        "script": "plot_additional_budget_policies.py",
        "seeds": "42 exploratory; data-volume arrival 46",
        "rounds": "100 FL rounds, actual scored checkpoints and executed work",
        "version": "October RTX Exposure-2/4/8, Distributed-2; static cap with fair R41 data-volume update",
        "metric": "balanced Pi/NX full-sequence 64-token NLL; cumulative NX executed samples/exposure",
        "boundary": "actual RTX learning only; D512 Exposure-8 and DFull inactive caps are exact aliases",
        "value_files": ["figS5_plot_values.csv", "figS5_schedule_values.csv", "figS5_arrival_budget_values.csv"],
    },
    "s6": {
        "audit": "figS6_validation.json",
        "script": "plot_historical_estimator_errors.py",
        "seeds": "not recorded in July CSVs",
        "rounds": "post-warmup R3 onward",
        "version": "historical July Pi5/NX MAXN/15W/10W predictor",
        "metric": "absolute relative error of local compute time and warm-up-power-times-duration energy proxy",
        "boundary": "July local compute duration and derived power-times-duration proxy; no direct per-round joule integral",
        "value_files": ["figS6_plot_data.csv"],
    },
}


def group_for(stem: str) -> str:
    if stem in ("fig5a_normal_prediction_mare", "fig5b_contention_prediction_ape",
                "fig5c_runtime_workload_adaptation"):
        return "fig5new"
    if stem.startswith("fig5a_alternative"):
        return "fig5_alt"
    if stem.startswith("fig5"):
        return "fig5"
    if stem.startswith("fig6"):
        return "fig6"
    if stem.startswith("fig7") or stem.startswith("figS_epoch_"):
        return "fig7"
    if stem.startswith("fig8") or stem == "figS_dfull_inactive_exposure_cap":
        return "fig8"
    if stem.startswith("s1_") or stem.startswith("figS1"):
        return "s1"
    if stem.startswith("figS2_") or stem.startswith("s2_"):
        return "s2"
    if stem.startswith("figS3"):
        return "s3"
    if stem.startswith("figS4"):
        return "s4"
    if stem.startswith("figS5"):
        return "s5"
    if stem.startswith("figS6"):
        return "s6"
    raise ValueError(f"Unmapped figure: {stem}")


def sources_from_record(record: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in ("sources_sha256", "inputs_sha256", "input_source_sha256", "input_sha256", "raw_file_sha256", "raw_source_sha256", "source_sha256"):
        if isinstance(record.get(key), dict):
            result.update(record[key])
    for key, value in record.items():
        if not key.endswith("_sha256") or not isinstance(value, str) or len(value) != 64:
            continue
        path_key = key[: -len("_sha256")]
        for candidate_key in (path_key, path_key + "_source", path_key + "_csv", path_key + "_json", path_key + "_jsonl"):
            if isinstance(record.get(candidate_key), str):
                result[record[candidate_key]] = value
                break
    if isinstance(record.get("source_sha256"), str):
        for path_key in ("source", "source_json"):
            if isinstance(record.get(path_key), str):
                result[record[path_key]] = record["source_sha256"]
    if "raw_files" in record:
        for item in record["raw_files"]:
            for name, val in item.items():
                if name.endswith("_sha256") and isinstance(val, str):
                    base = name[: -len("_sha256")]
                    path = next((item.get(k) for k in (base, base + "_jsonl", base + "_json") if item.get(k)), None)
                    if path:
                        result[path] = val
    if "fig8" in str(record.get("derived_file", "")) or "raw_files" in record:
        for path_key, hash_key in (("derived_file", "derived_sha256"), ("endpoint_file", "endpoint_sha256"), ("experiment_manifest", "experiment_manifest_sha256")):
            if record.get(path_key) and record.get(hash_key):
                result[record[path_key]] = record[hash_key]
    return result


def run_ids_for(group: str, record: dict, stem: str) -> list[str]:
    if group == "fig5new":
        if stem.startswith("fig5a_"):
            nano_ids = {row["run_id"] for row in csv.DictReader(
                (VERIFY / "fig5a_nano_normal_prediction_values.csv").open())}
            return ["July-MAXN", "July-15W", "July-10W", *sorted(nano_ids)]
        value_file = ("fig5b_contention_prediction_by_seed.csv" if stem.startswith("fig5b_")
                      else "fig5c_executed_steps_by_seed.csv")
        return sorted({row["run_id"] for row in csv.DictReader((VERIFY / value_file).open())})
    if group == "fig5":
        if stem.startswith("fig5a_historical"):
            return record.get("panel_a", {}).get("run_ids", [])
        return record.get("panel_b", {}).get("run_ids", [])
    if group == "fig6" or group == "s3":
        return sorted({r["run_id"] for r in csv.DictReader((VERIFY / "fig6_plot_data.csv").open()) if r.get("record_type") == "observed_seed_checkpoint"})
    if group == "fig7":
        rows = list(csv.DictReader((VERIFY / "fig7_plot_values.csv").open()))
        budget = "D128" if stem.startswith("fig7") else "D512" if "d512" in stem.lower() else "DFull"
        regime = "equal_work" if "equal_work" in stem.lower() or stem.startswith("fig7a") else "heterogeneous" if "window" in stem.lower() or stem.startswith("fig7b") else None
        return sorted({r["run_id"] for r in rows if r.get("budget", "").lower() == budget.lower() and (regime is None or r.get("regime") == regime)})
    if group == "fig8":
        if stem == "figS_dfull_inactive_exposure_cap":
            return [r["run_id"] for r in record.get("raw_files", []) if r.get("budget", "").upper() == "DFULL"]
        budgets = ("D128",) if stem.startswith("fig8a") or stem.endswith("_a") else ("D512",) if stem.startswith("fig8b") or stem.endswith("_b") else ("D128", "D512")
        return [r["run_id"] for r in record.get("raw_files", []) if r.get("budget") in budgets]
    if group == "s5":
        if stem == "figS5_dfull_inactive":
            return ["dfull_stationary_e1_heterogeneous__nr", "dfull_stationary_e1_heterogeneous__exposurecap2", "dfull_stationary_e1_heterogeneous__exposurecap4"]
        if stem == "figS5_equal_budget_timing":
            return sorted({r["run_id"] for r in csv.DictReader((VERIFY / "figS5_schedule_values.csv").open()) if r.get("figure") == stem and r.get("run_id")})
        return sorted({r["run_id"] for r in csv.DictReader((VERIFY / "figS5_plot_values.csv").open()) if r.get("figure") == stem and r.get("run_id")})
    if group == "fig5_alt":
        return [record["run_id"]]
    if isinstance(record.get("run_ids"), list):
        return record["run_ids"]
    if isinstance(record.get("run_ids"), dict):
        return list(record["run_ids"].values())
    if record.get("run_id"):
        return [record["run_id"]]
    return []


def figure_sources(group: str, stem: str, record: dict, run_ids: list[str]) -> dict[str, str]:
    """Narrow a suite-level audit to files actually backing this export."""
    all_sources = sources_from_record(record)
    if group == "fig5new":
        if stem.startswith("fig5a_"):
            return {p: h for p, h in all_sources.items()
                    if ("/EEFL/" in p or
                        "learning_efficiency_confirmation/" in p and
                        ("estimator_predictions.csv" in p or "per_round_hardware_results.csv" in p
                         or "campaign_manifest.json" in p
                         or "/C/" in p and ("fedavg_" in p or
                                             "nano_decision_round_03.json" in p or
                                             "nano_decision_round_04.json" in p)))}
        if stem.startswith("fig5b_"):
            return {p: h for p, h in all_sources.items()
                    if ("learning_efficiency_confirmation/" in p and
                        ("estimator_predictions.csv" in p or "per_round_hardware_results.csv" in p
                         or "campaign_manifest.json" in p or "nano_decision_round_" in p
                         or "/C/" in p and "fedavg_" in p))}
        return {p: h for p, h in all_sources.items()
                if ("learning_efficiency_confirmation/" in p and
                    ("per_round_hardware_results.csv" in p or "campaign_manifest.json" in p
                     or "fedavg_" in p))}
    if group == "fig5" and stem.startswith("fig5a_historical"):
        return {p: h for p, h in all_sources.items()
                if "EEFL/E2FL/eval/evaluate/" in p
                or p.endswith("/exposure_policy/raw/hardware_seed42/estimator_predictions.csv")
                or "/nano_nx_final_campaign/raw/standard_short/C/" in p
                or p == str(STYLE)}
    if group == "fig5" and stem.startswith("fig5b_"):
        return {p: h for p, h in all_sources.items() if "learning_efficiency_confirmation/" in p or p == str(STYLE)}
    if group == "fig7":
        return {p: h for p, h in all_sources.items() if "/runs/" not in p or any(f"/runs/{run_id}/" in p for run_id in run_ids)}
    if group == "fig8":
        return {p: h for p, h in all_sources.items() if "/raw/confirmation_runs/" not in p or any(f"/{run_id}/" in p for run_id in run_ids)}
    if group == "s5" and stem != "figS5_cap_sensitivity":
        keep = set(run_ids)
        if stem == "figS5_dfull_inactive":
            keep |= {"dfull_stationary_e1_heterogeneous__nr", "dfull_stationary_e1_heterogeneous__exposurecap2", "dfull_stationary_e1_heterogeneous__exposurecap4"}
        return {p: h for p, h in all_sources.items() if "/runs/" not in p and "/raw/confirmation_runs/" not in p or any(f"/{run}/" in p for run in keep)}
    return all_sources


def main() -> None:
    style_hash = sha256(STYLE)
    records = {}
    for group, metadata in GROUPS.items():
        path = VERIFY / metadata["audit"]
        if path.exists():
            records[group] = audit(metadata["audit"])

    all_figures = sorted(ROOT.glob("main/*.pdf")) + sorted(ROOT.glob("supplementary/*.pdf"))
    entries = []
    errors = []
    for pdf in all_figures:
        stem = pdf.stem
        group = group_for(stem)
        config = GROUPS[group]
        if group not in records:
            errors.append(f"{stem}: missing group audit {config['audit']}")
            continue
        record = records[group]
        formats = {ext: str(pdf.with_suffix("." + ext)) for ext in ("pdf", "svg", "png")}
        for path in formats.values():
            if not Path(path).is_file():
                errors.append(f"{stem}: missing output {path}")
        script_path = ROOT / "scripts" / config["script"]
        if not script_path.is_file():
            errors.append(f"{stem}: missing script {script_path}")
            continue
        run_ids = run_ids_for(group, record, stem)
        source_hashes = figure_sources(group, stem, record, run_ids)
        resolved_sources = {}
        for source, expected_hash in source_hashes.items():
            path = Path(source)
            if not path.is_absolute():
                path = ROOT.parents[1] / path
            if not path.is_file():
                errors.append(f"{stem}: missing input source {path}")
                continue
            actual_hash = sha256(path)
            if actual_hash != expected_hash:
                errors.append(f"{stem}: hash mismatch {path}")
            resolved_sources[str(path)] = expected_hash
        value_files = []
        for relative in config["value_files"]:
            path = (VERIFY / relative).resolve()
            if path.exists():
                value_files.append({"path": str(path), "sha256": sha256(path)})
            else:
                errors.append(f"{stem}: missing plotted-value file {path}")
        status = "READY"
        if stem in ("plot_cost_prediction_adaptation", "fig5a_historical_pi5_nx_prediction", "fig5a_alternative_reference_window"):
            status = "CONDITIONAL"
        if group == "s2":
            status = "SUPPLEMENTARY_DEVELOPMENT_TRACE"
        entries.append({
            "figure": stem,
            "placement": pdf.parent.name,
            "status": status,
            "formats": formats,
            "plot_script": str(script_path),
            "plot_script_sha256": sha256(script_path),
            "reference_style": str(STYLE),
            "reference_style_sha256": style_hash,
            "source_audit": str(VERIFY / config["audit"]),
            "source_audit_sha256": sha256(VERIFY / config["audit"]),
            "run_ids": run_ids,
            "seeds": config["seeds"],
            "rounds": config["rounds"],
            "policy_version": config["version"],
            "metric_definition": config["metric"],
            "measurement_boundary": config["boundary"],
            "source_files_sha256": resolved_sources,
            "plotted_value_files": value_files,
        })

    if errors:
        raise SystemExit("\n".join(errors))
    manifest = {
        "scope": "verified pre-existing experimental evidence, no new training/scoring/hardware runs",
        "style_sha256": style_hash,
        "figure_count": len(entries),
        "figures": entries,
    }
    (ROOT / "figure_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_data_mapping()
    print(f"Wrote provenance for {len(entries)} PDF/SVG/PNG sets")


def write_data_mapping() -> None:
    """Map displayed quantities to logged fields and noninvented transformations."""
    rows = [
        ("Fig. 5(a)", "July Pi5/NX and October Nano prediction", "July Pi5/NX hardware CSV; October Nano pre-round decision JSON and Flower CSV", "pred_train_time_self; actual_train_time_self; reference_predicted_seconds; loop_time; phase; round", "mean(100*abs(pred-actual)/actual), train rows after R2; Nano separated into normal R3-4, contention R5-8, recovery R9-12", "percent", "Five plotted bars span two different campaigns; Nano pooled 24.60% is archived but not drawn"),
        ("Fig. 5(b)", "NX adaptation", "Nano/NX Flower server CSV plus per_round_hardware_results.csv", "local_steps; round; policy; seed", "executed NX steps, round-matched arithmetic mean across seeds 43–45", "optimizer steps", "Actual execution; R5 decision precedes R5 feedback"),
        ("Fig. 5 alternative (a)", "reference window", "system_design_figures/inputs/pi5_nx_s43_r03_preallocation_decision.json", "candidate_workloads; predicted_compute_s; reference_compute_s; selected", "plot 16 unique logged S/H candidates; allowed=0.95*reference", "seconds", "Pre-dispatch predictions, not measured execution"),
        ("Fig. 6(a)", "active time", "checkpoint_time_energy_alignment.csv", "cumulative_active_time_s; full_sequence_nll_bf16; seed; policy; round", "seconds/60; arithmetic means of time and NLL at matched observed rounds", "minutes; NLL", "Measured Nano+NX server-active time; RTX checkpoint NLL"),
        ("Fig. 6(b)", "client energy", "checkpoint_time_energy_alignment.csv", "cumulative_client_energy_j; full_sequence_nll_bf16; seed; policy; round", "joules/1000; arithmetic means of energy and NLL at matched observed rounds", "kJ; NLL", "Measured Nano+NX client VDD_IN only; server/AP excluded"),
        ("Table 1", "real-device run summary", "policy_summary.csv; checkpoint_validation_loss.csv", "total_input_tokens; active_time_s; client_energy_j; full_sequence_nll_bf16", "calculate per-run tokens/s and J/1K, then mean/SD across seeds", "tokens/s; J/1K; NLL", "Run is the independent unit"),
        ("Fig. 7(a)", "equal-work epochs", "long_horizon_100r/runs/d128_stationary_e*_equal_work__nr/{validation,rounds}.jsonl", "round; balanced_client_loss; clients.nx.optimizer_steps; clients.nx.samples", "observed checkpoint NLL; sum actual executed NX steps for work audit", "NLL; steps", "Seed42 RTX; E1/E2/E4 each 400 NX optimizer steps"),
        ("Fig. 7(b)", "window-filling epochs", "long_horizon_100r/runs/d128_stationary_e*_heterogeneous__nr/{validation,rounds}.jsonl", "round; balanced_client_loss; clients.nx.optimizer_steps", "observed checkpoint NLL; actual NX step totals are 3176/5528/4744", "NLL; steps", "Seed42 RTX; E2/E4 edge-window feasibility unsupported"),
        ("Fig. 8(a)", "D128 budget policies", "exposure_policy/raw/confirmation_runs/confirm_s{43,44,45}_d128_{nr,full,exposurecap2}/validation.jsonl", "round; balanced_client_loss; checkpoint_sha256", "round-matched arithmetic mean across observed checkpoints; no checkpoint interpolation", "NLL", "Actual RTX learning; October hard omission only"),
        ("Fig. 8(b)", "D512 budget policies", "exposure_policy/raw/confirmation_runs/confirm_s{43,44,45}_d512_{nr,full,exposurecap2}/validation.jsonl", "round; balanced_client_loss; checkpoint_sha256", "round-matched arithmetic mean across observed checkpoints; no checkpoint interpolation", "NLL", "Actual RTX learning; October hard omission only"),
        ("Fig. S1", "Pi5/NX F/A/B/C", "final_dynamic_policy_comparison_20261019/policy_summary.csv", "total FL input tokens; measured active server seconds; measured active client joules", "per-run tokens/s=tokens/time; J/1K=1000*joules/tokens; mean over two seeds", "tokens/s; J/1K", "F and A outperform C in these two paired runs"),
        ("Fig. S2(a)", "quota response", "cross_client_development_pilot_20261019/policy_rounds.csv; raw Flower server CSV", "Pi scheduled quota; Pi actual local compute s; NX executed optimizer steps", "plot recorded 16 rounds for four policies", "seconds; steps", "Single-seed development trace"),
        ("Fig. S2(b)", "reply barrier and efficiency", "cross_client_development_pilot_20261019/policy_reply_barrier.csv; raw reply_arrivals.jsonl", "server dispatch and first all-replies poll; executed input tokens; server-active time", "reply barrier=first all-replies poll-dispatch; efficiency=tokens/server-active second", "seconds; tokens/s", "Barrier includes polling and noncompute envelope"),
        ("Fig. S3(a)", "time to target", "checkpoint_time_energy_alignment.csv", "cumulative_active_time_s; full_sequence_nll_bf16; seed", "first observed checkpoint at or below frozen target; median among reached seeds", "minutes", "Nonattainment excluded with reached count shown"),
        ("Fig. S3(b)", "energy to target", "checkpoint_time_energy_alignment.csv", "cumulative_client_energy_j; full_sequence_nll_bf16; seed", "first observed checkpoint at or below frozen target; median among reached seeds", "kJ", "Measured Nano+NX client VDD_IN; no imputed censored runs"),
        ("Fig. S4", "quota and omission", "confirm_s43_d512_exposurecap2/rounds.jsonl", "clients.nx.samples; clients.nx.optimizer_steps; nx_exposure; action", "cumulative executed NX samples / 512; remaining exposure=2-exposure", "samples; exposure", "Actual RTX work: partial R33, omission R34–R100"),
        ("Fig. S5 cap sensitivity", "D128/D512", "long_horizon_100r and exposure_cap_regulation_100r validation.jsonl", "round; balanced_client_loss; checkpoint_sha256", "plot scored R10–R100 checkpoints; D512 Exposure-8 reuses verified nonbinding Unbounded", "NLL", "Exploratory seed42"),
        ("Fig. S5 scheduling", "same exposure/timing", "Exposure-2/Distributed-2 rounds.jsonl", "clients.nx.samples; nx_exposure; round", "cumulative actual executed NX samples / D; both policies execute the same total sample budget", "exposure", "RTX scheduling ablation, no measured edge energy"),
        ("Fig. S5 arrival", "R41 data-volume arrival", "arrival_s46_{nr,exposurecap2}/{rounds,validation}.jsonl", "round; balanced_client_loss; clients.nx.samples; nx_eligible_unique_samples", "cumulative executed samples; cap=2*currently eligible D", "NLL; samples", "RTX data-volume change, not a physical data-arrival deployment"),
        ("Fig. S5 DFull", "inactive caps", "DFull NR/Exposure-2/Exposure-4 rounds.jsonl and validation.jsonl", "round; balanced_client_loss; checkpoint_sha256; clients.nx.samples", "one shared curve after verifying exact work and 13 checkpoint-hash equality", "NLL", "No separate improvement from inactive cap"),
        ("Fig. S6(a)", "historical compute-time error", "July Pi5/NX MAXN/15W/10W hardware CSV", "pred_train_time_self; actual_train_time_self", "100*abs(pred-actual)/actual for train rows after R2", "percent", "Historical July predictor, local compute only"),
        ("Fig. S6(b)", "historical compute-energy error", "July Pi5/NX MAXN/15W/10W hardware CSV", "predicted compute duration; warm-up median compute power; observed compute power/duration", "100*abs(predicted_local_compute_energy-observed_local_compute_energy)/observed_local_compute_energy", "percent", "Local compute only; not total device energy"),
        ("Supplementary D512/DFull epochs", "epoch sensitivity", "long_horizon_100r/runs/d{512,full}_stationary_e*_{equal_work,heterogeneous}__nr/{rounds,validation}.jsonl", "round; balanced_client_loss; clients.nx.optimizer_steps", "observed checkpoint NLL and executed-step audit", "NLL; steps", "Seed42, E2/E4 edge-time feasibility unsupported"),
        ("Supplementary DFull budgets", "inactive Exposure-2", "confirm_s{43,44,45}_dfull_{nr,full}/validation.jsonl; manifest equivalence", "round; balanced_client_loss; checkpoint_sha256", "mean observed seed checkpoints; Exposure-2 is exact NR alias", "NLL", "Three seeds, no independent cap run"),
    ]
    path = ROOT / "figure_data_mapping.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("Figure", "Panel", "Source file or record", "Original columns", "Plot transformation", "Units", "Interpretation limit"))
        writer.writerows(rows)


MAIN_FINAL = {
    "fig5a_normal_prediction_mare": ("fig5new", "Fig. 5(a)", "CONDITIONAL_CROSS_CAMPAIGN_ENERGY_PROXY", "Pi5/NX seed IDs unrecorded; Nano 43–45 independent", "Pi5/NX post-R2; Nano normal R3–R4", "Pi5/NX pred_train_time_self, loop_time, client-logged power; Nano native reference_predicted_compute_s, nano_actual_compute_s, nano_active_energy_j, server_active_round_s", "per-device arithmetic mean of APE; energy proxy = warm-up median power × predicted local duration versus observed mean power × observed local duration; Nano power = active-round VDD_IN joules / server-active seconds", "percent", "Pi5/NX and Nano use different campaigns/model/settings/active correction and different power-measurement scopes; Nano proxy is reconstructed offline, not native energy prediction or measured compute-only joules; bars are descriptive, not a controlled ranking", ("fig5a_normal_prediction_values.csv", "fig5a_nano_normal_prediction_values.csv", "fig5a_normal_prediction_mare.csv")),
    "fig5b_contention_prediction_ape": ("fig5new", "Fig. 5(b)", "READY_WITH_NEGATIVE_RECOVERY_RESULT", "43–45 independent", "R3–R12, after R1–R2 calibration", "reference_predicted_compute_s; nano_actual_compute_s; seed; round", "seed-level 100 × |pre-round predicted − observed| / observed; round-matched mean; frozen R3 state held at fixed Nano workload", "percent", "Frozen is an offline prediction-only replay, not a separately executed scheduling policy; Online is worse during recovery", ("fig5b_contention_prediction_by_seed.csv", "fig5b_contention_prediction_ape.csv", "fig5b_contention_phase_summary.csv")),
    "fig5c_runtime_workload_adaptation": ("fig5new", "Fig. 5(c)", "READY", "43–45 independent", "R1–R12", "client-reported NX local_steps; policy; seed; round", "round-matched arithmetic mean of actually executed NX optimizer steps", "optimizer steps", "LoRACast, Fixed-Hetero, and restricted PyramidFL-style follow distinct real Flower trajectories; allocator response alone does not establish causal benefit", ("fig5c_executed_steps_by_seed.csv", "fig5c_runtime_workload_adaptation.csv", "fig5c_local_compute_window_diagnostics.csv")),
    "fig6a_learning_vs_active_time": ("fig6", "Fig. 6(a)", "READY", "43–45 independent", "observed R0–R12 checkpoints", "cumulative_measured_active_s; full_sequence_nll_bf16_l64; seed; policy; round", "seconds / 60; arithmetic means of time and NLL at each matched round; connect observed mean coordinates; marked y break", "minutes; NLL", "Measured Nano+NX server-active time; RTX checkpoint scoring", ("fig6_plot_data.csv", "fig6_seed_run_summary.csv")),
    "fig6b_learning_vs_client_energy": ("fig6", "Fig. 6(b)", "READY", "43–45 independent", "observed R0–R12 checkpoints", "cumulative_measured_client_j; full_sequence_nll_bf16_l64; seed; policy; round", "joules / 1000; arithmetic means of energy and NLL at each matched round; connect observed mean coordinates; marked y break", "kJ; NLL", "Measured Nano+NX client VDD_IN during server-active rounds; excludes server/AP", ("fig6_plot_data.csv", "fig6_seed_run_summary.csv")),
    "fig7a_equal_work_local_epochs": ("fig7", "Fig. 7(a)", "READY", "42 exploratory training; 777 evaluation", "R100 displayed; 13 checkpoints preserved", "round; balanced_client_loss; clients.nx.optimizer_steps; completed_passes", "R100 observed endpoint dot plot; E1/E2/E4 each execute 400 NX optimizer steps; full trajectory retained in verification", "NLL; steps", "Single seed and endpoint only; equal steps do not imply equal input tokens or edge time", ("fig7_single_panel_plotted_values.csv", "fig7_executed_work.csv")),
    "fig7b_window_filling_local_epochs": ("fig7", "Fig. 7(b)", "CONDITIONAL_EDGE_FEASIBILITY", "42 exploratory training; 777 evaluation", "100 rounds; 13 preserved checkpoints", "round; balanced_client_loss; clients.nx.optimizer_steps", "observed checkpoint NLL; NX E1/E2/E4 totals 3176/5528/4744 steps", "NLL; steps", "RTX learning observed; E2/E4 edge-window feasibility uses an unvalidated E1 timing proxy", ("fig7_single_panel_plotted_values.csv", "fig7_executed_work.csv")),
    "fig8a_d128_training_budget_management": ("fig8", "Fig. 8(a)", "READY", "43–45 independent training", "100 rounds; 13 checkpoints/run", "round; balanced_client_loss; checkpoint_sha256; seed; policy", "round-matched arithmetic mean of observed seed checkpoints, D128", "NLL", "October RTX hard-omission policy; no edge energy is plotted", ("fig8_single_panel_plotted_values.csv", "../tables/fig8_mean_values.csv")),
    "fig8b_d512_training_budget_management": ("fig8", "Fig. 8(b)", "READY", "43–45 independent training", "100 rounds; 13 checkpoints/run", "round; balanced_client_loss; checkpoint_sha256; seed; policy", "round-matched arithmetic mean of observed seed checkpoints, D512", "NLL", "October RTX hard-omission policy; no edge energy is plotted", ("fig8_single_panel_plotted_values.csv", "../tables/fig8_mean_values.csv")),
}

MAIN_POLICY_VERSION = {
    "fig5a_normal_prediction_mare": "July Pi 5/NX MAXN/15W/10W estimator plus October Nano Super LoRACast-NR normal R3–R4; active correction differs",
    "fig5b_contention_prediction_ape": "October Nano Super/NX LoRACast-NR with active online correction; frozen prediction-only replay",
    "fig5c_runtime_workload_adaptation": "October Nano Super/NX LoRACast-NR, Fixed-Hetero, restricted PyramidFL-style",
    "fig6a_learning_vs_active_time": "Nano Super/NX FedAvg, Fixed-Hetero, restricted PyramidFL-style, and LoRACast-NR",
    "fig6b_learning_vs_client_energy": "Nano Super/NX FedAvg, Fixed-Hetero, restricted PyramidFL-style, and LoRACast-NR",
    "fig7a_equal_work_local_epochs": "October RTX NR, D128 stationary equal-work E1/E2/E4",
    "fig7b_window_filling_local_epochs": "October RTX NR, D128 stationary heterogeneous window-filling E1/E2/E4",
    "fig8a_d128_training_budget_management": "October RTX D128 NR, W3/theta0.3 pre-round hard omission, and pretraining Exposure-2",
    "fig8b_d512_training_budget_management": "October RTX D512 NR, W3/theta0.3 pre-round hard omission, and pretraining Exposure-2",
}

MAIN_METRIC = {
    "fig5a_normal_prediction_mare": "Three-device local compute-time MARE and separately qualified derived energy-proxy MARE",
    "fig5b_contention_prediction_ape": "Native pre-round Nano local compute-time APE versus offline frozen replay",
    "fig5c_runtime_workload_adaptation": "Actual client-reported NX executed optimizer steps",
    "fig6a_learning_vs_active_time": "BF16 full-sequence 64-token held-out validation NLL versus measured server-active time",
    "fig6b_learning_vs_client_energy": "BF16 full-sequence 64-token held-out validation NLL versus measured Nano+NX client VDD_IN energy",
    "fig7a_equal_work_local_epochs": "Balanced Pi/NX held-out full-sequence 64-token validation NLL",
    "fig7b_window_filling_local_epochs": "Balanced Pi/NX held-out full-sequence 64-token validation NLL",
    "fig8a_d128_training_budget_management": "Balanced Pi/NX held-out full-sequence 64-token validation NLL",
    "fig8b_d512_training_budget_management": "Balanced Pi/NX held-out full-sequence 64-token validation NLL",
}

MAIN_BOUNDARY = {
    "fig5a_normal_prediction_mare": "Pi5/NX local compute time plus client-power-times-duration proxy; Nano local compute time plus active-round VDD_IN mean-power-times-duration proxy; neither proxy is phase-isolated measured compute energy",
    "fig5b_contention_prediction_ape": "October Nano Super actual local compute time, same fixed executed workload; no separate frozen FL trajectory",
    "fig5c_runtime_workload_adaptation": "October actual Nano Super/NX client-reported optimizer steps; no model-quality metric in this graph",
    "fig6a_learning_vs_active_time": "Measured Nano+NX server-active time paired with RTX-scored hardware-generated checkpoints",
    "fig6b_learning_vs_client_energy": "Measured Nano+NX VDD_IN energy during server-active rounds paired with RTX-scored checkpoints; excludes server/AP",
    "fig7a_equal_work_local_epochs": "Actual seed-42 RTX LoRA training and validation; no edge cost plotted",
    "fig7b_window_filling_local_epochs": "Actual seed-42 RTX LoRA training and validation; E2/E4 edge timing remains unvalidated",
    "fig8a_d128_training_budget_management": "Actual seeds 43–45 RTX LoRA training and validation; no edge cost plotted",
    "fig8b_d512_training_budget_management": "Actual seeds 43–45 RTX LoRA training and validation; no edge cost plotted",
}


SUPP_FIELDS = {
    "figS1a": ("input_tokens; measured_server_active_s; seed; policy", "input_tokens / measured_server_active_s per run; two-seed mean with seed dots", "input tokens/s"),
    "figS1b": ("measured_client_device_active_j; input_tokens; seed; policy", "1000 × measured_client_device_active_j / input_tokens per run; two-seed mean with seed dots", "J/1K input tokens"),
    "figS2a": ("round; policy_code; pi_actual_local_compute_s", "plot observed Pi compute duration per recorded round and policy", "seconds"),
    "figS2b": ("round; policy_code; nx_executed_optimizer_steps", "plot observed NX executed optimizer steps per recorded round and policy", "optimizer steps"),
    "figS2c": ("policy_code; input_tokens_per_server_active_s", "plot measured FL input tokens per server-active second by policy", "input tokens/s"),
    "figS2d": ("server_dispatch_to_all_replies_s; policy_code", "all-replies polling timestamp minus server dispatch timestamp", "seconds"),
    "figS3a": ("full_sequence_nll_bf16; cumulative_active_time_s; seed; frozen target_nll", "first observed checkpoint at or below target; median among attained seeds; show reached count", "minutes"),
    "figS3b": ("full_sequence_nll_bf16; cumulative_client_energy_j; seed; frozen target_nll", "first observed checkpoint at or below target; median among attained seeds; show reached count", "kJ"),
    "figS4a": ("round; nx_executed_samples; action", "actual executed NX samples per round; partial R33 and pre-training omission R34–R100", "samples"),
    "figS4b": ("round; cumulative_executed_samples; exposure_after", "cumulative executed NX samples / 512", "sample exposure"),
    "figS5a": ("round; balanced_client_loss; checkpoint_sha256; run_id", "observed D128 validation checkpoints by prespecified cap/schedule; no interpolation", "NLL"),
    "figS5b": ("round; balanced_client_loss; checkpoint_sha256; run_id", "observed D512 validation checkpoints; nonbinding Cap-8 aliases NR", "NLL"),
    "figS5c": ("round; clients.nx.samples; cumulative_executed_nx_samples", "cumulative actual NX executed samples / 128 for front-loaded and distributed schedules", "sample exposure"),
    "figS5d": ("round; clients.nx.samples; cumulative_executed_nx_samples", "cumulative actual NX executed samples / 512 for front-loaded and distributed schedules", "sample exposure"),
    "figS5e": ("round; balanced_client_loss; checkpoint_sha256; run_id", "actual RTX validation at preserved checkpoints, before/after R41 volume arrival", "NLL"),
    "figS5f": ("round; clients.nx.samples; nx_eligible_unique_samples", "cumulative executed NX samples; eligibility expands 128 to 512 at R41", "samples"),
    "figS5g": ("round; balanced_client_loss; checkpoint_sha256; clients.nx.samples", "shared DFull curve after verifying NR/Cap-2/Cap-4 work and checkpoint equality", "NLL"),
    "figS6a": ("pred_train_time_self; actual_train_time_self; phase; round", "100 × |predicted − actual| / actual, local train records after R2", "percent"),
    "figS6b": ("predicted local compute duration; warm-up median logged power; logged mean power and observed duration", "100 × |predicted − observed power-times-duration proxy| / observed proxy", "percent"),
    "figS7a": ("round; balanced_client_loss; clients.nx.optimizer_steps; completed_passes", "observed D512 equal-work E1/E2/E4 checkpoint NLL; four NX steps/round", "NLL; steps"),
    "figS7b": ("round; balanced_client_loss; clients.nx.optimizer_steps", "observed D512 window-filling E1/E2/E4 checkpoint NLL", "NLL; steps"),
    "figS7c": ("round; balanced_client_loss; clients.nx.optimizer_steps; completed_passes", "observed DFull equal-work E1/E2/E4 checkpoint NLL; four NX steps/round", "NLL; steps"),
    "figS7d": ("round; balanced_client_loss; clients.nx.optimizer_steps", "observed DFull window-filling E1/E2/E4 checkpoint NLL", "NLL; steps"),
    "figS8_": ("round; balanced_client_loss; checkpoint_sha256; seed; policy", "three-seed DFull checkpoint means; inactive Exposure-2 is an exact NR alias", "NLL"),
}


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def supplementary_group(stem: str) -> str:
    match = re.match(r"figS(\d)", stem)
    if not match:
        raise ValueError(f"Unrecognized final supplementary filename: {stem}")
    number = int(match.group(1))
    return {1: "s1", 2: "s2", 3: "s3", 4: "s4", 5: "s5", 6: "s6", 7: "fig7", 8: "fig8"}[number]


def supplementary_seeds(stem: str) -> str:
    if stem.startswith("figS1"):
        return "43–44 paired"
    if stem.startswith("figS2"):
        return "42 single development seed"
    if stem.startswith("figS3") or stem.startswith("figS8"):
        return "43–45 independent"
    if stem.startswith("figS4"):
        return "43"
    if stem.startswith(("figS5e", "figS5f")):
        return "46"
    if stem.startswith("figS5") or stem.startswith("figS7"):
        return "42 exploratory"
    return "Not recorded in July source CSVs"


def supplementary_run_ids(stem: str, sources: dict[str, str], fig8_record: dict) -> list[str]:
    ids = {match.group(1) for path in sources for match in
           [re.search(r"/(?:runs|confirmation_runs)/([^/]+)/", path)] if match}
    if stem.startswith("figS1"):
        ids |= {r["run_id"] for r in csv_rows(VERIFY / "fig5_s1_pi5_nx_ablation_values.csv")}
    elif stem.startswith("figS2"):
        ids |= {r["run_id"] for r in csv_rows(VERIFY / "figS2_plot_data.csv")}
    elif stem.startswith("figS3"):
        ids |= {r["run_id"] for r in csv_rows(VERIFY / "fig6_plot_data.csv")
                if r.get("record_type") == "observed_seed_checkpoint"}
    elif stem.startswith("figS4"):
        ids |= {r["run_id"] for r in csv_rows(ROOT / "tables/figS4_quota_omission_trace.csv")}
    elif stem.startswith(("figS5a", "figS5b")):
        budget = "d128" if stem.startswith("figS5a") else "d512"
        ids |= {r["run_id"] for r in csv_rows(VERIFY / "figS5_plot_values.csv")
                if r["figure"] == "figS5_cap_sensitivity" and r["panel"] == budget}
    elif stem.startswith(("figS5c", "figS5d")):
        budget = "D128" if stem.startswith("figS5c") else "D512"
        ids |= {r["run_id"] for r in csv_rows(VERIFY / "figS5_schedule_values.csv")
                if r["figure"] == "figS5_equal_budget_timing" and r["budget"] == budget}
    elif stem.startswith("figS5e"):
        ids |= {r["run_id"] for r in csv_rows(VERIFY / "figS5_plot_values.csv")
                if r["figure"] == "figS5_data_volume_arrival"}
    elif stem.startswith("figS5f"):
        ids |= {r["run_id"] for r in csv_rows(VERIFY / "figS5_schedule_values.csv")
                if r["figure"] == "figS5_data_volume_arrival"}
    elif stem.startswith("figS8_"):
        ids |= {r["run_id"] for r in fig8_record["raw_files"] if r["budget"] == "DFULL"}
    if not ids and not stem.startswith("figS6"):
        raise ValueError(f"No traceable run ID for {stem}")
    return sorted(ids)


def supplementary_experiment_sources(stem: str, panel_sources: dict[str, str],
                                     group_records: dict[str, dict],
                                     previously_verified: dict[str, str] | None = None) -> tuple[dict[str, str], str]:
    """Extend panel inputs only with already-hashed upstream experiment records."""
    sources = dict(panel_sources)
    note = "Per-panel sources include run-specific experiment records."
    if stem.startswith("figS1"):
        record = group_records["s1"]
        for path_key, digest_key in (
            ("source_admitted_matrix_csv", "source_admitted_matrix_sha256"),
            ("source_final_qc_json", "source_final_qc_sha256"),
            ("source_manifest_json", "source_manifest_sha256"),
        ):
            sources[record[path_key]] = record[digest_key]
        original_manifest_path = Path(record["source_manifest_json"])
        if (getattr(stat, "SF_DATALESS", 0)
                and original_manifest_path.stat().st_flags & stat.SF_DATALESS):
            if not previously_verified:
                raise ValueError("Dataless S1 campaign manifest requires the prior verified source map")
            sources.update(previously_verified)
            note = ("Earlier source-verified S1 campaign files were retained by SHA-256; "
                    "the external iCloud manifest is dataless during this revision.")
            return sources, note
        original_manifest = json.loads(original_manifest_path.read_text())
        campaign_root = Path(record["source_manifest_json"]).parent
        sources[str(campaign_root / "policy_rounds.csv")] = original_manifest["result_files_sha256"]["policy_rounds.csv"]
        for run in original_manifest["runs"]:
            sources[run["verification_file"]] = run["verification_sha256"]
        note = "Original measured campaign policy_summary/policy_rounds, frozen admitted matrix, per-run verification, and source manifest; raw sensor-stream hashes were not added."
    elif stem.startswith("figS2"):
        raw = group_records["s2"]["raw_file_sha256"]
        selected = {path: digest for path, digest in raw.items()
                    if "/devices/server/server/" in path and
                    (Path(path).name.startswith("fedavg_") and path.endswith(".csv") or
                     Path(path).name == "reply_arrivals.jsonl")}
        if len(selected) != 8:
            raise ValueError(f"Expected four raw Flower server CSVs and four reply-arrival logs; got {len(selected)}")
        sources.update(selected)
        note = "Single-seed development pilot: four original Flower server CSVs and four reply-arrival logs are hashed upstream of the derived policy tables."
    elif stem.startswith("figS3"):
        raw = group_records["fig6"].get("prior_original_campaign_source_sha256",
                                          group_records["fig6"]["input_source_sha256"])
        selected = {path: digest for path, digest in raw.items()
                    if Path(path).name in ("checkpoint_validation_loss.csv", "per_round_results.csv", "analysis_qc.json")}
        if len(selected) != 3:
            raise ValueError("Cannot find all three hashed independent-confirmation measurement records")
        sources.update(selected)
        sources.update(group_records["fig6"]["input_source_sha256"])
        note = "Frozen target table connects to measured per-round hardware results and actual RTX-scored validation records via their hashed alignment and QC files."
    elif stem.startswith("figS8_"):
        raw_runs = [run for run in group_records["fig8"]["raw_files"] if run["budget"] == "DFULL"]
        if len(raw_runs) != 6:
            raise ValueError("DFull confirmation source audit lacks six NR/Full seed-level runs")
        for run in raw_runs:
            for base in ("validation", "rounds", "state"):
                suffix = "jsonl" if base in ("validation", "rounds") else "json"
                sources[run[f"{base}_{suffix}"]] = run[f"{base}_sha256"]
        note = "Six original DFull NR/Full confirmation validation, executed-round, and state logs are hashed; inactive Cap-2 is a verified NR alias, not a seventh run."
    elif stem.startswith("figS6"):
        note = "Historical July device-local compute CSVs are direct inputs; no seed IDs are recorded."
    return sources, note


def final_main() -> None:
    """Build one provenance entry and one value-mapping row per final graph."""
    previous_manifest_path = ROOT / "figure_source_manifest.json"
    previous_manifest = json.loads(previous_manifest_path.read_text())
    previous_entries = {entry["figure"]: entry for entry in previous_manifest["figures"]}
    dataless_flag = getattr(stat, "SF_DATALESS", 0)

    def is_cloud_placeholder(path: Path) -> bool:
        return bool(dataless_flag and path.stat().st_flags & dataless_flag)

    panel_manifest_path = VERIFY / "supplementary_single_panel_manifest.json"
    panel_manifest = json.loads(panel_manifest_path.read_text())
    if is_cloud_placeholder(STYLE):
        # macOS iCloud placeholders retain metadata but block when read. The
        # reference digest was verified in the previous figure-production run.
        style_hash = previous_manifest["style_sha256"]
        style_check = "prior_verified_hash; current iCloud file is dataless"
    else:
        style_hash = sha256(STYLE)
        style_check = "rehashed_now"
    supp_by_stem = {r["name"]: r for r in panel_manifest["figures"]}
    if panel_manifest["panel_count"] != 24 or len(supp_by_stem) != 24 or not panel_manifest["one_axes_per_file"]:
        raise ValueError("Supplementary single-panel manifest must contain exactly 24 distinct verified graphs")
    if panel_manifest["style_sha256"] != style_hash:
        raise ValueError("Supplementary style digest differs from EEFL reference")
    expected_main = set(MAIN_FINAL)
    expected_supp = set(supp_by_stem)
    actual_main = {p.stem for p in (ROOT / "main").glob("*.pdf")}
    actual_supp = {p.stem for p in (ROOT / "supplementary").glob("*.pdf")}
    if actual_main != expected_main or actual_supp != expected_supp:
        raise ValueError(f"Final PDF inventory mismatch. Main extra={sorted(actual_main-expected_main)}, "
                         f"main missing={sorted(expected_main-actual_main)}, supplement extra={sorted(actual_supp-expected_supp)}, "
                         f"supplement missing={sorted(expected_supp-actual_supp)}")
    group_records = {group: audit(meta["audit"]) for group, meta in GROUPS.items()}
    all_entries = []
    errors = []
    hash_cache: dict[str, str] = {}
    source_rechecks: dict[str, dict[str, str]] = {}

    def checked_hash(path: Path) -> str:
        key = str(path.resolve())
        if key not in hash_cache:
            hash_cache[key] = sha256(path)
        return hash_cache[key]

    def checked_sources(stem: str, expected: dict[str, str]) -> dict[str, str]:
        output = {}
        states = {}
        prior_sources = previous_entries.get(stem, {}).get("source_files_sha256", {})
        is_new_fig5 = stem in ("fig5a_normal_prediction_mare", "fig5b_contention_prediction_ape",
                               "fig5c_runtime_workload_adaptation")
        fig5_prior_audit = audit("fig5_estimator_contention_validation_initial_pass.json") if is_new_fig5 else {}
        fig5_prior_sources = fig5_prior_audit.get("source_sha256", {})
        fig5_current_audit = group_records.get("fig5new", {}) if is_new_fig5 else {}
        fig5_unread = set(fig5_current_audit.get("original_records_not_reread_this_run", []))
        for source, wanted in expected.items():
            path = Path(source)
            if not path.is_absolute():
                path = ROOT.parents[1] / path
            if not path.is_file():
                errors.append(f"{stem}: source missing: {path}")
                continue
            resolved = str(path.resolve())
            if source in fig5_unread:
                if fig5_prior_sources.get(source) != wanted:
                    errors.append(f"{stem}: original-record audit hash mismatch: {source}")
                states[resolved] = "prior_original_record_audit; current iCloud source not reread"
                output[resolved] = wanted
                continue
            if is_cloud_placeholder(path):
                if prior_sources.get(resolved, fig5_prior_sources.get(source)) != wanted:
                    errors.append(f"{stem}: dataless source has no matching prior verified hash: {path}")
                states[resolved] = "prior_verified_hash; current iCloud file is dataless"
            else:
                found = checked_hash(path)
                if found != wanted:
                    errors.append(f"{stem}: source SHA-256 mismatch: {path}")
                states[resolved] = "rehashed_now"
            output[resolved] = wanted
        source_rechecks[stem] = states
        return output

    def checked_formats(stem: str, placement: str) -> dict[str, str]:
        paths = {ext: ROOT / placement / f"{stem}.{ext}" for ext in ("pdf", "svg", "png")}
        for path in paths.values():
            if not path.is_file():
                errors.append(f"{stem}: export missing: {path}")
        return {ext: str(path.resolve()) for ext, path in paths.items()}

    def checked_values(stem: str, relatives) -> list[dict]:
        output = []
        for relative in relatives:
            path = (VERIFY / relative).resolve()
            if not path.is_file():
                errors.append(f"{stem}: plotted-value file missing: {path}")
            else:
                output.append({"path": str(path), "sha256": checked_hash(path)})
        return output

    for stem, (group, label, status, seeds, rounds, fields, transform, units, limit, value_files) in MAIN_FINAL.items():
        record = group_records[group]
        run_ids = run_ids_for(group, record, stem)
        if group not in ("fig5",) and not run_ids:
            errors.append(f"{stem}: missing selected run IDs")
        hashes = checked_sources(stem, figure_sources(group, stem, record, run_ids))
        script_name = "plot_epoch_budget_panels.py" if group in ("fig7", "fig8") else GROUPS[group]["script"]
        script_path = ROOT / "scripts" / script_name
        panel_audit = (VERIFY / ("fig7_single_panel_validation.json" if group == "fig7" else
                                 "fig8_single_panel_validation.json" if group == "fig8" else GROUPS[group]["audit"]))
        if not script_path.is_file() or not panel_audit.is_file():
            errors.append(f"{stem}: script or panel audit missing")
            continue
        if group in ("fig7", "fig8"):
            verified_outputs = {x["stem"] for x in json.loads(panel_audit.read_text())["outputs"]}
            if stem not in verified_outputs:
                errors.append(f"{stem}: absent from single-panel validation")
        all_entries.append({
            "figure": stem, "figure_label": label, "placement": "main", "status": status,
            "formats": checked_formats(stem, "main"),
            "plot_script": str(script_path.resolve()), "plot_script_sha256": checked_hash(script_path),
            "reference_style": str(STYLE), "reference_style_sha256": style_hash,
            "source_audit": str((VERIFY / GROUPS[group]["audit"]).resolve()),
            "source_audit_sha256": checked_hash(VERIFY / GROUPS[group]["audit"]),
            "single_panel_audit": str(panel_audit.resolve()),
            "single_panel_audit_sha256": checked_hash(panel_audit),
            "run_ids": run_ids, "seeds": seeds, "rounds": rounds,
            "policy_version": MAIN_POLICY_VERSION[stem], "metric_definition": MAIN_METRIC[stem],
            "measurement_boundary": MAIN_BOUNDARY[stem], "interpretation_limit": limit,
            "source_files_sha256": hashes,
            "source_recheck_state": source_rechecks[stem],
            "plotted_value_files": checked_values(stem, value_files),
            "source_fields": fields, "plot_transformation": transform, "units": units,
        })

    supplement_script = Path(panel_manifest["plot_script"])
    if not supplement_script.is_file() or checked_hash(supplement_script) != panel_manifest["plot_script_sha256"]:
        errors.append("Supplementary plotting script is missing or has changed since panel validation")
    for stem, panel in sorted(supp_by_stem.items()):
        group = supplementary_group(stem)
        prefix = next((key for key in SUPP_FIELDS if stem.startswith(key)), None)
        if prefix is None:
            errors.append(f"{stem}: no value transformation map")
            continue
        fields, transform, units = SUPP_FIELDS[prefix]
        enriched_sources, source_chain_note = supplementary_experiment_sources(
            stem, panel["sources_sha256"], group_records,
            previous_entries.get(stem, {}).get("source_files_sha256"))
        run_ids = supplementary_run_ids(stem, enriched_sources, group_records["fig8"])
        sources = checked_sources(stem, enriched_sources)
        original_audit = VERIFY / GROUPS[group]["audit"]
        if panel["pdf"] != str((ROOT / "supplementary" / f"{stem}.pdf").resolve()):
            errors.append(f"{stem}: panel-manifest PDF path differs from final inventory")
        if panel["svg"] != str((ROOT / "supplementary" / f"{stem}.svg").resolve()):
            errors.append(f"{stem}: panel-manifest SVG path differs from final inventory")
        if panel["png_600dpi"] != str((ROOT / "supplementary" / f"{stem}.png").resolve()):
            errors.append(f"{stem}: panel-manifest PNG path differs from final inventory")
        if stem.startswith("figS2"):
            status = "SUPPLEMENTARY_SINGLE_SEED_DEVELOPMENT"
        elif stem.startswith("figS6"):
            status = "SUPPLEMENTARY_HISTORICAL"
        elif stem.startswith(("figS7b", "figS7d")):
            status = "CONDITIONAL_EDGE_FEASIBILITY"
        else:
            status = "READY"
        all_entries.append({
            "figure": stem, "figure_label": "Fig. S" + stem[4] +
            (f"({stem[5]})" if len(stem) > 5 and stem[5] in "abcdefg" else ""),
            "placement": "supplementary", "status": status,
            "suggested_subcaption": panel["suggested_subcaption"],
            "formats": checked_formats(stem, "supplementary"),
            "plot_script": str(supplement_script.resolve()),
            "plot_script_sha256": panel_manifest["plot_script_sha256"],
            "reference_style": str(STYLE), "reference_style_sha256": style_hash,
            "source_audit": str(original_audit.resolve()), "source_audit_sha256": checked_hash(original_audit),
            "single_panel_audit": str(panel_manifest_path.resolve()),
            "single_panel_audit_sha256": checked_hash(panel_manifest_path),
            "run_ids": run_ids, "seeds": supplementary_seeds(stem),
            "rounds": GROUPS[group]["rounds"], "policy_version": GROUPS[group]["version"],
            "metric_definition": f"{transform} [{units}]", "measurement_boundary": GROUPS[group]["boundary"],
            "interpretation_limit": panel["note"], "source_files_sha256": sources,
            "source_recheck_state": source_rechecks[stem],
            "source_chain_note": source_chain_note,
            "plotted_value_files": checked_values(stem, GROUPS[group]["value_files"]),
            "source_fields": fields, "plot_transformation": transform, "units": units,
        })

    if errors:
        raise SystemExit("\n".join(errors))
    if len(all_entries) != 33:
        raise ValueError(f"Expected 33 final graphs, found {len(all_entries)}")
    manifest = {
        "scope": "33 existing-evidence single-graph outputs; no new training, scoring, or device run",
        "style_sha256": style_hash,
        "style_recheck_state": style_check,
        "figure_count": len(all_entries),
        "main_count": 9, "supplementary_count": 24,
        "figures": all_entries,
    }
    (ROOT / "figure_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    mapping_path = ROOT / "figure_data_mapping.csv"
    with mapping_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Figure", "PDF file", "Plot script", "Run IDs", "Source files and SHA-256", "Source fields", "Plot transformation", "Units", "Measurement boundary", "Interpretation limit"))
        for entry in all_entries:
            writer.writerow((entry["figure_label"], entry["formats"]["pdf"], entry["plot_script"],
                             json.dumps(entry["run_ids"]), json.dumps(entry["source_files_sha256"], sort_keys=True),
                             entry["source_fields"], entry["plot_transformation"], entry["units"],
                             entry["measurement_boundary"], entry["interpretation_limit"]))
    print(f"Wrote exact provenance and one mapping row for each of {len(all_entries)} final graphs")


def main() -> None:
    """Public entry point for the final one-graph-per-file package."""
    final_main()


if __name__ == "__main__":
    main()
