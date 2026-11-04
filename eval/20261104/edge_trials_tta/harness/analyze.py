#!/usr/bin/env python3
"""LoRACast TTA calibration/SLO locking and official scoring.

No pandas/numpy dependency: Python 3.11+ standard library only.
"""
from __future__ import annotations

import argparse
import bisect
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def read_fedavg(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def parse_server_ts(s, tz_name):
    # Current LoRACast fedavg format: 2026-07-22_13-39-10
    dt = datetime.strptime(s, "%Y-%m-%d_%H-%M-%S")
    return dt.replace(tzinfo=ZoneInfo(tz_name)).timestamp()

def round_end_times(rows, tz_name):
    by_round = {}
    # Prefer evaluate timestamp because it marks completion of the full LLM round.
    for phase in ("evaluate", "train"):
        for row in rows:
            if row.get("phase") != phase:
                continue
            try:
                r = int(float(row["round"]))
                t = parse_server_ts(row["timestamp"], tz_name)
            except Exception:
                continue
            if r not in by_round or phase == "evaluate":
                by_round[r] = max(t, by_round.get(r, t))
        if by_round and phase == "evaluate":
            # Do not mix train timestamps if evaluate timestamps exist.
            rounds_eval = {
                int(float(r["round"])) for r in rows
                if r.get("phase") == "evaluate" and r.get("round")
            }
            if rounds_eval:
                return {r: by_round[r] for r in rounds_eval if r in by_round}
    return by_round

def read_power(path):
    samples = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                t = float(row["unix_time_s"])
                p = float(row["power_w"])
            except Exception:
                continue
            if math.isfinite(t) and math.isfinite(p) and p >= 0:
                samples.append((t, p))
    if any(b[0] <= a[0] for a,b in zip(samples,samples[1:])):
        raise ValueError("nonmonotonic or duplicate power timestamps")
    return samples

def interp(samples, t):
    ts = [x[0] for x in samples]
    j = bisect.bisect_left(ts, t)
    if j == 0:
        if samples and samples[0][0] == t:
            return samples[0][1]
        return None
    if j >= len(samples):
        return None
    t0, p0 = samples[j-1]
    t1, p1 = samples[j]
    if t1 == t0:
        return p1
    a = (t - t0) / (t1 - t0)
    return p0 + a * (p1 - p0)

def integrate(samples, start, end, max_gap):
    if end <= start or len(samples) < 2:
        raise ValueError("invalid integration interval/trace")
    ts = [x[0] for x in samples]
    left = bisect.bisect_left(ts, start)
    right = bisect.bisect_right(ts, end)
    inner = samples[left:right]

    p_start = interp(samples, start)
    p_end = interp(samples, end)
    if p_start is None or p_end is None:
        raise ValueError("power trace does not cover round boundary")

    # Check the original bracketing interval too: clipping a large gap to a
    # short round must never hide missing power measurements.
    brackets = samples[max(0, left-1):min(len(samples), right+1)]
    if any(b[0]-a[0] > max_gap for a,b in zip(brackets, brackets[1:]) if a[0] < end and b[0] > start):
        raise ValueError("power sample gap crossing boundary exceeds limit")
    pts = [(start, p_start)] + inner + [(end, p_end)]
    pts.sort()

    # Remove duplicate timestamps.
    uniq = []
    for t, p in pts:
        if uniq and abs(t - uniq[-1][0]) < 1e-9:
            uniq[-1] = (t, p)
        else:
            uniq.append((t, p))

    gaps = [uniq[i+1][0] - uniq[i][0] for i in range(len(uniq)-1)]
    max_seen_gap = max(gaps) if gaps else float("inf")
    if max_seen_gap > max_gap:
        raise ValueError(
            f"power sample gap {max_seen_gap:.3f}s exceeds {max_gap:.3f}s"
        )

    energy = 0.0
    for (t0, p0), (t1, p1) in zip(uniq, uniq[1:]):
        energy += 0.5 * (p0 + p1) * (t1 - t0)
    return energy, max_seen_gap, len(inner)

def measured_rounds(rows, warmup, requested):
    rounds = sorted({
        int(float(r["round"])) for r in rows
        if r.get("round") and r.get("phase") == "train"
    })
    valid = [r for r in rounds if r > warmup]
    if requested:
        expected = list(range(warmup+1, warmup+requested+1))
        if valid != expected:
            raise ValueError(f"expected contiguous measured rounds {expected}, got {valid}")
    return valid

def energy_observations(rows, traces, tz_name, warmup, requested, max_gap):
    measured = set(range(warmup+1,warmup+requested+1))
    expected = {(r,d) for r in measured for d in traces}
    keys = [(int(row['round']),row.get('device_name')) for row in rows
            if row.get('phase')=='train' and int(row['round']) in measured]
    if set(keys)!=expected or len(keys)!=len(expected):
        raise ValueError('missing/duplicate/unexpected training rows for energy calibration')
    ends = round_end_times(rows, tz_name)
    rounds = measured_rounds(rows, warmup, requested)
    observations = []
    for r in rounds:
        if r not in ends or (r-1 not in ends and not (r==1 and warmup==0)):
            raise RuntimeError(f"missing round boundary for round {r}")
        if r==1 and warmup==0:
            starts=[float(row['prediction_recorded_at']) for row in rows
                    if row.get('phase')=='train' and int(row['round'])==1
                    and row.get('prediction_recorded_at')]
            if len(starts)!=len(traces) or not all(math.isfinite(t) for t in starts):
                raise RuntimeError('missing first-round pre-dispatch energy boundary')
            start=min(starts)
        else:
            start=ends[r-1]
        end=ends[r]
        for dev, trace_path in traces.items():
            samples = read_power(trace_path)
            e, gap, n = integrate(samples, start, end, max_gap)
            observations.append({
                "round": r,
                "device_name": dev,
                "round_start_unix_s": start,
                "round_end_unix_s": end,
                "round_duration_s": end - start,
                "round_energy_j": e,
                "max_power_gap_s": gap,
                "power_samples": n,
            })
    return observations

def time_observations(rows, warmup, requested):
    allowed_rounds = set(measured_rounds(rows, warmup, requested))
    out = []
    for row in rows:
        if row.get("phase") != "train":
            continue
        try:
            r = int(float(row["round"]))
            pred = float(row["pred_train_time_self"])
            actual = float(row["actual_train_time_self"])
        except Exception:
            continue
        if r not in allowed_rounds or not math.isfinite(pred) or not math.isfinite(actual) or pred <= 0 or actual <= 0:
            continue
        err = abs(pred - actual) / actual
        out.append({
            "round": r,
            "device_name": row.get("device_name", ""),
            "pred_train_time_s": pred,
            "actual_train_time_s": actual,
            "relative_error": err,
        })
    return out

def dump_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def lock_slo(args):
    rows = read_fedavg(args.fedavg)
    traces = dict(x.split("=", 1) for x in args.power)
    obs = energy_observations(
        rows, traces, args.timezone, args.warmup,
        args.rounds, args.max_gap
    )
    by_dev = {}
    for o in obs:
        by_dev.setdefault(o["device_name"], []).append(o["round_energy_j"])

    expected = set(traces)
    if set(by_dev) != expected:
        raise SystemExit(f"device mismatch: traces={expected}, observed={set(by_dev)}")
    for d, vals in by_dev.items():
        if len(vals) != args.rounds:
            raise SystemExit(f"{d}: expected {args.rounds} calibration rounds, got {len(vals)}")

    if any(not math.isfinite(v) or v <= 0 for vals in by_dev.values() for v in vals):
        raise ValueError('calibration energy must be positive and finite')
    slo = {d: max(vals) * args.margin for d, vals in by_dev.items()}
    if not math.isfinite(args.margin) or args.margin < 1:
        raise ValueError("SLO margin must be finite and >= 1")
    result = {
        "schema": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "formula": "device_slo_j = max(calibration_whole_round_energy_j) * margin",
        "margin": args.margin,
        "warmup_rounds": args.warmup,
        "measurement_rounds": args.rounds,
        "timezone": args.timezone,
        "slo_joules": slo,
        "calibration_energy_j": by_dev,
        "inputs_sha256": {
            "fedavg": sha256(args.fedavg),
            **{f"power:{d}": sha256(p) for d, p in traces.items()},
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        f.write(json.dumps(result, indent=2, allow_nan=False))
    dump_csv(out.with_name("calibration_energy.csv"), obs)
    print(json.dumps({"slo_joules": slo, "out": str(out)}, indent=2))

def score(args):
    if getattr(args,'accuracy_metric','within_error_limit') != 'within_error_limit':
        raise ValueError('TTA PASS/FAIL requires accuracy_metric=within_error_limit; one_minus_mare is auxiliary only')
    rows = read_fedavg(args.fedavg)
    traces = dict(x.split("=", 1) for x in args.power)
    slo_lock = json.loads(Path(args.slo).read_text())
    slos = {k: float(v) for k, v in slo_lock["slo_joules"].items()}

    t_obs = time_observations(rows, args.warmup, args.rounds)
    e_obs = energy_observations(
        rows, traces, args.timezone, args.warmup,
        args.rounds, args.max_gap
    )
    devices = sorted(traces)
    if set(slos) != set(devices) or any(not math.isfinite(v) or v <= 0 for v in slos.values()):
        raise ValueError("invalid SLO device set or budget")
    expected = {(r,d) for r in range(args.warmup+1,args.warmup+args.rounds+1) for d in devices}
    for observations in (t_obs, e_obs):
        keys=[(o['round'],o['device_name']) for o in observations]
        if set(keys)!=expected or len(keys)!=len(expected):
            raise ValueError("missing, duplicate or unexpected device-round observations")
    audited = bool(getattr(args,'run_dir',None))
    prediction_audit_valid = False
    slo_audit_valid = False
    dataset_independence = None
    if audited:
        run = Path(args.run_dir)
        with (run/'report/tta_prediction_audit.csv').open(newline='') as stream:
            audit_rows=list(csv.DictReader(stream))
        audit_keys=[(int(x['round']),x['device_name']) for x in audit_rows]
        prediction_audit_valid = (set(audit_keys)==expected and len(audit_keys)==len(expected)
                                  and all(x['valid_prediction']=='True' for x in audit_rows))
        if not prediction_audit_valid:
            raise ValueError('pre-dispatch prediction audit missing or invalid; no PASS can be issued')
        dataset_independence=json.loads((run/'report/tta_dataset_independence.json').read_text())
        slo_audit=json.loads((run/'report/tta_slo_audit.json').read_text())
        slo_audit_valid=(all(slo_audit.get(key) is True for key in
                            ('unchanged','formula_matches','calibration_hashes_match',
                             'calibration_protocol_valid','lock_preceded_official'))
                         and slo_audit.get('slo_sha256_after')==sha256(args.slo))
        if not slo_audit_valid:
            raise ValueError('SLO pre-lock audit missing or invalid; no PASS can be issued')

    # Join time + energy by (round, device)
    idx = {(o["round"], o["device_name"]): dict(o) for o in e_obs}
    for t in t_obs:
        key = (t["round"], t["device_name"])
        idx.setdefault(key, {}).update(t)

    final = []
    for key in sorted(idx):
        o = idx[key]
        dev = o.get("device_name", key[1])
        if dev not in slos:
            raise SystemExit(f"SLO missing for device {dev}")
        o["time_pass"] = (
            o.get("relative_error") is not None
            and o["relative_error"] <= args.time_error_limit
        )
        o["energy_slo_j"] = slos[dev]
        o["energy_violation"] = o["round_energy_j"] > slos[dev]
        final.append(o)

    if audited:
        import audit_evidence
        energy_audit=audit_evidence.build_energy_audit(
            args.run_dir,{'suite':{'max_power_sample_gap_s':args.max_gap,
                                  'max_clock_offset_ms':args.clock_tolerance_s*1000}},final)
        if len(energy_audit)!=len(expected) or not all(x['valid_energy'] for x in energy_audit):
            raise ValueError('energy boundary or power trace audit invalid; no PASS can be issued')

    def stats_for(dev=None):
        items = [o for o in final if dev is None or o["device_name"] == dev]
        expected = args.rounds * (len(devices) if dev is None else 1)
        time_valid = [o for o in items if "relative_error" in o]
        energy_valid = [o for o in items if "round_energy_j" in o]
        time_acc = (
            sum(bool(o["time_pass"]) for o in time_valid) / len(time_valid)
            if time_valid else 0.0
        )
        within_limit = time_acc
        mare = statistics.mean(o['relative_error'] for o in time_valid) if time_valid else float('inf')
        viol = (
            sum(bool(o["energy_violation"]) for o in energy_valid) / len(energy_valid)
            if energy_valid else 1.0
        )
        complete = len(items) == expected and len(time_valid) == expected and len(energy_valid) == expected
        return {
            "mare": mare,
            "one_minus_mare_accuracy": 1.0 - mare,
            "one_minus_mare_accuracy_pct": (1.0-mare)*100.0,
            "successful_predictions": sum(bool(o['time_pass']) for o in time_valid),
            "samples_within_error_limit": within_limit,
            "expected_observations": expected,
            "observations": len(items),
            "complete": complete,
            "time_prediction_accuracy": time_acc,
            "time_prediction_accuracy_pct": time_acc * 100.0,
            "energy_violation_rate": viol,
            "energy_violation_rate_pct": viol * 100.0,
            "time_pass": complete and time_acc >= args.time_accuracy_target,
            "energy_pass": complete and viol <= args.energy_violation_target,
        }

    per_device = {d: stats_for(d) for d in devices}
    overall = stats_for(None)

    # Conservative certificate rule: overall AND every physical device must pass.
    pass_all = (
        audited and overall["time_pass"] and overall["energy_pass"]
        and all(x["time_pass"] and x["energy_pass"] for x in per_device.values())
        and dataset_independence['sample_nonoverlap_proven']
    )

    summary = {
        "schema": 1,
        "pass": pass_all,
        "criteria": {
            "accuracy_metric": 'within_error_limit',
            "auxiliary_accuracy_metric": 'one_minus_mare',
            "warmup_rounds_excluded": args.warmup,
            "official_measurement_rounds_per_device": args.rounds,
            "time_error_limit": args.time_error_limit,
            "time_accuracy_target": args.time_accuracy_target,
            "energy_violation_target": args.energy_violation_target,
            "energy_definition": (
                "Integral of declared sensor power (Jetson VDD_IN; RPi matched PMIC rail sum). "
                + ("First round starts at earliest pre-dispatch prediction; subsequent rounds start at previous evaluate completion. "
                   if args.warmup==0 else "Each round starts at previous evaluate completion. ")
                + "Each ends at current evaluate completion; includes training, communication, idle waiting, and client-side round overhead."
            ),
        },
        "slo_lock_sha256": sha256(args.slo),
        "slo_joules": slos,
        "prediction_audit_valid": prediction_audit_valid,
        "slo_audit_valid": slo_audit_valid,
        "audit_status": 'verified' if audited else 'not_audited',
        "dataset_independence": dataset_independence['status'] if dataset_independence else None,
        "dataset_independence_pass": dataset_independence['sample_nonoverlap_proven'] if dataset_independence else None,
        "overall": overall,
        "per_device": per_device,
        "inputs_sha256": {
            "fedavg": sha256(args.fedavg),
            "slo_lock": sha256(args.slo),
            **{f"power:{d}": sha256(p) for d, p in traces.items()},
        },
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dump_csv(outdir / "tta_observations.csv", final)
    (outdir / "tta_summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "LoRACast TTA Harness Result (proposed test criteria)",
        "=" * 35,
        f"OVERALL: {'PASS' if pass_all else 'FAIL'}",
        "",
        f"Time prediction accuracy: {overall['time_prediction_accuracy_pct']:.2f}% "
        f"(target >= {args.time_accuracy_target*100:.1f}%)",
        f"Auxiliary (1-MARE): {overall['one_minus_mare_accuracy_pct']:.2f}%",
        f"Sample independence: {dataset_independence['status'] if dataset_independence else 'not audited'}",
        f"Energy SLO violation rate: {overall['energy_violation_rate_pct']:.2f}% "
        f"(target <= {args.energy_violation_target*100:.1f}%)",
        "",
    ]
    for d in devices:
        s = per_device[d]
        lines += [
            f"[{d}] time accuracy={s['time_prediction_accuracy_pct']:.2f}%, "
            f"energy violation={s['energy_violation_rate_pct']:.2f}%, "
            f"complete={s['complete']}"
        ]
    (outdir / "tta_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if pass_all else 2

def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("lock-slo")
    a.add_argument("--fedavg", required=True)
    a.add_argument("--power", action="append", required=True, help="device_name=trace.csv")
    a.add_argument("--timezone", default="Asia/Seoul")
    a.add_argument("--warmup", type=int, default=2)
    a.add_argument("--rounds", type=int, default=5)
    a.add_argument("--margin", type=float, default=1.10)
    a.add_argument("--max-gap", type=float, default=3.0)
    a.add_argument("--out", required=True)
    a.set_defaults(func=lock_slo)

    a = sub.add_parser("score")
    a.add_argument("--fedavg", required=True)
    a.add_argument("--power", action="append", required=True, help="device_name=trace.csv")
    a.add_argument("--slo", required=True)
    a.add_argument("--timezone", default="Asia/Seoul")
    a.add_argument("--warmup", type=int, default=2)
    a.add_argument("--rounds", type=int, default=10)
    a.add_argument("--time-error-limit", type=float, default=0.10)
    a.add_argument("--time-accuracy-target", type=float, default=0.90)
    a.add_argument("--energy-violation-target", type=float, default=0.10)
    a.add_argument("--max-gap", type=float, default=3.0)
    a.add_argument("--accuracy-metric", choices=["within_error_limit"], default="within_error_limit")
    a.add_argument("--run-dir", help="official attempt containing prediction, dataset, SLO and energy audit logs")
    a.add_argument("--clock-tolerance-s", type=float, default=1.0)
    a.add_argument("--outdir", required=True)
    a.set_defaults(func=score)
    return p

def main():
    p = build_parser()
    args = p.parse_args()
    rv = args.func(args)
    return 0 if rv is None else rv

if __name__ == "__main__":
    raise SystemExit(main())
