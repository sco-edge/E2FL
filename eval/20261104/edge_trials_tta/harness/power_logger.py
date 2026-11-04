#!/usr/bin/env python3
"""Standalone absolute-time power logger for LoRACast TTA runs.

Runs independently from Flower.  Output power is always in W and time is
absolute Unix time, so the trace can later be aligned with server round-end
timestamps.

Supported sensors:
  jetson : INA3221 sysfs input rail
  rpi5   : `vcgencmd pmic_read_adc`, sum matching current*voltage rails
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

STOP = False

def _sig_handler(signum, frame):
    global STOP
    STOP = True

signal.signal(signal.SIGTERM, _sig_handler)
signal.signal(signal.SIGINT, _sig_handler)

def _read_float(path: str):
    try:
        return float(Path(path).read_text().strip())
    except Exception:
        return None

def _jetson_paths():
    voltage_candidates = sorted(
        glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in1_input")
        + glob.glob("/sys/class/hwmon/hwmon*/in1_input")
    )
    current_candidates = sorted(
        glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/curr1_input")
        + glob.glob("/sys/class/hwmon/hwmon*/curr1_input")
    )
    for v in voltage_candidates:
        c = str(Path(v).with_name("curr1_input"))
        if Path(c).exists() and Path(v).with_name("in1_label").exists() and Path(v).with_name("in1_label").read_text().strip() == "VDD_IN":
            return v, c
    raise RuntimeError("INA3221 VDD_IN rail not found; refusing an unidentified rail")

def read_jetson_power_w():
    vpath, cpath = read_jetson_power_w.paths
    mv = _read_float(vpath)
    ma = _read_float(cpath)
    if mv is None or ma is None:
        return None
    return (mv / 1000.0) * (ma / 1000.0)
read_jetson_power_w.paths = (None, None)

def parse_pmic(text: str):
    currents, voltages = {}, {}
    for raw in text.splitlines():
        line = raw.strip()
        if "=" not in line:
            continue
        left, raw_value = line.split("=", 1)
        value = raw_value.strip()
        if len(value) < 2 or value[-1] not in ("A", "V"):
            continue
        try:
            number = float(value[:-1])
        except ValueError:
            continue
        fields = left.split()
        if len(fields) < 2:
            continue
        rail_with_kind, measurement = fields[0], fields[1]
        if rail_with_kind.endswith("_A"):
            rail = rail_with_kind[:-2]
        elif rail_with_kind.endswith("_V"):
            rail = rail_with_kind[:-2]
        else:
            continue
        if measurement.startswith("current") and value[-1] == "A":
            currents[rail] = number
        elif measurement.startswith("volt") and value[-1] == "V":
            voltages[rail] = number
    return currents, voltages

def read_rpi_power_w():
    try:
        cp = subprocess.run(
            (["sudo", "-n"] if os.environ.get("TTA_PMIC_SUDO") == "1" else []) + ["vcgencmd", "pmic_read_adc"],
            capture_output=True, text=True, timeout=5, check=True
        )
    except Exception:
        return None
    currents, voltages = parse_pmic(cp.stdout)
    common = currents.keys() & voltages.keys()
    if not common:
        return None
    return sum(currents[r] * voltages[r] for r in common)

def read_rpi_temp():
    try:
        s = subprocess.check_output(
            ["vcgencmd", "measure_temp"], text=True, timeout=2
        ).strip()
        return float(s.split("=")[1].split("'")[0])
    except Exception:
        return None

def read_rpi_throttled():
    try:
        s = subprocess.check_output(
            ["vcgencmd", "get_throttled"], text=True, timeout=2
        ).strip()
        return s.split("=", 1)[1] if "=" in s else s
    except Exception:
        return ""

def read_jetson_temp():
    vals = []
    for p in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        x = _read_float(p)
        if x is not None:
            vals.append(x / 1000.0 if x > 1000 else x)
    return max(vals) if vals else None

def probe(device: str):
    if device == "jetson":
        paths = _jetson_paths()
        read_jetson_power_w.paths = paths
        p = read_jetson_power_w()
        return {"device": device, "power_w": p, "paths": paths}
    if device == "rpi5":
        return {
            "device": device,
            "power_w": read_rpi_power_w(),
            "temperature_c": read_rpi_temp(),
            "throttled": read_rpi_throttled(),
        }
    raise ValueError(device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["jetson", "rpi5"], required=True)
    ap.add_argument("--output")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--meta")
    args = ap.parse_args()

    if args.probe:
        try:
            info = probe(args.device)
            print(json.dumps(info))
            if info.get("power_w") is None:
                return 2
            return 0
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}))
            return 2

    if not args.output:
        ap.error("--output is required unless --probe is used")

    if args.device == "jetson":
        read_jetson_power_w.paths = _jetson_paths()

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta).expanduser() if args.meta else out.with_suffix(".meta.json")

    start_wall = time.time()
    start_mono = time.monotonic()
    count = 0
    errors = 0
    last_aux = 0.0

    meta = {
        "hostname": socket.gethostname(),
        "device": args.device,
        "start_unix_s": start_wall,
        "start_utc": datetime.fromtimestamp(start_wall, timezone.utc).isoformat(),
        "interval_s": args.interval,
        "pid": os.getpid(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    with out.open("w", newline="", buffering=1) as f:
        w = csv.writer(f)
        w.writerow([
            "unix_time_s", "iso_time_utc", "monotonic_s",
            "power_w", "temperature_c", "throttled"
        ])
        next_tick = time.monotonic()
        while not STOP:
            now_wall = time.time()
            now_mono = time.monotonic()
            if args.device == "jetson":
                power = read_jetson_power_w()
            else:
                power = read_rpi_power_w()

            temp = ""
            throttled = ""
            if now_mono - last_aux >= 10.0:
                last_aux = now_mono
                if args.device == "jetson":
                    temp = read_jetson_temp()
                else:
                    temp = read_rpi_temp()
                    throttled = read_rpi_throttled()

            if power is None or not math.isfinite(power) or power < 0:
                errors += 1
            else:
                w.writerow([
                    f"{now_wall:.6f}",
                    datetime.fromtimestamp(now_wall, timezone.utc).isoformat(),
                    f"{now_mono - start_mono:.6f}",
                    f"{power:.9f}",
                    "" if temp is None else temp,
                    throttled,
                ])
                count += 1

            next_tick += max(args.interval, 0.05)
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

    end_wall = time.time()
    meta.update({
        "end_unix_s": end_wall,
        "end_utc": datetime.fromtimestamp(end_wall, timezone.utc).isoformat(),
        "samples": count,
        "read_errors": errors,
    })
    meta_path.write_text(json.dumps(meta, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
