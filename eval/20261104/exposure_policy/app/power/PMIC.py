"""Raspberry Pi 5 PMIC power monitor for LoRACast.

The monitor invokes the Raspberry Pi firmware command ``vcgencmd
pmic_read_adc`` and parses its textual output.  No third-party PMIC source code
is embedded in this implementation.
"""

from __future__ import annotations

import csv
from datetime import datetime
import logging
import subprocess
import threading
import time

from power._power_monitor_interface import PowerMonitor


class PMICMonitor(PowerMonitor):
    def __init__(self, command: tuple[str, ...] = ("vcgencmd", "pmic_read_adc")):
        super().__init__("PMIC")
        self.command = tuple(command)

    @staticmethod
    def _parse_adc_output(text: str) -> tuple[dict[str, float], dict[str, float]]:
        currents: dict[str, float] = {}
        voltages: dict[str, float] = {}

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or "=" not in line:
                continue

            left, raw_value = line.split("=", 1)
            value_text = raw_value.strip()
            if not value_text:
                continue

            # Firmware values end with A or V.
            unit = value_text[-1]
            if unit not in {"A", "V"}:
                continue

            try:
                value = float(value_text[:-1])
            except ValueError:
                continue

            fields = left.split()
            if len(fields) < 2:
                continue

            rail_with_kind = fields[0]
            measurement = fields[1]

            if rail_with_kind.endswith("_A"):
                rail = rail_with_kind[:-2]
            elif rail_with_kind.endswith("_V"):
                rail = rail_with_kind[:-2]
            else:
                continue

            if measurement.startswith("current") and unit == "A":
                currents[rail] = value
            elif measurement.startswith("volt") and unit == "V":
                voltages[rail] = value

        return currents, voltages

    def _read_power(self, timeout: float = 5.0) -> float | None:
        try:
            completed = subprocess.run(
                self.command,
                capture_output=True,
                text=True,
                check=True,
                timeout=float(timeout),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logging.error("PMIC: command failed: %s", exc)
            return None

        currents, voltages = self._parse_adc_output(completed.stdout)
        common_rails = currents.keys() & voltages.keys()
        if not common_rails:
            logging.error("PMIC: no matching current/voltage rail pairs found")
            return None

        # A * V = W.  This preserves the numeric convention used by the
        # existing LoRACast measurement pipeline.
        return float(
            sum(currents[rail] * voltages[rail] for rail in common_rails)
        )

    def read_power(self) -> float | None:
        return self._read_power()

    def read_power_avg(self) -> float:
        with self.lock:
            samples = list(self.power_data)
        if not samples:
            return 0.0
        return float(sum(power for _, power in samples) / len(samples))

    def _monitor(self):
        while self.monitoring:
            started = self.start_time
            if started is None:
                break

            elapsed = (datetime.now() - started).total_seconds()
            power = self.read_power()
            if power is not None:
                with self.lock:
                    self.power_data.append((elapsed, power))
            time.sleep(max(float(self.freq), 0.001))

    def get_clock(self, name: str):
        try:
            completed = subprocess.run(
                ("vcgencmd", "measure_clock", str(name)),
                capture_output=True,
                text=True,
                check=True,
                timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None

        _, separator, value = completed.stdout.partition("=")
        if not separator:
            return None
        value = value.strip()
        return value if value.isdigit() else None

    def start(self, freq: float):
        with self.lock:
            if self.monitoring:
                return

            self.freq = float(freq)
            self.power_data = []
            self.start_time = datetime.now()
            self.monitoring = True

        self.thread = threading.Thread(
            target=self._monitor,
            name="loracast-pmic",
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        with self.lock:
            if not self.monitoring:
                return None, None
            self.monitoring = False
            started = self.start_time

        thread = self.thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(2.0, 2.0 * float(self.freq)))

        self.end_time = datetime.now()
        elapsed = (
            (self.end_time - started).total_seconds()
            if started is not None
            else 0.0
        )
        with self.lock:
            size = len(self.power_data)
        return elapsed, size

    def save(self, filepath):
        with self.lock:
            rows = list(self.power_data)
            started = self.start_time

        with open(filepath, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["start_time", started.isoformat() if started else ""])
            writer.writerow(["timestamp_s", "power_w"])
            writer.writerows(rows)

    def close(self):
        if self.monitoring:
            self.stop()
