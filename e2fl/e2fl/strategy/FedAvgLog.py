# e2fl/strategy/FedAvgLog.py
import csv, os, time
from collections.abc import Iterable
from logging import INFO, WARNING
from typing import Callable, Optional
from flwr.serverapp.strategy import FedAvg
from flwr.server import Grid
from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
)


class FedAvgLog(FedAvg):
    """FedAvg strategy that logs unified client metrics (fit + evaluate)."""

    def __init__(self, log_dir="logs_fedavg", **kwargs):
        super().__init__(**kwargs)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.csv_path = os.path.join(log_dir, f"fedavg_{timestamp}.csv")

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "round", "phase", "cid",
                "num_examples", "loss", "accuracy", 
                "comp_time", "energy",
                "comm_time", "sent", "recv",
            ])

    def aggregate_train(self, server_round: int, replies: Iterable[Message],
                        ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        params, metrics = super().aggregate_train(server_round, replies)
        self._write_metrics("fit", server_round, replies)
        return params, metrics

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message],
                           ) -> Optional[MetricRecord]:
        metrics = super().aggregate_evaluate(server_round, replies)
        self._write_metrics("evaluate", server_round, replies)
        return metrics

    def _write_metrics(self, phase, server_round, results):
        now = time.strftime("%Y-%m-%d_%H-%M-%S")
        t_end = time.time()
        rows = []

        for msg in results:
            # Message 객체에서 client_id와 metrics 꺼내기
            cid = getattr(msg, "node_id", "unknown")  # or msg.metadata["cid"] depending on Flower version
            if hasattr(msg.content, "metrics"):
                m = msg.content.metrics
            elif isinstance(msg.content, dict) and "metrics" in msg.content:
                m = msg.content["metrics"]
            else:
                m = {}

            # phase별 metric 정리
            if phase == "fit":
                loss = m.get("train_loss", "")
                acc = ""
                t = m.get("train_time", "")
                e = m.get("train_energy", "")
                start_t = m.get("upload_start_time", 0.0)
                comm_t = max(t_end - start_t, 0) if start_t else ""
                sent = m.get("update_sent", "")
                recv = m.get("update_recv", "")
            else:  # evaluate
                loss = m.get("eval_loss", "")
                acc = m.get("eval_accuracy", "")
                t = m.get("eval_time", "")
                e = m.get("eval_energy", "")
                start_t = m.get("update_start_time", 0.0)
                comm_t = max(t_end - start_t, 0) if start_t else ""
                sent = m.get("upload_sent", "")
                recv = m.get("upload_recv", "")

            rows.append([
                now, server_round, phase, cid,
                m.get("num-examples", 0),
                loss, acc, t, e,
                comm_t, sent, recv
            ])

        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerows(rows)