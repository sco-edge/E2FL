"""Flower message and state tests for the Nano physical integration adapter."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import torch
from flwr.app import ArrayRecord, ConfigRecord, MetricRecord, RecordDict

from e2fl.strategy.LoRACast import LoRACastCommAware
from nano_final_exposure import NanoFinalExposure
from policy_resume import load_completed_round, save_completed_round


class Grid:
    def get_node_ids(self):
        return [1, 2]


class NanoFinalExposureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def make_strategy(self, cumulative):
        s = NanoFinalExposure(
            exposure_max=2, nx_eligible_unique=128,
            nano_config={"reference-nano-samples": 4, "reference-nano-steps": 4,
                         "reference-nano-correction-max-samples": 4,
                         "reference-nano-correction-max-steps": 4,
                         "reference-nano-correction-max-seconds": 32.0},
            fraction_train=1.0, fraction_evaluate=0.0, min_available_nodes=2,
            log_dir="isolated", ROOT=str(self.root), experiment_id="",
            aggregation_mode="processed_samples", enable_rpi_only_rounds=True,
            use_token_ratio=True, rebalancing_enabled=False,
        )
        s.nx_cumulative_samples = cumulative
        s.jetson_base_quota = 8
        s.jetson_sample_step = 4
        s.jetson_max_quota = 32
        s.policy_warmup_rounds = 1
        s.cid_to_device_uid = {1: "nano", 2: "nx", "1": "nano", "2": "nx"}
        s._known_client_device_map = lambda: {
            "nano": "jetson_orin_nano_super", "nx": "jetson_orin_nx"}
        s._predict_train_time = lambda device, samples, config: (
            32.0 if device == "jetson_orin_nano_super" else float(samples))
        s._select_jetson_workload_for_window = lambda ref, config: (32, 0, 24.0)
        return s

    def configure(self, s):
        config = ConfigRecord({
            "batch_size": 1, "max_seq_len": 64, "round_mode": "BOTH",
            "reference-nano-samples": 4, "reference-nano-steps": 4,
            "reference-nano-correction-max-samples": 4,
            "reference-nano-correction-max-steps": 4,
            "reference-nano-correction-max-seconds": 32.0,
        })
        arrays = ArrayRecord(torch_state_dict=OrderedDict({
            "adapter": torch.tensor([0.0])}))
        with patch.object(LoRACastCommAware, "update_train_config", lambda self, r, c: c):
            messages = list(s.configure_train(3, arrays, config, Grid()))
        return messages, s.round_configs[3]

    def test_partial_native_quota_and_actual_weights(self):
        s = self.make_strategy(232)
        messages, cfg = self.configure(s)
        self.assertEqual({m.metadata.dst_node_id for m in messages}, {1, 2})
        self.assertEqual(cfg["max_samples_per_round_jetson_orin_nx"], 24)
        self.assertEqual(s.exposure_decisions[3]["action"],
                         "partial_or_bounded_epoch_quota")
        replies = []
        for msg in messages:
            nano = msg.metadata.dst_node_id == 1
            n, value = (4, 1.0) if nano else (24, 5.0)
            device = "jetson_orin_nano_super" if nano else "jetson_orin_nx"
            uid = "nano" if nano else "nx"
            record = MetricRecord({
                "num-examples": n, "dataset_size": 128, "partition_id": 0 if nano else 1,
                "local_steps": n, "input_tokens": 64*n, "supervised_tokens": 64*n,
                "train_loss": 1.0, "train_time": 1.0, "train_energy": 1.0,
                "eligible_unique_samples": 128,
                "visited_sample_ids": list(range(n)),
                "device_name_chars": [ord(c) for c in device],
                "device_uid_chars": [ord(c) for c in uid],
            })
            adapter = ArrayRecord(torch_state_dict=OrderedDict({
                "adapter": torch.tensor([value])}))
            replies.append(msg.create_reply(RecordDict({
                "arrays": adapter, "metrics": record})))
        arrays, _ = s.aggregate_train(3, replies)
        self.assertAlmostEqual(arrays.to_torch_state_dict()["adapter"].item(),
                               (4*1.0 + 24*5.0)/28, places=5)
        self.assertEqual(s.nx_cumulative_samples, 256)
        self.assertEqual(s.nx_unique_visited, 24)

    def test_omission_precedes_training_and_nano_only_aggregates(self):
        s = self.make_strategy(256)
        messages, cfg = self.configure(s)
        self.assertEqual([m.metadata.dst_node_id for m in messages], [1])
        self.assertEqual(cfg["round_mode"], "PROTECTED_ONLY")
        self.assertEqual(s.expected_train_replies_by_round[3], 1)
        metric = MetricRecord({
            "num-examples": 4, "dataset_size": 128, "partition_id": 0,
            "local_steps": 4, "input_tokens": 256, "supervised_tokens": 256,
            "train_loss": 1.0, "train_time": 1.0, "train_energy": 1.0,
            "device_name_chars": [ord(c) for c in "jetson_orin_nano_super"],
            "device_uid_chars": [ord(c) for c in "nano"],
        })
        adapter = ArrayRecord(torch_state_dict=OrderedDict({
            "adapter": torch.tensor([2.0])}))
        reply = messages[0].create_reply(RecordDict({
            "arrays": adapter, "metrics": metric}))
        arrays, _ = s.aggregate_train(3, [reply])
        self.assertEqual(arrays.to_torch_state_dict()["adapter"].item(), 2.0)
        self.assertEqual(s.nx_cumulative_samples, 256)

    def test_round_boundary_resume_keeps_exposure_and_skips_prior_rounds(self):
        s = self.make_strategy(232)
        self.configure(s)
        s.round_stats[3] = {"committed": True}
        checkpoint = self.root / "adapter_3.pt"
        torch.save({"adapter": torch.tensor([3.0])}, checkpoint)
        for key in ("_known_client_device_map", "_predict_train_time",
                    "_select_jetson_workload_for_window"):
            del s.__dict__[key]
        save_completed_round(s, 3, str(checkpoint))
        restored = self.make_strategy(0)
        load_completed_round(restored, 3, str(checkpoint))
        self.assertEqual(restored.nx_cumulative_samples, 232)
        with patch("nano_final_exposure.run_remaining_rounds", return_value="resumed") as run:
            actual = restored.start(grid=None, initial_arrays=None, num_rounds=4,
                                    timeout=1.0, train_config=ConfigRecord())
        self.assertEqual(actual, "resumed")
        self.assertEqual(run.call_args.kwargs["completed_round"], 3)


if __name__ == "__main__":
    unittest.main()
