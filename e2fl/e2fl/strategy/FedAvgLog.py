# e2fl/strategy/FedAvgLog.py
import csv, os, time, json
from collections.abc import Iterable
from logging import INFO, WARNING
from typing import Callable, Optional
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.result import Result
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    aggregate_metricrecords,
    sample_nodes,
    validate_message_reply_consistency,
)
from flwr.server import Grid
from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
    log,
)
from sklearn.linear_model import SGDRegressor
import numpy as np
from e2fl.LATTE.Training_Time_Estimator import Training_Time_Estimator


class FedAvgLog(FedAvg):
    """FedAvg strategy that logs unified client metrics (train + evaluate)."""

    def __init__(self, log_dir="logs_fedavg", model_name=None, latte=False, algo_selection=None,
                 model_info=None, device_lookup=None, ROOT="/home/wwjang/EEFL/E2FL",**kwargs):
        super().__init__(**kwargs)
        self.log_dir = ROOT+log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.latte = latte

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.csv_path = os.path.join(self.log_dir, f"fedavg_{timestamp}.csv")

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "round", "phase", "cid",
                "num_examples", "loss", "accuracy", 
                "comp_time", "phase_sent", "phase_recv",
                "comm_time", "comm_sent", "comm_recv",
                "energy", 
                'pred_time', 'latte_residual',
            ])

        if self.latte:
            self.algo_selection = algo_selection
            self.model_info = model_info
            self.model_name = model_name
            
            self.time_residual = SGDRegressor(max_iter=1, warm_start=True)
            self.device_lookup = device_lookup if device_lookup is not None else {1:"RPi5", 2:"jetson_orin_nx"}
            self.estimator = Training_Time_Estimator()

            #ROOT = os.path.dirname(os.path.abspath(__file__))
            profile_root = os.path.join(ROOT, "predictor", "profile")
            self.device_betas = {}
            for _, device_name in self.device_lookup.items():
                profiled_path = f"{profile_root}/{device_name}/{self.model_name}_betas.json"
                with open(profiled_path, "r") as f:
                    self.device_betas[device_name] = json.load(f)

            # ===== LATTE PROFILE DEBUG =====
            try:
                from logging import getLogger
                dbg_logger = getLogger(__name__)
            except Exception:
                dbg_logger = None

            msg = [
                "=== LATTE MODEL PROFILE DEBUG ===",
                f"  type(model_info)  = {type(self.model_info)}",
            ]
            if isinstance(self.model_info, dict):
                msg.append(f"  keys(model_info) = {list(self.model_info.keys())}")
            else:
                msg.append("  model_info is not a dict (cannot show keys)")

            if dbg_logger:
                for line in msg:
                    dbg_logger.info(line)
            else:
                for line in msg:
                    print(line)
            # ================================
        
    def _check_and_log_replies(
        self, replies: Iterable[Message], is_train: bool, validate: bool = True
        ) -> tuple[list[Message], list[Message]]:
        """Check replies for errors and log them.

        Parameters
        ----------
        replies : Iterable[Message]
            Iterable of reply Messages.
        is_train : bool
            Set to True if the replies are from a training round; False otherwise.
            This impacts logging and validation behavior.
        validate : bool (default: True)
            Whether to validate the reply contents for consistency.

        Returns
        -------
        tuple[list[Message], list[Message]]
            A tuple containing two lists:
            - Messages with valid contents.
            - Messages with errors.
        """
        if not replies:
            return [], []

        # Filter messages that carry content
        valid_replies: list[Message] = []
        error_replies: list[Message] = []
        for msg in replies:
            if msg.has_error():
                error_replies.append(msg)
            else:
                valid_replies.append(msg)

        log(
            INFO,
            "%s: Received %s results and %s failures",
            "aggregate_train" if is_train else "aggregate_evaluate",
            len(valid_replies),
            len(error_replies),
        )

        # Log errors
        for msg in error_replies:
            log(
                INFO,
                "\t> Received error in reply from node %d: %s",
                msg.metadata.src_node_id,
                msg.error.reason,
            )

        # Ensure expected ArrayRecords and MetricRecords are received
        if validate and valid_replies:
            validate_message_reply_consistency(
                replies=[msg.content for msg in valid_replies],
                weighted_by_key=self.weighted_by_key,
                check_arrayrecord=is_train,
            )

        return valid_replies, error_replies

    def aggregate_train(self, server_round: int, replies: Iterable[Message]
                    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        print("\n==============================")
        print(f"=== DEBUG aggregate_train (round {server_round}) ===")

        # replies iterator 때문에 리스트화
        replies = list(replies)
        print("Raw replies count:", len(replies))

        # ---------------------------------------------------------
        # 1) Raw replies 내용 덤프 (서버가 실제로 받은 메시지 자체 확인)
        # ---------------------------------------------------------
        for i, msg in enumerate(replies):
            print(f"\n--- DEBUG RAW REPLY [{i}] ---")
            print("has_content:", msg.has_content())
            print("metadata:", msg.metadata)
            print("message_type:", msg.metadata.message_type)

            if msg.has_content():
                print("content keys:", list(msg.content.keys()))

                if "arrays" not in msg.content:
                    print(" !! WARNING: arrays key missing in msg.content")
                else:
                    print("arrays type:", type(msg.content["arrays"]))

                if "metrics" not in msg.content:
                    print(" !! WARNING: metrics key missing in msg.content")
                else:
                    print("metrics type:", type(msg.content["metrics"]))
            else:
                print("NO content available in msg")

        # ---------------------------------------------------------
        # 2) validator 통해 valid / invalid 분리
        # ---------------------------------------------------------
        valid_replies, invalid_replies = self._check_and_log_replies(
            replies, is_train=True
        )

        print("\n=== DEBUG reply validation summary ===")
        print("valid_replies:", len(valid_replies))
        print("invalid_replies:", len(invalid_replies))

        if invalid_replies:
            print("\n--- INVALID REPLIES DETAIL ---")
            for msg, reason in invalid_replies:
                print("metadata:", msg.metadata)
                print("reason:", reason)

        # ---------------------------------------------------------
        # 3) valid replies 없으면 early return
        # ---------------------------------------------------------
        arrays, metrics = None, None
        if not valid_replies:
            print("!!! aggregate_train: NO VALID REPLIES — skipping aggregation !!!")
            return None, None

        # ---------------------------------------------------------
        # 4) 로깅 + 기존 aggregation 실행
        # ---------------------------------------------------------
        print("=== VALID REPLIES FOUND → AGGREGATING ===")
        self._write_metrics("train", server_round, valid_replies)

        reply_contents = [msg.content for msg in valid_replies]

        # Arrays aggregation
        try:
            arrays = aggregate_arrayrecords(
                reply_contents, self.weighted_by_key
            )
            print("aggregate_arrayrecords SUCCESS")
        except Exception as e:
            print("!!! aggregate_arrayrecords FAILED !!!", e)
            arrays = None

        # Metrics aggregation
        try:
            metrics = self.train_metrics_aggr_fn(
                reply_contents, self.weighted_by_key
            )
            print("train_metrics_aggr_fn SUCCESS")
        except Exception as e:
            print("!!! train_metrics_aggr_fn FAILED !!!", e)
            metrics = None

        return arrays, metrics
    
    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message],
            ) -> MetricRecord | None:
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)
        metrics = None
        if valid_replies:
            self._write_metrics("evaluate", server_round, valid_replies)
            
            reply_contents = [msg.content for msg in valid_replies]
            metrics = self.evaluate_metrics_aggr_fn(
                reply_contents, self.weighted_by_key,)
        return metrics

    def _write_metrics(self, phase, server_round, valid_replies):
        print("=== DEBUG _write_metrics ===")
        print("phase:", phase)
        print("num valid_replies:", len(valid_replies))

        now = time.strftime("%Y-%m-%d_%H-%M-%S")
        t_end = time.time()
        rows = []

        for idx, msg in enumerate(valid_replies):
            print(f"\n--- DEBUG msg[{idx}] ---")
            print("has_content:", msg.has_content())

            if not msg.has_content():
                print("  has_content=False → SKIP")
                continue

            content = msg.content
            print("\tcontent keys:", type(content.keys()))
            if "metrics" not in content:
                print("  metrics not found → SKIP")
                continue
                
            metric_record = content["metrics"]     # MetricRecord
            if not hasattr(metric_record, "items"):
                print("  metric_record has no items() → SKIP")
                continue

            m = dict(metric_record.items())        # dict conversion
            print("  METRIC KEYS:", list(m.keys()))

            print("metadata:", msg.metadata)
            cid = msg.metadata.src_node_id
            print("cid extracted:", cid)
            if phase == "train":
                loss = m.get("train_loss", "")
                acc = ""
                t = m.get("train_time", "")
                e = m.get("train_energy", "")
                start_t = m.get("upload_start_time", 0.0)
                comm_t = max(t_end - start_t, 0) if start_t else ""
                phase_sent = m.get("train_sent", "")
                phase_recv = m.get("train_recv", "")
                comm_sent = m.get("update_sent", "")
                comm_recv = m.get("update_recv", "")
                device_name = self.device_lookup.get(m.get("device_code"), "RPi5")
            else:
                loss = m.get("eval_loss", "")
                acc = m.get("eval_accuracy", "")
                t = m.get("eval_time", "")
                e = m.get("eval_energy", "")
                start_t = m.get("update_start_time", 0.0)
                comm_t = max(t_end - start_t, 0) if start_t else ""
                phase_sent = m.get("eval_sent", "")
                phase_recv = m.get("eval_recv", "")
                comm_sent = m.get("upload_sent", "")
                comm_recv = m.get("upload_recv", "")
                device_name = None

            base_row =[
                now, server_round, phase, cid,
                m.get("num-examples", 0),
                loss, acc, 
                t, phase_sent, phase_recv, 
                comm_t, comm_sent, comm_recv,
                e
            ]
            if self.latte and phase == "train":
                print("LATTE mode enabled, applying _latte_update...")
                try:
                    latte_vals = self._latte_update(m, device_name)
                    print("LATTE update success. Output length:", len(latte_vals))
                    rows.append(base_row + latte_vals)
                except Exception as ex:
                    print("!!! LATTE update FAILED !!!", ex)
                    rows.append(base_row)
            else:   
                rows.append(base_row)

        with open(self.csv_path, "a", newline="") as f:
            print("\n=== DEBUG rows before CSV write ===")
            print("rows count:", len(rows))
            if len(rows) > 0:
                print("example row[0]:", rows[0])
            else:
                print("NO rows -> CSV will be empty!")

            csv.writer(f).writerows(rows)

    def _latte_update(self, metrics, device_name):
        # 기본 정보
        flops = metrics.get("flops")
        flops = flops if flops is not None else 0.0
        t_real = metrics.get("train_time")
        update_sent = metrics.get("update_sent", 0)
        update_recv = metrics.get("update_recv", 0)
        bytes_used = update_sent + update_recv

        if self.algo_selection is None:
            print("[LATTE] WARNING: algo_selection is None, skip LATTE")
            return []

        # device betas 로드
        if not device_name or device_name not in self.device_betas:
            print(f"[LATTE] WARNING: unknown device_name={device_name}, skip LATTE")
            return []

        beta = self.device_betas[device_name]
        self.estimator.load_profiled_betas(beta)

        # model_info에서 필요한 키 확인
        if not isinstance(self.model_info, dict):
            print("[LATTE] WARNING: model_info is not dict, skip LATTE")
            return []

        required_keys = ["algo_selection", "C_key", "C_non"]
        missing = [k for k in required_keys if k not in self.model_info]
        if missing:
            print(
                f"[LATTE] WARNING: model_info missing keys {missing}, "
                f"available={list(self.model_info.keys())}. Skip LATTE."
            )
            return []
        
        self.C_key = self.model_info["C_key"]
        self.C_non = self.model_info["C_non"]
        self.algo_selection = self.model_info["algo_selection"]
        t_pred = self.estimator.estimate_single_pass(
            self.algo_selection,
            self.C_key,
            self.C_non
        )

        # residual
        if t_real is None:
            print("[LATTE] WARNING: train_time missing, cannot compute residual")
            return [float(t_pred), float("nan")]

        residual = float(t_real) - float(t_pred)
        
        # ----------------------------
        # Residual learner update
        # ----------------------------
        if hasattr(self, "time_residual"):
            x = np.array([[flops, bytes_used]], dtype=float)
            y = np.array([residual])
            try:
                self.time_residual.partial_train(x, y)
            except:
                pass

        # ----------------------------
        # LATTE 컬럼만 반환
        # ----------------------------
        return [
            float(t_pred),
            float(residual)
        ]
    
    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 36000,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None,
    ):
        log(INFO, "=== FedAvgLog.start(): timeout=%s seconds ===", timeout)

        # 같은 코드
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        arrays = initial_arrays

        # 초기 global eval 동일
        if evaluate_fn:
            res = evaluate_fn(0, arrays)
            result.evaluate_metrics_serverapp[0] = res

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "*** ROUND %s/%s ***", current_round, num_rounds)

            # ============================================================
            # TRAIN CONFIGURE
            # ============================================================
            train_msgs = self.configure_train(
                current_round,
                arrays,
                train_config,
                grid,
            )
            log(INFO, "[DEBUG START] configure_train produced %d messages", len(train_msgs))

            # ============================================================
            # TRAIN SEND & RECEIVE
            # ============================================================
            log(INFO, "[DEBUG START] Waiting for train replies (timeout=%s)", timeout)
            train_replies = grid.send_and_receive(
                messages=train_msgs,
                timeout=timeout,
            )
            log(INFO, "[DEBUG START] train_replies received: %d", len(train_replies))

            # ============================================================
            # AGGREGATE TRAIN
            # ============================================================
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

            if agg_arrays is not None:
                arrays = agg_arrays
                result.arrays = agg_arrays

            if agg_train_metrics is not None:
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # ============================================================
            # EVALUATE CONFIGURE
            # ============================================================
            eval_msgs = self.configure_evaluate(
                current_round,
                arrays,
                evaluate_config,
                grid,
            )
            log(INFO, "[DEBUG START] configure_evaluate produced %d messages", len(eval_msgs))

            # ============================================================
            # EVALUATE SEND & RECEIVE
            # ============================================================
            log(INFO, "[DEBUG START] Waiting for evaluate replies (timeout=%s)", timeout)
            eval_replies = grid.send_and_receive(
                messages=eval_msgs,
                timeout=timeout,
            )
            log(INFO, "[DEBUG START] evaluate_replies received: %d", len(eval_replies))

            # ============================================================
            # AGGREGATE EVALUATE
            # ============================================================
            agg_eval_metrics = self.aggregate_evaluate(
                current_round,
                eval_replies,
            )
            if agg_eval_metrics is not None:
                result.evaluate_metrics_clientapp[current_round] = agg_eval_metrics

            # SERVER-SIDE central eval
            if evaluate_fn:
                res = evaluate_fn(current_round, arrays)
                result.evaluate_metrics_serverapp[current_round] = res

        log(INFO, "=== FedAvgLog.start() FINISHED ===")
        return result
