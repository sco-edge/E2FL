from logging import WARNING
from typing import Callable, Optional, Union
import math, time
from flwr.serverapp.strategy import FedAvg

class EEFL(FedAvg):
    """Energy-Aware Federated Learning Strategy (EEFL)."""

    def __init__(self,
                 fraction_train: float = 1.0,
                 fraction_evaluate: float = 1.0,
                 min_fit_clients: int = 2,
                 min_evaluate_clients: int = 2,
                 min_available_clients: int = 2,
                 accept_failures: bool = True,
                 fit_metrics_aggregation_fn=None,
                 base_H=4, delta_H=1, alpha=0.5, Lambda_th=1e-4,
                 **kwargs):

        # FedAvg 초기화 (super 호출)
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            accept_failures=accept_failures,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            **kwargs
        )
        
        self.H_i = {}               # the number of local training iterations for mobile device i
        self.prev_loss = {}         # 
        self.prev_energy = {}       #   
        self.base_H = base_H        # the initial value of the number of local iterations
        self.delta_H = delta_H      # the increment unit
        self.alpha = alpha          # non-negative decreasing function of wireless transmission rate
        self.Lambda_th = Lambda_th  # 

    # [1] 기존 FedAvg configure_fit 확장
    def configure_fit(self, server_round, parameters, client_manager):
        cfgs = super().configure_fit(server_round, parameters, client_manager)
        for client, fit_ins in cfgs:
            cid = getattr(client, "cid", None)
            # 이전 라운드에서 계산한 H_i가 있으면 적용
            if cid in self.H_i:
                fit_ins.config["local-epochs"] = self.H_i[cid]
        return cfgs

    # [2] 기존 FedAvg aggregate_fit 확장
    def aggregate_fit(self, server_round: int,
                      results: list[tuple[ClientProxy, FitRes]],
                      failures: list):
        params, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        next_H = {} # H for next round

        upload_end_time = time.time()
        for client, fit_res in results:
            cid = getattr(client, "cid", "unknown")
            m = fit_res.metrics or {}

            # --- metrics에서 에너지/시간/속도 계산 ---
            E = m.get("train_energy", 1.0)
            T = m.get("train_time", 1e-3)
            tx = m.get("train_sent", 0)
            loss = m.get("train_loss", 0)
            upload_start_time = m.get("upload_start_time", 0.0)
            upload_duration = max(upload_end_time - upload_start_time, 1e-3)
            s_rate = tx / upload_duration

            # --- AdaH 정책 계산 ---
            alpha_s = self.alpha * (1 / (1 + s_rate))    # 무선속도 감쇠함수
            prev_E = self.prev_energy.get(cid, E)
            prev_L = self.prev_loss.get(cid, loss)
            Λ_i = abs(loss - prev_L) / max(prev_E, 1)    # 효율 척도

            if Λ_i < self.Lambda_th:
                H_new = self.H_i.get(cid, self.base_H)   # 효율 정체 시 H 유지
            else:
                H_new = self.H_i.get(cid, self.base_H) + alpha_s * self.delta_H

            next_H[cid] = max(1, int(math.ceil(H_new)))
            self.prev_energy[cid] = E
            self.prev_loss[cid] = loss

        self.H_i = next_H

        # --- 서버에서 라운드별 요약 metric으로 반환 가능 ---
        aggregated_metrics.update({
            "policy_next_H": next_H,
            "avg_H": sum(next_H.values()) / len(next_H),
        })

        return params, aggregated_metrics
