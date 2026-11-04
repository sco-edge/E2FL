# server_submodel_planner.py

import json
import os
from typing import List, Dict

from predictor.Training_Time_Estimator import Training_Time_Estimator


class SubmodelPlanner:
    """
    fine mode용 서버 side submodel generator + T_single calculator.
    """

    def __init__(self, device_name: str, model_name: str, profile_root: str = None):
        # e.g., ~/EEFL/E2FL/predictor/profile/OrinNX/
        if profile_root is None:
            home = os.path.expanduser("~")
            profile_root = f"{home}/EEFL/E2FL/predictor/profile/{device_name}"

        self.profile_root = profile_root
        self.model_name = model_name
        self.device_name = device_name

        # β 파일 로드
        beta_path = f"{profile_root}/{model_name}_betas_fine.json"
        with open(beta_path, "r") as f:
            self.betas = json.load(f)

    # --------------------------------------------------
    # 1. Submodels 후보 생성 (예시)
    # --------------------------------------------------
    def generate_submodel_candidates(self) -> List[Dict]:
        """
        네가 정의하는 서브모델 조합을 서버에서 만들어냄.
        아래는 width multiplier 예시.
        """
        candidates = []

        # 예시: MobileNetV2 width-multiplier 기반 submodel
        width_list = [1.0, 0.75, 0.5]   # 네가 정의한 조합

        for wid in width_list:
            wid_str = str(wid).replace(".", "_")
            sm_id = f"wm_{wid_str}"

            # workload 파일 경로
            wl_path = f"{self.profile_root}/workload_{sm_id}.json"

            if not os.path.exists(wl_path):
                print(f"[WARN] workload file not found: {wl_path}, skipping")
                continue

            candidates.append({
                "id": sm_id,
                "width_mult": wid,
                "workload_path": wl_path,
            })

        return candidates

    # --------------------------------------------------
    # 2. Single-pass latency 계산
    # --------------------------------------------------
    def compute_T_single_for_submodels(self, candidates: List[Dict]) -> List[Dict]:
        """
        각 submodel의 workload_fine.json을 로딩 후,
        Training_Time_Estimator로 T_single 계산.
        """
        results = []

        for sm in candidates:
            with open(sm["workload_path"], "r") as f:
                workload = json.load(f)

            algo_sel = workload["algo_selection"]
            C_key = workload["C_key"]
            C_non = workload["C_non"]
            model_size = workload.get("model_size", None)

            estimator = Training_Time_Estimator()
            estimator.load_profiled_betas(self.betas)

            # LATTE: T_single
            T_single = estimator.estimate_single_pass(
                algo_selection=algo_sel,
                C_key=C_key,
                C_non=C_non
            )

            results.append({
                "id": sm["id"],
                "width_mult": sm["width_mult"],
                "T_single": T_single,
                "model_size": model_size,
                "workload_path": sm["workload_path"],
            })

        return results

    # --------------------------------------------------
    # 3. 클라이언트가 쓰도록 export
    # --------------------------------------------------
    def export_metadata_for_clients(self, results: List[Dict]) -> Dict:
        """
        Flower server strategy에서 broadcast_config에 넣을 수 있는 형태로 변환.
        """
        return {
            "candidate_submodels": [
                {
                    "id": r["id"],
                    "width_mult": r["width_mult"],
                    "T_single": r["T_single"],
                    "model_size": r["model_size"],
                }
                for r in results
            ]
        }
