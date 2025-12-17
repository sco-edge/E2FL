"""
latte_predictor.py
------------------
Fine-mode LATTE training-time predictor.

This module provides `LatteFinePredictor`, a lightweight wrapper that
approximates LATTE's Training_Time_Estimator behavior for fine mode.

Given:
  - device-specific β coefficients (flattened into a single dict), and
  - a submodel description (algo_selection, C_key, C_non, param_size_bytes,
    plus an optional per-layer window/mask),

it computes:
  - T_single : single forward+backward pass latency
  - T_train  : total training time for (epochs, batch_size, num_batches)
  - T_comm   : communication time given bandwidth and model size
  - T_total  : T_train + T_comm

The "submodel" argument can be either a plain dict or an object exposing
these attributes:
  - algo_selection: List[str]
  - C_key: List[float]
  - C_non: float
  - param_size_bytes: float
  - optional window/mask/width_factors: List[float]

Notes
-----
* We assume `betas` is already device-specific (i.e., one predictor
  instance per device). The predictor does not switch on `device_name`;
  that argument is accepted only for interface compatibility.
* Non-key FLOPs are scaled by a single scalar `beta_nonkey`.
* Window/mask (if provided) scales per-layer FLOPs before latency
  estimation, enabling width-scaled submodels.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union


class LatteFinePredictor:
    """Fine-mode LATTE predictor.

    Parameters
    ----------
    betas:
        A dict mapping algorithm keys (e.g., "conv3x3_fwd_algo0") to
        latency coefficients β (seconds per FLOP).
    beta_nonkey:
        Scalar coefficient for non-key FLOPs. Defaults to 1.0, meaning
        `C_non` is directly added as latency units.
    residual_learner:
        Optional second-stage model for residual correction. If provided
        and `use_residual=True`, its prediction is added to `T_train`.
    """

    def __init__(
        self,
        betas: Dict[str, float],
        beta_nonkey: float = None,
        residual_learner: Optional[object] = None,
    ) -> None:

        # Auto-detect fine JSON structure
        if isinstance(betas, dict) and "key_fwd" in betas and "key_bwd" in betas:
            flat_betas, beta_nonkey_val = self._flatten_fine_betas(betas)
            self.betas = flat_betas
            self.beta_nonkey = float(beta_nonkey_val)
        else:
            self.betas = betas
            self.beta_nonkey = float(beta_nonkey or 1.0)

        self.residual_learner = residual_learner

    def _flatten_fine_betas(self, betas_json):
        key_fwd = betas_json.get("key_fwd", {})
        key_bwd = betas_json.get("key_bwd", {})

        non_fwd = float(betas_json.get("non_fwd", 0.0))
        non_bwd = float(betas_json.get("non_bwd", 0.0))
        beta_nonkey = non_fwd + non_bwd

        flat = {}
        for fwd_key, fwd_val in key_fwd.items():
            algo_suffix = fwd_key.split("_fwd_")[-1]
            prefix = fwd_key.split("_fwd_")[0]

            bwd_data_key = f"{prefix}_bwd_data_{algo_suffix}"
            bwd_filter_key = f"{prefix}_bwd_filter_{algo_suffix}"

            bwd_data = key_bwd.get(bwd_data_key, 0.0)
            bwd_filter = key_bwd.get(bwd_filter_key, 0.0)

            flat[fwd_key] = fwd_val + bwd_data + bwd_filter

        return flat, beta_nonkey

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_attr_or_key(self, obj: Union[object, dict], key: str, default=None):
        """Utility: attribute-or-dict-key accessor."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _extract_submodel_info(
        self, submodel: Union[object, dict]
    ) -> Tuple[List[str], List[float], float, float, Optional[List[float]]]:
        """Extract required fields from a submodel.

        Returns
        -------
        algo_sel : List[str]
        C_key    : List[float]
        C_non    : float
        param_sz : float
        window   : Optional[List[float]]
        """

        algo_sel = self._get_attr_or_key(submodel, "algo_selection")
        C_key = self._get_attr_or_key(submodel, "C_key")
        C_non = self._get_attr_or_key(submodel, "C_non")
        param_size = self._get_attr_or_key(submodel, "param_size_bytes")

        # Try window/mask keys in order
        window_mask: Optional[List[float]] = None
        for candidate in ("window", "windows", "mask", "width_factors"):
            wm = self._get_attr_or_key(submodel, candidate, None)
            if wm is not None:
                window_mask = wm
                break

        if algo_sel is None or C_key is None or C_non is None or param_size is None:
            raise ValueError("Submodel is missing required fields (algo_selection, C_key, C_non, param_size_bytes)")

        if len(algo_sel) != len(C_key):
            raise ValueError("algo_selection and C_key must have the same length")

        return list(algo_sel), list(C_key), float(C_non), float(param_size), window_mask

    def _apply_window_mask(
        self,
        C_key: List[float],
        C_non: float,
        window: Optional[List[float]],
    ) -> Tuple[List[float], float]:
        """Apply per-layer window/mask to FLOPs.

        If `window` is provided, each key-layer FLOP is scaled by the
        corresponding mask entry. Non-key FLOPs are scaled by the
        average mask (a simple but effective approximation for width
        scaling).
        """

        if window is None:
            return C_key, C_non

        if len(window) != len(C_key):
            raise ValueError("window/mask length must match C_key length")

        C_key_eff = [c * m for c, m in zip(C_key, window)]
        if len(window) > 0:
            avg_scale = sum(window) / float(len(window))
        else:
            avg_scale = 1.0
        C_non_eff = C_non * avg_scale
        return C_key_eff, C_non_eff

    def _single_pass_latency(
        self,
        algo_sel: List[str],
        C_key_eff: List[float],
        C_non_eff: float,
        use_residual: bool,
        submodel_for_residual: Union[object, dict],
    ) -> float:
        """Compute T_single for one forward+backward pass.

        This is a simple linear model:

            T_single = sum_i β[algo_i] * FLOPs_i + β_nonkey * C_non

        plus an optional residual correction.
        """

        if len(algo_sel) != len(C_key_eff):
            raise ValueError("algo_selection and C_key_eff must have the same length")

        latency = self.beta_nonkey * C_non_eff
        for k, flops in zip(algo_sel, C_key_eff):
            beta = self.betas.get(k, 0.0)
            latency += beta * flops

        if use_residual and self.residual_learner is not None:
            # Residual learner is expected to operate in the same units
            # as T_single. The concrete implementation is left to the
            # caller.
            try:
                residual = float(self.residual_learner.predict(submodel_for_residual))
                latency += residual
            except Exception:
                # Fail-safe: ignore residual if anything goes wrong
                pass

        return latency

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        submodel: Union[object, dict],
        B_bytes_per_sec: float,
        num_epochs: int,
        batch_size: int,
        num_batches: int,
        device_name: Optional[str] = None,
        use_residual: bool = True,
    ) -> Dict[str, float]:
        """
        Estimate training + communication latency for a submodel.

        Assumptions
        -----------
        * C_key / C_non in the submodel are FLOPs for batch_size=1.
        * We scale FLOPs linearly with the runtime batch_size.
        * T_single is defined as the latency of one forward+backward
          pass of a single mini-batch with the given batch_size.
        """

        algo_sel, C_key, C_non, param_size, window = self._extract_submodel_info(submodel)

        # 1) width/window 반영 (여전히 per-sample 기준 FLOPs)
        C_key_eff_sample, C_non_eff_sample = self._apply_window_mask(C_key, C_non, window)

        # 2) runtime batch_size 반영: per-batch FLOPs로 스케일
        bs = max(1, int(batch_size))
        C_key_eff_batch = [c * bs for c in C_key_eff_sample]
        C_non_eff_batch = C_non_eff_sample * bs

        # 3) Single-pass latency (one mini-batch)
        T_single = self._single_pass_latency(
            algo_sel=algo_sel,
            C_key_eff=C_key_eff_batch,
            C_non_eff=C_non_eff_batch,
            use_residual=use_residual,
            submodel_for_residual=submodel,
        )

        # 4) Training latency: epochs × batches_per_epoch
        n_epochs = max(1, int(num_epochs))
        n_batches = max(1, int(num_batches))
        total_steps = n_epochs * n_batches
        T_train = T_single * float(total_steps)

        # 5) Communication latency: 모델 파라미터 업로드 시간
        B = max(float(B_bytes_per_sec), 1e-9)  # avoid division by zero
        T_comm = float(param_size) / B

        T_total = T_train + T_comm

        return {
            "T_single": T_single,
            "T_train": T_train,
            "T_comm": T_comm,
            "T_total": T_total,
            "latency": T_single,  # backward compatibility
            "param_size_bytes": float(param_size),
        }
