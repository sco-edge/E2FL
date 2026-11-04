"""
latte_selector.py
------------------
Layer Algorithm Selector for LATTE fine-mode.

This module wraps the trained MLP selector from LATTE's OSS
(Layer_Algorithm_Selector.py). It performs per-layer kernel
algorithm selection at runtime.

Given:
  - layer_features: numerical feature vectors per key-layer
  - algo_candidates: list of possible CUDA kernel algorithm keys

It outputs:
  - algo_selection: fastest algorithm for each layer
"""

from __future__ import annotations
import torch
from typing import List


class LatteLayerSelector:
    """
    MLP-based layer-algorithm selector for LATTE fine mode.
    """

    def __init__(self, mlp_path: str):
        """
        Parameters
        ----------
        mlp_path : str
            Path to the trained MLP (.pth) produced by
            LATTE's Layer_Algorithm_Selector.py.
        """
        self.model = torch.load(mlp_path, map_location="cpu")
        self.model.eval()

    def pick_algo(self, layer_features: List[float], algo_candidates: List[str]) -> str:
        """
        Predict the best kernel algorithm for a single layer.

        Parameters
        ----------
        layer_features : List[float]
            Example: [Cin, Cout, H, W, Kh, Kw, stride, groups]

        algo_candidates : List[str]
            Example: ["conv3x3_fwd_algo0", "conv3x3_fwd_algo1", "conv3x3_fwd_algo5"]

        Returns
        -------
        str
            The predicted fastest kernel algorithm key.
        """
        x = torch.tensor(layer_features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            scores = self.model(x).flatten()

        idx = int(torch.argmax(scores).item())
        return algo_candidates[idx]

    def pick_for_model(
        self,
        layer_feature_table: List[List[float]],
        algo_candidate_table: List[List[str]],
    ) -> List[str]:
        """
        Predict kernel algorithms for all key layers in a model.

        Parameters
        ----------
        layer_feature_table : List[List[float]]
            Each item is the layer feature vector.

        algo_candidate_table : List[List[str]]
            Each item is the candidate algorithms for that layer.

        Returns
        -------
        List[str]
            Selected algorithm key per layer.
        """
        assert len(layer_feature_table) == len(algo_candidate_table), \
            "layer_feature_table and algo_candidate_table must match in length"

        result = []
        for feat, cand in zip(layer_feature_table, algo_candidate_table):
            result.append(self.pick_algo(feat, cand))
        return result
