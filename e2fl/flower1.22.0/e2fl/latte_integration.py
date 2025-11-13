"""
LATTE (Layer-wise Algorithm Selection for Training Time Estimation) Integration Module
E2FL 시스템에 LATTE를 통합하여 에너지 효율적인 레이어별 알고리즘 선택을 수행합니다.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import os

# LATTE 모듈들 import
from .LATTE.Layer_Algorithm_Selector import Layer_Algorithm_Selector
from .LATTE.Training_Time_Estimator import Training_Time_Estimator

class LATTEIntegration:
    """
    LATTE를 E2FL 시스템에 통합하는 클래스
    """
    
    def __init__(self, latte_data_path: str = None):
        """
        LATTE 통합 초기화
        :param latte_data_path: LATTE ground truth 데이터 경로
        """
        if latte_data_path is None:
            # 현재 디렉토리 기준으로 LATTE 데이터 경로 설정
            current_dir = Path(__file__).parent
            latte_data_path = current_dir / "LATTE" / "ground_truth.csv"
        
        self.latte_data_path = latte_data_path
        self.layer_selector = None
        self.time_estimator = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # LATTE 모델들 초기화
        self._initialize_latte_models()
    
    def _initialize_latte_models(self):
        """LATTE 모델들을 초기화합니다."""
        try:
            # Layer Algorithm Selector 초기화
            self.layer_selector = Layer_Algorithm_Selector().to(self.device)
            self.logger.info("Layer Algorithm Selector initialized successfully")
            
            # Training Time Estimator 초기화
            self.time_estimator = Training_Time_Estimator()
            self.logger.info("Training Time Estimator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LATTE models: {e}")
            raise
    
    def select_optimal_algorithm(self, layer_features: Dict) -> str:
        """
        주어진 레이어 특성에 대해 최적의 알고리즘을 선택합니다.
        :param layer_features: 레이어 특성 딕셔너리
        :return: 선택된 알고리즘 이름
        """
        try:
            # 레이어 특성을 텐서로 변환
            feature_tensor = self._convert_features_to_tensor(layer_features)
            
            # Layer Algorithm Selector로 예측
            with torch.no_grad():
                prediction = self.layer_selector(feature_tensor)
                algorithm_idx = torch.argmax(prediction).item()
            
            # 알고리즘 인덱스를 이름으로 변환
            algorithm_names = ['algorithm_0', 'algorithm_1', 'algorithm_2']
            selected_algorithm = algorithm_names[algorithm_idx]
            
            self.logger.info(f"Selected algorithm: {selected_algorithm}")
            return selected_algorithm
            
        except Exception as e:
            self.logger.error(f"Error in algorithm selection: {e}")
            return 'algorithm_0'  # 기본값 반환
    
    def estimate_training_time(self, model_config: Dict, device_specs: Dict) -> float:
        """
        주어진 모델 설정과 디바이스 사양에 대해 훈련 시간을 추정합니다.
        :param model_config: 모델 설정
        :param device_specs: 디바이스 사양
        :return: 추정된 훈련 시간 (초)
        """
        try:
            # 간단한 추정 로직 (실제로는 더 복잡한 계산 필요)
            model_complexity = model_config.get('model_complexity', 1.0)
            batch_size = model_config.get('batch_size', 32)
            num_epochs = model_config.get('num_epochs', 1)
            
            # 디바이스 성능 고려
            device_factor = 1.0
            if device_specs.get('gpu', False):
                device_factor = 0.3  # GPU 사용 시 3배 빠름
            elif device_specs.get('cpu_cores', 1) > 4:
                device_factor = 0.7  # 멀티코어 CPU 사용 시 1.4배 빠름
            
            estimated_time = model_complexity * batch_size * num_epochs * device_factor
            self.logger.info(f"Estimated training time: {estimated_time:.2f} seconds")
            return estimated_time
            
        except Exception as e:
            self.logger.error(f"Error in training time estimation: {e}")
            return 0.0
    
    def _convert_features_to_tensor(self, features: Dict) -> torch.Tensor:
        """
        레이어 특성 딕셔너리를 텐서로 변환합니다.
        :param features: 레이어 특성 딕셔너리
        :return: 변환된 텐서
        """
        # 예상되는 특성 순서 (ground_truth.csv 기준)
        feature_order = [
            'layer_type', 'input_c', 'input_h', 'input_w', 'output_c',
            'kernel_h', 'kernel_w', 'stride_h', 'stride_w', 'padding_h',
            'padding_w', 'groups', 'bias', 'activation'
        ]
        
        feature_vector = []
        for feature_name in feature_order:
            if feature_name in features:
                feature_vector.append(float(features[feature_name]))
            else:
                feature_vector.append(0.0)  # 기본값
        
        return torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)
    
    def optimize_model_for_device(self, model: nn.Module, device_specs: Dict) -> nn.Module:
        """
        디바이스 사양에 맞게 모델을 최적화합니다.
        :param model: 최적화할 모델
        :param device_specs: 디바이스 사양
        :return: 최적화된 모델
        """
        try:
            optimized_model = model
            
            # 모델의 각 레이어에 대해 최적 알고리즘 선택
            for name, layer in model.named_modules():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    # 레이어 특성 추출
                    layer_features = self._extract_layer_features(layer)
                    
                    # 최적 알고리즘 선택
                    optimal_algorithm = self.select_optimal_algorithm(layer_features)
                    
                    # 레이어 최적화 적용 (실제 구현에서는 더 구체적인 최적화 필요)
                    self.logger.info(f"Optimizing layer {name} with algorithm {optimal_algorithm}")
            
            return optimized_model
            
        except Exception as e:
            self.logger.error(f"Error in model optimization: {e}")
            return model
    
    def _extract_layer_features(self, layer: nn.Module) -> Dict:
        """
        레이어에서 특성을 추출합니다.
        :param layer: 특성을 추출할 레이어
        :return: 추출된 특성 딕셔너리
        """
        features = {}
        
        if isinstance(layer, nn.Conv2d):
            features.update({
                'layer_type': 0,  # Conv2d
                'input_c': layer.in_channels,
                'input_h': 224,  # 기본값, 실제로는 입력 크기에서 계산 필요
                'input_w': 224,  # 기본값
                'output_c': layer.out_channels,
                'kernel_h': layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size,
                'kernel_w': layer.kernel_size[1] if isinstance(layer.kernel_size, tuple) else layer.kernel_size,
                'stride_h': layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride,
                'stride_w': layer.stride[1] if isinstance(layer.stride, tuple) else layer.stride,
                'padding_h': layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding,
                'padding_w': layer.padding[1] if isinstance(layer.padding, tuple) else layer.padding,
                'groups': layer.groups,
                'bias': 1 if layer.bias is not None else 0,
                'activation': 0  # 기본값
            })
        elif isinstance(layer, nn.Linear):
            features.update({
                'layer_type': 1,  # Linear
                'input_c': layer.in_features,
                'input_h': 1,
                'input_w': 1,
                'output_c': layer.out_features,
                'kernel_h': 1,
                'kernel_w': 1,
                'stride_h': 1,
                'stride_w': 1,
                'padding_h': 0,
                'padding_w': 0,
                'groups': 1,
                'bias': 1 if layer.bias is not None else 0,
                'activation': 0
            })
        
        return features

# 사용 예시
if __name__ == "__main__":
    # LATTE 통합 테스트
    latte_integration = LATTEIntegration()
    
    # 테스트용 레이어 특성
    test_features = {
        'layer_type': 0,
        'input_c': 3,
        'input_h': 224,
        'input_w': 224,
        'output_c': 64,
        'kernel_h': 7,
        'kernel_w': 7,
        'stride_h': 2,
        'stride_w': 2,
        'padding_h': 3,
        'padding_w': 3,
        'groups': 1,
        'bias': 1,
        'activation': 0
    }
    
    # 알고리즘 선택 테스트
    selected_algorithm = latte_integration.select_optimal_algorithm(test_features)
    print(f"Selected algorithm: {selected_algorithm}")
    
    # 훈련 시간 추정 테스트
    model_config = {'model_type': 'resnet18', 'batch_size': 32, 'num_epochs': 1}
    device_specs = {'cpu_cores': 4, 'memory_gb': 8, 'gpu': False}
    estimated_time = latte_integration.estimate_training_time(model_config, device_specs)
    print(f"Estimated training time: {estimated_time:.2f} seconds")

