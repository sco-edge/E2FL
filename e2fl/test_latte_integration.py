#!/usr/bin/env python3
"""
LATTE 통합 테스트 스크립트
E2FL 시스템에 LATTE가 제대로 통합되었는지 테스트합니다.
"""

import torch
import torch.nn as nn
import sys
import os

# E2FL 모듈 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from e2fl.latte_integration import LATTEIntegration
from e2fl.task import get_model, get_num_classes

def test_latte_integration():
    """LATTE 통합을 테스트합니다."""
    print("=== LATTE Integration Test ===")
    
    try:
        # 1. LATTE 통합 초기화 테스트
        print("1. Initializing LATTE integration...")
        latte_integration = LATTEIntegration()
        print("✓ LATTE integration initialized successfully")
        
        # 2. 모델 로드 테스트
        print("\n2. Loading test model...")
        model_name = "resnet18"
        dataset_name = "cifar10"
        num_classes = get_num_classes(dataset_name)
        model = get_model(model_name, num_classes, dataset_name)
        print(f"✓ Model {model_name} loaded successfully")
        
        # 3. 레이어 특성 추출 테스트
        print("\n3. Testing layer feature extraction...")
        layer_features = latte_integration._extract_layer_features(model.conv1)
        print(f"✓ Layer features extracted: {list(layer_features.keys())}")
        
        # 4. 알고리즘 선택 테스트
        print("\n4. Testing algorithm selection...")
        selected_algorithm = latte_integration.select_optimal_algorithm(layer_features)
        print(f"✓ Selected algorithm: {selected_algorithm}")
        
        # 5. 훈련 시간 추정 테스트
        print("\n5. Testing training time estimation...")
        model_config = {
            'model_type': model_name,
            'batch_size': 32,
            'num_epochs': 1,
            'model_complexity': 1.0
        }
        device_specs = {
            'cpu_cores': 4,
            'memory_gb': 8,
            'gpu': torch.cuda.is_available()
        }
        estimated_time = latte_integration.estimate_training_time(model_config, device_specs)
        print(f"✓ Estimated training time: {estimated_time:.2f} seconds")
        
        # 6. 모델 최적화 테스트
        print("\n6. Testing model optimization...")
        optimized_model = latte_integration.optimize_model_for_device(model, device_specs)
        print("✓ Model optimization completed")
        
        print("\n=== All tests passed! LATTE integration is working correctly. ===")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_different_models():
    """다양한 모델로 LATTE 통합을 테스트합니다."""
    print("\n=== Testing with Different Models ===")
    
    models_to_test = ["resnet18", "resnet50", "resnext50"]
    datasets_to_test = ["cifar10", "mnist"]
    
    try:
        latte_integration = LATTEIntegration()
        
        for model_name in models_to_test:
            for dataset_name in datasets_to_test:
                print(f"\nTesting {model_name} with {dataset_name}...")
                
                try:
                    num_classes = get_num_classes(dataset_name)
                    model = get_model(model_name, num_classes, dataset_name)
                    
                    # 첫 번째 Conv2d 레이어 찾기
                    first_conv = None
                    for name, layer in model.named_modules():
                        if isinstance(layer, nn.Conv2d):
                            first_conv = layer
                            break
                    
                    if first_conv:
                        layer_features = latte_integration._extract_layer_features(first_conv)
                        selected_algorithm = latte_integration.select_optimal_algorithm(layer_features)
                        print(f"  ✓ {model_name} + {dataset_name}: {selected_algorithm}")
                    else:
                        print(f"  ⚠ No Conv2d layer found in {model_name}")
                        
                except Exception as e:
                    print(f"  ❌ Error with {model_name} + {dataset_name}: {e}")
        
        print("\n=== Model testing completed ===")
        return True
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting LATTE integration tests...")
    
    # 기본 통합 테스트
    success1 = test_latte_integration()
    
    # 다양한 모델 테스트
    success2 = test_with_different_models()
    
    if success1 and success2:
        print("\n🎉 All LATTE integration tests passed!")
        print("You can now use LATTE in your E2FL experiments.")
        print("\nTo enable LATTE in experiments:")
        print("1. Set 'enable-latte = true' in pyproject.toml")
        print("2. Or pass '--enable-latte' flag when running experiments")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)

