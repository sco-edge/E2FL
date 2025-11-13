#!/usr/bin/env python3
"""
E2FL 실험 결과 분석 스크립트
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import numpy as np

def analyze_network_usage():
    """네트워크 사용량 분석"""
    print("=== 네트워크 사용량 분석 ===")
    
    # CSV 파일들 찾기
    csv_files = glob.glob("fl_*.csv")
    if not csv_files:
        print("네트워크 사용량 CSV 파일을 찾을 수 없습니다.")
        return
    
    all_data = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, header=None, names=['timestamp', 'phase', 'bytes_sent', 'bytes_recv'])
            device_name = file.split('_')[2]  # fl_20250115_RPi5_19.csv -> RPi5
            df['device'] = device_name
            all_data.append(df)
            print(f"✅ {file}: {len(df)} 레코드")
        except Exception as e:
            print(f"❌ {file} 읽기 실패: {e}")
    
    if not all_data:
        return
    
    # 전체 데이터 합치기
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 통계 출력
    print("\n📊 네트워크 사용량 통계:")
    device_stats = combined_df.groupby('device').agg({
        'bytes_sent': ['sum', 'mean', 'max'],
        'bytes_recv': ['sum', 'mean', 'max']
    }).round(2)
    print(device_stats)
    
    # 그래프 생성
    plt.figure(figsize=(15, 10))
    
    # 1) 디바이스별 총 송신량
    plt.subplot(2, 2, 1)
    sent_totals = combined_df.groupby('device')['bytes_sent'].sum()
    sent_totals.plot(kind='bar', color='skyblue')
    plt.title('디바이스별 총 송신량 (bytes)')
    plt.ylabel('Bytes Sent')
    plt.xticks(rotation=45)
    
    # 2) 디바이스별 총 수신량
    plt.subplot(2, 2, 2)
    recv_totals = combined_df.groupby('device')['bytes_recv'].sum()
    recv_totals.plot(kind='bar', color='lightcoral')
    plt.title('디바이스별 총 수신량 (bytes)')
    plt.ylabel('Bytes Received')
    plt.xticks(rotation=45)
    
    # 3) 시간별 네트워크 사용량 (첫 번째 디바이스)
    plt.subplot(2, 2, 3)
    first_device = combined_df['device'].iloc[0]
    device_data = combined_df[combined_df['device'] == first_device]
    plt.plot(device_data['bytes_sent'], label='Sent', marker='o')
    plt.plot(device_data['bytes_recv'], label='Received', marker='s')
    plt.title(f'{first_device} 시간별 네트워크 사용량')
    plt.xlabel('Time Steps')
    plt.ylabel('Bytes')
    plt.legend()
    
    # 4) 디바이스별 평균 사용량 비교
    plt.subplot(2, 2, 4)
    avg_data = combined_df.groupby('device')[['bytes_sent', 'bytes_recv']].mean()
    avg_data.plot(kind='bar', ax=plt.gca())
    plt.title('디바이스별 평균 네트워크 사용량')
    plt.ylabel('Average Bytes')
    plt.xticks(rotation=45)
    plt.legend(['Sent', 'Received'])
    
    plt.tight_layout()
    plt.savefig('network_usage_analysis.png', dpi=300, bbox_inches='tight')
    print("📈 네트워크 사용량 그래프 저장: network_usage_analysis.png")
    plt.show()

def analyze_flower_logs():
    """Flower 로그 분석"""
    print("\n=== Flower 로그 분석 ===")
    
    log_dir = os.path.expanduser("~/.flwr/logs")
    if not os.path.exists(log_dir):
        print("Flower 로그 디렉토리를 찾을 수 없습니다.")
        return
    
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    if not log_files:
        print("Flower 로그 파일을 찾을 수 없습니다.")
        return
    
    print(f"📁 발견된 로그 파일: {len(log_files)}개")
    for log_file in log_files:
        print(f"  - {os.path.basename(log_file)}")
        
        # 로그 파일 크기 확인
        size_mb = os.path.getsize(log_file) / (1024 * 1024)
        print(f"    크기: {size_mb:.2f} MB")
        
        # 최근 로그 몇 줄 읽기
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"    총 라인 수: {len(lines)}")
                if lines:
                    print(f"    마지막 로그: {lines[-1].strip()}")
        except Exception as e:
            print(f"    로그 읽기 실패: {e}")
        print()

def check_experiment_status():
    """실험 상태 확인"""
    print("=== 실험 상태 확인 ===")
    
    # 1) 현재 실행 중인 프로세스 확인
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        flower_processes = [line for line in result.stdout.split('\n') if 'flower' in line.lower()]
        
        if flower_processes:
            print("🔄 실행 중인 Flower 프로세스:")
            for proc in flower_processes:
                print(f"  {proc}")
        else:
            print("❌ 실행 중인 Flower 프로세스가 없습니다.")
    except Exception as e:
        print(f"프로세스 확인 실패: {e}")
    
    # 2) 네트워크 연결 상태 확인
    try:
        result = subprocess.run(['netstat', '-tuln'], capture_output=True, text=True)
        flower_ports = [line for line in result.stdout.split('\n') if ':909' in line]
        
        if flower_ports:
            print("\n🌐 Flower 포트 상태:")
            for port in flower_ports:
                print(f"  {port}")
        else:
            print("\n❌ Flower 포트가 열려있지 않습니다.")
    except Exception as e:
        print(f"포트 확인 실패: {e}")

def generate_summary_report():
    """요약 보고서 생성"""
    print("\n=== 실험 요약 보고서 생성 ===")
    
    report = []
    report.append("# E2FL 실험 결과 요약")
    report.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 네트워크 사용량 요약
    csv_files = glob.glob("fl_*.csv")
    if csv_files:
        report.append("## 네트워크 사용량 요약")
        for file in csv_files:
            try:
                df = pd.read_csv(file, header=None, names=['timestamp', 'phase', 'bytes_sent', 'bytes_recv'])
                device_name = file.split('_')[2]
                total_sent = df['bytes_sent'].sum()
                total_recv = df['bytes_recv'].sum()
                report.append(f"- **{device_name}**: 송신 {total_sent:,} bytes, 수신 {total_recv:,} bytes")
            except:
                pass
        report.append("")
    
    # 파일 목록
    report.append("## 생성된 파일들")
    all_files = glob.glob("fl_*.csv") + glob.glob("*.png") + glob.glob("*.log")
    for file in all_files:
        size_kb = os.path.getsize(file) / 1024
        report.append(f"- {file} ({size_kb:.1f} KB)")
    
    # 보고서 저장
    with open('experiment_summary.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("📄 실험 요약 보고서 저장: experiment_summary.md")

if __name__ == "__main__":
    print("🔍 E2FL 실험 결과 분석 시작...")
    print("=" * 50)
    
    # 1. 실험 상태 확인
    check_experiment_status()
    
    # 2. 네트워크 사용량 분석
    analyze_network_usage()
    
    # 3. Flower 로그 분석
    analyze_flower_logs()
    
    # 4. 요약 보고서 생성
    generate_summary_report()
    
    print("\n✅ 분석 완료!")
    print("📁 생성된 파일들:")
    print("  - network_usage_analysis.png (네트워크 사용량 그래프)")
    print("  - experiment_summary.md (실험 요약 보고서)")
