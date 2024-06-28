import re
from datetime import datetime, timedelta
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--file_path1",
    type=str,
    required=True,
    help="Path to the first dataset file",
)
parser.add_argument(
    "--file_path2",
    type=str,
    required=True,
    help="Path to the second dataset file"
)
parser.add_argument(
    "--file_path3",
    type=str,
    required=True,
    help="Path to the second dataset file"
)
    
    


# FL 로그 파일에서 타임스탬프 추출
def extract_fl_timestamps(fl_log_file):
    computation_phases = []
    communication_phases = []
    
    with open(fl_log_file, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        timestamp_match = re.search(r'\d+\.\d+', line)
        timestamp = float(timestamp_match.group(0)) if timestamp_match else None
        if 'Computation phase started' in line:
            computation_start = timestamp
        elif 'Computation pahse completed' in line:
            computation_end = timestamp
            computation_phases.append((computation_start, computation_end))
        elif 'Wi-Fi start' in line:
            communication_start = timestamp
        elif 'Wi-Fi end' in line:
            communication_end = timestamp
            communication_phases.append((communication_start, communication_end))
    
    return computation_phases, communication_phases

# nethogs 로그 파일에서 네트워크 사용량 계산
def analyze_nethogs_log(nethogs_log_file, phases, pid):
    network_usage = []
    
    with open(nethogs_log_file, 'r') as file:
        lines = file.readlines()
    
    for phase in phases:
        phase_start, phase_end = phase
        phase_usage = {'sent': 0, 'received': 0}
        
        for line in lines:
            match = re.search(r'(\d+):(\d+):(\d+)\s+(\d+.\d+.\d+.\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', line)
            if match:
                hour, minute, second = map(int, match.groups()[:3])
                timestamp = hour * 3600 + minute * 60 + second
                if phase_start <= timestamp <= phase_end:
                    current_pid = int(match.group(5))
                    if current_pid == pid:
                        sent = int(match.group(6))
                        received = int(match.group(7))
                        phase_usage['sent'] += sent
                        phase_usage['received'] += received
        
        network_usage.append(phase_usage)
    
    return network_usage

# 주어진 로그 파일을 분석하는 메인 함수
def main():
    args = parser.parse_args()
    plt.rcParams.update(params)
    plt.tight_layout()

    fl_log_file = 'fl_log.txt'  # FL 로그 파일 경로
    nethogs_log_file = 'nethogs_log.txt'  # nethogs 로그 파일 경로
    pid = 1234  # 추적할 프로세스의 PID
    
    computation_phases, communication_phases = extract_fl_timestamps(fl_log_file)
    
    computation_usage = analyze_nethogs_log(nethogs_log_file, computation_phases, pid)
    communication_usage = analyze_nethogs_log(nethogs_log_file, communication_phases, pid)
    
    print("Computation Phase Network Usage:")
    for usage in computation_usage:
        print(f"Sent: {usage['sent']} bytes, Received: {usage['received']} bytes")
    
    print("\nCommunication Phase Network Usage:")
    for usage in communication_usage:
        print(f"Sent: {usage['sent']} bytes, Received: {usage['received']} bytes")

if __name__ == "__main__":
    main()
