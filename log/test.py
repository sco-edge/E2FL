import subprocess
import re

def parse_iwconfig_output():
    # iwconfig 명령어 실행
    result = subprocess.run(['iwconfig'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    
    # 출력 결과를 줄 단위로 분리
    lines = result.stdout.split('\n')
    
    # 각 인터페이스의 정보를 저장할 사전 초기화
    interfaces = {}
    
    current_interface = None
    
    # 출력 결과 파싱
    for line in lines:
        # 새로운 인터페이스 섹션의 시작을 확인
        if re.match("^[a-zA-Z0-9]+", line):
            current_interface = line.split(' ')[0]
            interfaces[current_interface] = {}
        # ESSID 추출
        elif "ESSID" in line:
            essid = re.search('ESSID:"([^"]+)"', line)
            if essid:
                interfaces[current_interface]['ESSID'] = essid.group(1)
        # Quality 추출
        elif "Link Quality" in line:
            quality = re.search('Link Quality=([^ ]+)', line)
            if quality:
                interfaces[current_interface]['Link Quality'] = quality.group(1)
        # Signal level 추출
        elif "Signal level" in line:
            signal_level = re.search('Signal level=([^ ]+)', line)
            if signal_level:
                interfaces[current_interface]['Signal level'] = signal_level.group(1)
    
    return interfaces

# 결과 출력
interfaces_info = parse_iwconfig_output()
print(interfaces_info)
