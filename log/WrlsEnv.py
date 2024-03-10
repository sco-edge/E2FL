from wifi import Cell, Scheme
import subprocess
import re
from pprint import pprint 


class Timer():
    '''
    Time logging for FL and energy monitoring.
    '''

    def __init__(self, edgeDev_name):
        self.edgeDev_name = edgeDev_name
        self.log_entries = []

    def log_event(self, event_name):
        timestamp = datetime.datetime.now()
        log_entry = {'device_name': self.edgeDev_name, 'event_name': event_name, 'timestamp': timestamp}
        self.log_entries.append(log_entry)
    
    def get_log_entries(self):
        return self.log_entries
    
class WiFI():
    def read_rssi_from_proc_wireless():
        # /proc/net/wireless 파일 경로
        wireless_path = '/proc/net/wireless'
        '''
         Inter-| sta-|   Quality         |   Discarded packets       | Missed |  WE
           face| tus | link level noise  | nwid crypt frag retry misc| beacon |  22
         wlan0: 00000   70  -20.   -256      0     0     0    0     0      0 

         link means link quality (iwconfig shows 70/70)
         level means signal level (iwconfig shows -20 dBm)
        '''
        try:
            # 파일 열기
            with open(wireless_path, 'r') as file:
                # 파일의 각 줄을 순회하면서 처리
                for line in file.readlines():
                    # 무선 인터페이스의 통계 정보가 있는 줄을 찾기
                    if ':' in line:
                        # 줄을 공백 문자로 나눔
                        data = line.split()
                        # 인터페이스 이름, RSSI(level) 값을 추출
                        interface = data[0].rstrip(':')
                        rssi = data[3]
                        #print(f"Interface: {interface}, RSSI: {rssi} dBm")
                        return [interface, rssi]
        except FileNotFoundError:
            print(f"File not found: {wireless_path}")
        except PermissionError:
            print(f"Permission denied: {wireless_path}")

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
                    quality = re.search('Link Quality=(\d+/+\d+)', line)
                    if quality:
                        interfaces[current_interface]['Link Quality'] = quality.group(1)
                # Signal level 추출
                elif "Signal level" in line:
                    signal_level = re.search('Signal level=([^]+)', line)
                    if signal_level:
                        interfaces[current_interface]['Signal level'] = signal_level.group(1)
                elif "TX-Power" in line:
                    TX_power = re.search('TX-Power=([^]+)', line)
                    if TX_power:
                        interfaces[current_interface]['TX-Power'] = TX_power.group(1)
                elif "Bit Rate" in line:
                    bit_rate = re.search('Bit Rate=(\d+[.]?\d?) Mb/s', line)
                    if bit_rate:
                        interfaces[current_interface]['Bit Rate'] = bit_rate.group(1)
                elif "Retry short limit" in line:
                    retry_limit = re.search('Retry short limit:(\d)', line)
                    if retry_limit:
                        interfaces[current_interface]['Retry short limit'] = retry_limit.group(1)
                elif "Frequency" in line:
                    freq = re.search('Frequency:(\d+.+\d) GHz', line)
                    if freq:
                        interfaces[current_interface]['Frequency'] = freq.group(1)
                elif "Rx invalid nwid" in line:
                    rx_nwid = re.search('Rx invalid nwid:(\d)', line)
                    if rx_nwid:
                        interfaces[current_interface]['Rx invalid nwid'] = rx_nwid.group(1)
            return interfaces

        # 결과 출력
        interfaces_info = parse_iwconfig_output()
        pprint(interfaces_info)


    

