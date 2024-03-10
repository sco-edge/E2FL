from wifi import Cell, Scheme


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
                        print(f"Interface: {interface}, RSSI: {rssi} dBm")
        except FileNotFoundError:
            print(f"File not found: {wireless_path}")
        except PermissionError:
            print(f"Permission denied: {wireless_path}")

        # 함수 실행
        # read_rssi_from_proc_wireless()

