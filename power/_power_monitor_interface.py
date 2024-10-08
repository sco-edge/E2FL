from abc import ABC, abstractmethod

class PowerMonitor(ABC):
    def __init__(self, device_name):
        self.device_name = device_name  # 장치 이름
        self.sampling_interval = 1  # 샘플링 주기 (초)
        self.power_data = []  # 전력 데이터를 저장할 리스트 [(timestamp, power)]
        self.is_monitoring = False
        self.start_time = None  # 측정 시작 시간
    
    @abstractmethod
    def start(self, freq):
        """
        주어진 주기로 전력 측정을 시작하는 메서드.
        :param freq: 샘플링 주기 (초 단위)
        """
        pass

    @abstractmethod
    def stop(self):
        """
        전력 측정을 중단하고, 측정된 데이터의 크기와 측정 기간을 반환.
        :return: (elapsed_time, data_size)
        """
        pass

    @abstractmethod
    def read_power(self):
        """
        현재 전력 소모량을 읽는 메서드.
        :return: float (전력, 단위는 mW)
        """
        pass

    @abstractmethod
    def save(self, filepath):
        """
        측정된 데이터를 지정된 경로에 저장하는 메서드.
        :param filepath: 파일 경로
        """
        pass

    @abstractmethod
    def close(self):
        """
        모니터링을 종료하고 리소스를 정리하는 메서드.
        """
        pass
