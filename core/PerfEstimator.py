from dataclasses import dataclass
import numpy as np

'''
Collects RSSI and CSI from edge devices. 
Estimates conditions of wireless environment from the measurements.
Predicts the federated learning performance (energy consumption, learning time, so on).
'''

@dataclass #https://www.daleseo.com/python-dataclasses/
class WirlENV:
    '''
    Those measurements are collected on a per-packet basis.
    Pkt means 
    
    '''
    Pkt: int
    RSSI: float
    CSI: np.complex

class PerfEST():
    def __init__(self, RSSI, CSI, )
        