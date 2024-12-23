from dataclasses import dataclass
import numpy as np

'''
"SERVER Perspective"
Collects RSSI and CSI from edge devices. 
Estimates conditions of wireless environment from the measurements.
Predicts the federated learning performance (energy consumption, learning time, so on).
'''

energy_profile = {
    'RPi 3B+': 0.3
}


@dataclass #https://www.daleseo.com/python-dataclasses/
class WirlENV:
    '''
    * Those measurements are collected on a per-packet basis.
    Pkt means the received packet index
    RSSI means received signal strength index.
    CSI means channel state information.
    DR means transmission data rate
    '''
    Pkt: int
    RSSI: float
    CSI: np.complex
    DR: float

class Energy_Time_Estimator:
    '''
    * Performance Estimator for each device.
    '''
    def __init__(self, edgeDev_name):
        self.WirlENV = []
        self.pktIdx = 0
        self.edgeDev_name = edgeDev_name
        self.H = 4

    def addWirl(self, RSSI, CSI, DR):
        self.WirlENV.append(WirlENV(Pkt = self.pktIdx, RSSI = RSSI, CSI = CSI, DR = DR))
        self.pktIdx = self.pktIdx + 1
    
    def getWirl(self, pktIdx):
        return (self.edgeDev_name, self.WirlENV[pktIdx])

    def calcH(self):
        '''
        calculates {H} from the wireless environmetns and transmission data rate.
        {H} is the number of local training iterations.
        '''
        
        # estimates the channel conditions from the WirlENV list

        # update H

        return self.H

    def estEslope(self):
        '''
        estimates the slope of energy consumption according to device type and wireless condition for the global round.
        '''
        return 0


class Learning_Time_Estimator:
    def __init__(self):
    