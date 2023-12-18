import time
import logging


class Timer():
    '''
    Time logging for FL and energy monitoring.
    '''

    def __init__(self, edgeDev_name):
        self.edgeDev_name = edgeDev_name
        self.log_entries = []


    def startTimer(self):
        self.