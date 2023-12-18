import datetime
import logging


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