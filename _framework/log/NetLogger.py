import psutil
from scapy.all import sniff

pid = 0
mac_addr = 0
pid2traffic = {'bytes_sent': 0, 'bytes_recv': 0} # 0: outgoing packet,

# https://psutil.readthedocs.io/en/latest/
# https://psutil.readthedocs.io/en/latest/index.html#process-class
# https://stackoverflow.com/questions/75983163/what-exactly-does-psutil-net-io-counters-byte-recv-mean
# https://github.com/giampaolo/psutil/blob/master/scripts/nettop.py
def wrong_get_network_usage(interf):
    p = psutil.Process()
    net_io = p.net_io_counters(pernic=True)
    #net_io = psutil.net_io_counters(pernic=True)
    return {"bytes_sent": net_io[interf].bytes_sent, "bytes_recv": net_io[interf].bytes_recv}

# https://thepythoncode.com/article/make-a-network-usage-monitor-in-python#network-usage-per-process
def get_size(bytes):
    '''
    Returns size of bytes in a nice format
    '''
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if bytes < 1024:
            return f"{bytes:.2f}{unit}B"
        bytes /= 1024

def process_packet(packet):
    
    if packet.src == mac_addr:
        # outgoing packet
        pid2traffic['bytes_sent'] += len(packet)
    else:
        pid2traffic['bytes_recv'] += len(packet)
        

def get_network_usage(interf):
    pid = psutil.Process().ppid()
    # get the MAC addresses of the interface
    mac_addr = psutil.net_if_addrs()[interf][1].address

    sniff(iface=interf, prn=process_packet, store=False)

    return 

    
    


