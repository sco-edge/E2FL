import yaml

# 설정할 데이터 구조
data = {
    'server': {
        'host': '192.168.0.17',
        'ssh_key': 'server',
        'interface': 'wlx30b5c212273b'
    },
    'RPi3B+': {
        'host': '192.168.0.14',
        'ssh_key': 'rpi3b_plus',
        'interface': 'wlan0'
    },
    'RPi3B+_b': {
        'host': '192.168.0.22',
        'ssh_key': 'rpi3b_plus',
        'interface': 'wlan0'
    },
    'RPi4B': {
        'host': '192.168.0.15',
        'ssh_key': 'rpi3b_plus',
        'interface': 'wlan0'
    },
    'RPi5': {
        'host': '192.168.0.19',
        'ssh_key': 'rpi3b_plus',
        'interface': 'wlan0'
    }
}

# YAML 파일 쓰기
with open('config.yaml', 'w') as file:
    yaml.dump(data, file, default_flow_style=False)