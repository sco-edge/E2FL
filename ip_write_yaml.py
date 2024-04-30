import yaml

# 설정할 데이터 구조
data = {
    'server': {
        'host': '192.168.0.17'
    },
    'RPi3B+': {
        'host': '192.168.0.14'
    },
    'RPi4B': {
        'host': '192.168.0.15'
    },
    'RPi5': {
        'host': '192.168.0.19'
    }
}

# YAML 파일 쓰기
with open('config.yaml', 'w') as file:
    yaml.dump(data, file, default_flow_style=False)