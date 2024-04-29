import yaml

# 설정할 데이터 구조
data = {
    'server': {
        'host': '192.168.1.10'
    }
}

# YAML 파일 쓰기
with open('config.yaml', 'w') as file:
    yaml.dump(data, file, default_flow_style=False)