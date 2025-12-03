# server.py
import flwr as fl

def main():
    # 서버 설정 및 실행
    strategy = fl.server.strategy.FedAvg()  # 기본 연합 학습 전략
    fl.server.start_server(server_address="localhost:8080", strategy=strategy)

if __name__ == "__main__":
    main()
