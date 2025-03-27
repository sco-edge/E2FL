import time
from power.PMIC import PMICMonitor

def test_pmic_monitoring():
    monitor = PMICMonitor()

    print("Starting PMIC monitoring for 10 seconds...")
    monitor.start(freq=1)  # 1초 간격으로 측정
    time.sleep(10)         # 10초 동안 측정
    elapsed, count = monitor.stop()

    print(f"Monitoring finished. Duration: {elapsed}, Samples: {count}")

    # 결과 저장
    monitor.save("pmic_output.csv")
    print("Saved data to 'pmic_output.csv'")

    # 종료
    monitor.close()

if __name__ == "__main__":
    test_pmic_monitoring()
