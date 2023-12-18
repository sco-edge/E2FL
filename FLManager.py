import serial
import time
import timeit
import logging
import PerfEstimator

logging.basicConfig(
    format = '%(asctime)s:%(levelname)s:%(message)s',
    datefmt = '%m\%d\%Y %I:%M:%S %p',
    level = logging.DEBUG
)

def execute_shell_script(serial_port, script_path):
    ser = serial.Serial(serial_port, baudrate=9600, timeout=1)

    try:
        command = f'bash {script_path}\n'
        ser.write(command.encode('utf-8'))
        time.sleep(1)
    
        response = ser.read(ser.in_waiting).decode('utf-8')
        print("Shell Script Execution Result")
        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")    

    finally:
        ser.close()
        
def updatePolicy():
    comp_round = 0
    comm_round = 0



    return (comp_round,comm_round)

if __name__ == "__main__":
    # Configuration
    rpi_serial_port = "/dev/ttyUSB0"
    shell_path = "/home/E2FL/test/"
    
    execute_shell_script(rpi_serial_port, shell_path)

    # execute server on local edge server
    # command = f'bash run_server.sh'


    # execute client
    # command = f'bash run_client.sh 1'
    # start_c1 = timeit.default_timer()
    

    # command = f'bash run_client.sh 2'
    # start_c2 = timeit.default_timer()

    # execute PyMonsoon



    # and log time 
    # start = timeit.default_timer()
    # stop = timeit.default_timer()
    # duration = stop - start


'''
[Reference]
https://stackoverflow.com/questions/5622976/how-do-you-calculate-program-run-time-in-python
https://doc.fedml.ai/federate/getting_started
https://doc.fedml.ai/federate/cross-silo/example/mqtt_s3_fedavg_hierarchical_mnist_lr_example
https://doc.fedml.ai/federate/simulation/examples/mpi_torch_fedavg_mnist_lr_example

https://aws.amazon.com/ko/blogs/machine-learning/part-1-federated-learning-on-aws-with-fedml-health-analytics-without-sharing-sensitive-data/
https://aws.amazon.com/ko/blogs/machine-learning/part-2-federated-learning-on-aws-with-fedml-health-analytics-without-sharing-sensitive-data/

https://github.com/usc-sail/fed-multimodal

https://github.com/FedML-AI/FedML/blob/master/python/fedml/core/distributed/communication/message.py
https://github.com/FedML-AI/FedML/blob/master/python/fedml/cross_silo/client/fedml_client_master_manager.py#L18

'''