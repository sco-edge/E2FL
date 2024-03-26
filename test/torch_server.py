import fedml


if __name__ == "__main__":
    fedml.run_hierarchical_cross_silo_server()
    #python server.py --rounds 3 --min_num_clients 2 --sample_fraction 1.0

    

'''
[Reference]
FLOWER
https://doc.fedml.ai/federate/simulation/examples/sp_fedavg_mnist_lr_example
https://doc.fedml.ai/federate/simulation/examples/mpi_torch_fedavg_mnist_lr_example
https://doc.fedml.ai/federate/cross-silo/example/mqtt_s3_fedavg_hierarchical_mnist_lr_example
https://github.com/QuanlingZhao/FedHD/blob/main/FedML-IoT-HD/raspberry_pi/fedhd/fedhd_rpi_client.py
'''