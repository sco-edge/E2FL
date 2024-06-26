#!/bin/sh
num_clients=2
sample_frac=1.0
round=3

# Launch your server.
# Will wait for at least 2 clients to be connected, then will train for 3 FL rounds
# The command below will sample all clients connected (since sample_fraction=1.0)
# The server is dataset agnostic (use the same command for MNIST and CIFAR10)
python3.10 ./FLOWER_embedded_devices/server.py --rounds $round --min_num_clients $num_clients --sample_fraction $sample_frac