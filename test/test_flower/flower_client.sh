#!/bin/sh
server_address=192.168.0.6
client_id=1

echo "Enter the client id: "
read client_id
echo "client_id: $client_id"

echo "Enter the server address: "
read server_address
echo "server_address: $server_address"

# Run the default example (CIFAR-10)
# python3 client_pytorch.py --cid=$client_id --server_address=$server_address

# Use MNIST (and a smaller model) if your devices require a more lightweight workload
python3.10 ./FLOWER_embedded_devices/client_pytorch.py --cid=$client_id --server_address=$server_address --mnist
# python3.10 ./FLOWER_embedded_devices/client_pytorch.py --cid=0 --server_address=192.168.0.6 --mnist