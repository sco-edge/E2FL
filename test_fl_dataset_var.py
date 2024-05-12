
dataset = ['CIFAR']

client_cmd = "python3.10 ./FLOWER_embedded_devices/client_pytorch.py --cid=$client_id --server_address=$server_address --mnist"
server_cmd = "python3.10 ./FLOWER_embedded_devices/server.py --rounds $round --min_num_clients $num_clients --sample_fraction $sample_frac"
