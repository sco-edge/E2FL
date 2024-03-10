# torch_client.py
import fedml
import torch
from fedml.cross_silo.hierarchical import Client
from fedml.data.MNIST.data_loader import download_mnist, load_partition_data_mnist
from fedml.data.data_loader_cross_silo import split_data_for_dist_trainers


def load_data(args):
    n_dist_trainer = args.n_proc_in_silo
    download_mnist(args.data_cache_dir)
    fedml.logging.info("load_data. dataset_name = %s" % args.dataset)

    """
    Please read through the data loader at to see how to customize the dataset for FedML framework.
    """
    (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = load_partition_data_mnist(
        args.batch_size,
        train_path=args.data_cache_dir + "/MNIST/train",
        test_path=args.data_cache_dir + "/MNIST/test",
    )
    """
    For shallow NN or linear models, 
    we uniformly sample a fraction of clients each round (as the original FedAvg paper)
    """
    args.client_num_in_total = client_num
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]

    # Split training data between distributed trainers
    train_data_local_dict = split_data_for_dist_trainers(
        train_data_local_dict, n_dist_trainer
    )

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset, class_num


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs



if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load_data(args)

    # load model (the size of MNIST image is 28 x 28)
    model = LogisticRegression(28 * 28, output_dim)

    # start training
    client = Client(args, device, dataset, model)
    client.run()
    

    
'''
import torch
from torch import nn
from fedml import FedML_init, FedML_FedAvg_training

# 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(28 * 28, 10)  # 예: MNIST 데이터셋을 위한 설정

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.linear(x)
        return x

# 연합 학습 설정 및 초기화
args = {
    "model": SimpleModel(),
    "dataset": "mnist",  # 데이터셋 선택
    "data_dir": "./data",  # 데이터 저장 위치
    "client_num_in_total": 4,  # 클라이언트의 총 수
    "client_num_per_round": 4,  # 각 라운드에 참여하는 클라이언트 수
    "comm_round": 10,  # 통신 라운드 수
    "epochs": 1,  # 로컬 에폭 수
    "lr": 0.01,  # 학습률
    "batch_size": 32  # 배치 크기
}

# 연합 학습 초기화 및 실행
FedML_init(args)
FedML_FedAvg_training(args)


[Reference]
https://aws.amazon.com/ko/blogs/machine-learning/part-1-federated-learning-on-aws-with-fedml-health-analytics-without-sharing-sensitive-data/
https://aws.amazon.com/ko/blogs/machine-learning/part-2-federated-learning-on-aws-with-fedml-health-analytics-without-sharing-sensitive-data/
https://github.com/FedML-AI/FedML/tree/master
https://github.com/FedML-AI/FedML/blob/master/python/fedml/core/distributed/communication/message.py
https://github.com/FedML-AI/FedML/blob/master/python/fedml/cross_silo/client/fedml_client_master_manager.py#L18
https://doc.fedml.ai/federate/cross-silo/example/mqtt_s3_fedavg_hierarchical_mnist_lr_example
'''