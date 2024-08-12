# install
## git clone https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning/
## bash install.sh

# LLM-adapters; https://github.com/AGI-Edgerunners/LLM-Adapters
#  - adpter-based PEFT (Paremeter Efficient Fine Tuning) methods of LLMs for different tasks
# 

import os
import torch
from peft import PeftModel
from search.supernet import ShearsSuperNet
from transformers import AutoModelForCausalLM

from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.multi_elasticity_handler import SubnetConfig

from utils.utils import load_nncf_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

SHEARS_PATH = "./shears-llama-7b-50-math-super"
BASE_MODEL_PATH = f"{SHEARS_PATH}/base_model"
ADAPTER_MODEL_PATH = f"{SHEARS_PATH}/adapter_model"
NNCF_CONFIG = "../nncf_config/nncf_shears_llama.json"

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH,trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_PATH)
nncf_config = load_nncf_config(NNCF_CONFIG, num_hidden_layers=model.config.num_hidden_layers)
supernet = ShearsSuperNet.from_checkpoint(model, nncf_config, supernet_elasticity_path=None, supernet_weights_path=None)

supernet.get_search_space()

# maximal subnetwork
supernet.activate_maximal_subnet()
supernet.get_active_config()
supernet.get_macs_for_active_config()
# heuristic subnetwork
supernet.activate_heuristic_subnet()
supernet.get_active_config()
supernet.get_macs_for_active_config()
# minimal subnetwork
supernet.activate_minimal_subnet()
supernet.get_active_config()
supernet.get_macs_for_active_config()
# random subnetwork
import random
subnet_config = SubnetConfig()
search_space = supernet.get_search_space()
subnet_config[ElasticityDim.WIDTH] = {i: random.choice(space) for i, space in search_space[ElasticityDim.WIDTH.value].items()}

supernet.activate_config(subnet_config)
supernet.get_active_config()
supernet.get_macs_for_active_config()

# extract and save the active sub-adapter
supernet.extract_and_save_active_sub_adapter(super_adapter_dir=ADAPTER_MODEL_PATH, sub_adapter_dir=os.path.join(ADAPTER_MODEL_PATH, "sub_adapter"))