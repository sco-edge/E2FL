# This script loads a base Gemma model and a LoRA adapter checkpoint for evaluation.
# Before running, please check the following:
# - base_model_id matches the base model used during federated fine-tuning.
# - adapter_ckpt path points to the correct adapter checkpoint for the desired round.
# - LoRA config (r, alpha, dropout, target modules) matches the training setup.
# - Device and torch_dtype are appropriate for your hardware.
# - Evaluation task configuration is set up as needed for your use case.

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

# This must match the base model used during federated fine-tuning.
base_model_id = "google/gemma-3-270m"

# If the Gemma model is gated on Hugging Face, you need an access token that has
# accepted the model license. The recommended options are:
#   1) Run the HF CLI login once on this machine:
#        pip install -U "huggingface_hub[cli]"
#        huggingface-cli login
#      (Then follow the instructions in the terminal.)
#   2) Or export an environment variable, e.g.:
#        export HF_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
#
# If HF_TOKEN is set, it will be passed to from_pretrained; otherwise, the
# huggingface cache/default login will be used.
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Path/filename should point to the desired round’s adapter checkpoint exported by the server
# (e.g., last round vs. other rounds). Update as needed depending on resume runs.
adapter_ckpt = "../eval/peft_hf_google_gemma-3-270m_10.pt"

device = "cuda"

# NOTE: For Gemma models we typically:
# - enable `trust_remote_code=True` (if required by the HF implementation),
# - set `pad_token` to `eos_token`, and
# - use right-padding.
# If this differs from the training-side settings, mirror whatever you used there.
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    token=HF_TOKEN,          # Uses HF_TOKEN if set, otherwise defaults to cached credentials
    # trust_remote_code=True,  # Uncomment if needed for Gemma implementation
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    token=HF_TOKEN,          # Uses HF_TOKEN if set, otherwise defaults to cached credentials
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# LoRA hyperparameters (r, lora_alpha, lora_dropout, target_modules, bias) must match
# the training configuration used in E2FL, otherwise the loaded adapter checkpoint will be incompatible.
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_cfg)

# The script assumes a PEFT-only state dict (as saved by the Flower/E2FL server),
# not a full model checkpoint.
state_dict = torch.load(adapter_ckpt, map_location="cpu")
set_peft_model_state_dict(model, state_dict)
model.to(device)
model.eval()

# 여기서부터는 너가 정한 eval 태스크 (예: alpaca-style prompt set, MT-bench 서브셋 등) 돌리면 됨