# task_llm.py
import torch
import numpy as np

# ------------------------------
# Backbone loading
# ------------------------------
def load_backbone(model_name: str, device: str):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(device)

    # freeze parameters
    for p in model.parameters():
        p.requires_grad = False

    return model


# ------------------------------
# Forward until split layer
# ------------------------------
def forward_until_split(backbone, input_ids, attention_mask, split_layer: int):
    # Embedding
    x = backbone.model.embed_tokens(input_ids)

    # Decoder layers
    for i, block in enumerate(backbone.model.layers):
        x = block(x, attention_mask=attention_mask)[0]
        if i == split_layer:
            return x

    # fallback: last hidden state
    return x


# ------------------------------
# Activation extraction
# ------------------------------
def extract_activation(hidden):
    # hidden: (B, T, D)
    return hidden


# ------------------------------
# Quantization: INT8
# ------------------------------
def quantize_activation(hidden, num_bits=8):
    max_val = hidden.abs().max()
    scale = max_val / 127.0
    h_q = (hidden / scale).round().clamp(-128, 127).to(torch.int8)
    return h_q, float(scale), hidden.shape


# ------------------------------
# Top-k sparsity compression
# ------------------------------
def compress_topk(hidden, ratio=0.1):
    B, T, D = hidden.shape
    flat = hidden.flatten()
    k = int(ratio * flat.numel())

    values, indices = torch.topk(flat.abs(), k)
    top_vals = flat[indices]

    return top_vals.cpu().numpy(), indices.cpu().numpy(), (B, T, D)


# ------------------------------
# Main API for client
# ------------------------------
def get_activation_for_server(
    backbone,
    batch,
    device,
    split_layer: int,
    compression: str = "int8",
    ratio: float = 0.1,
):
    input_ids = batch["input_ids"].to(device)
    attn_mask = batch.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    hidden = forward_until_split(backbone, input_ids, attn_mask, split_layer)
    hidden = extract_activation(hidden)

    if compression == "none":
        return {
            "type": "none",
            "h": hidden.cpu().numpy(),
            "shape": hidden.shape,
        }

    elif compression == "int8":
        h_q, scale, shape = quantize_activation(hidden)
        return {
            "type": "int8",
            "h_q": h_q.cpu().numpy(),
            "scale": scale,
            "shape": shape,
        }

    elif compression == "topk":
        vals, idx, shape = compress_topk(hidden, ratio)
        return {
            "type": "topk",
            "vals": vals,
            "idx": idx,
            "shape": shape,
        }

    else:
        raise ValueError(f"Unknown compression type: {compression}")

# ==========================================================
# MobiLLM-style adapter and server-side learning utilities
# ==========================================================

import torch.nn as nn

def build_adapter(hidden_size: int, ratio: float = 0.25):
    """Create a simple MLP adapter as used in split-learning setups."""
    bottleneck = int(hidden_size * ratio)
    adapter = nn.Sequential(
        nn.Linear(hidden_size, bottleneck),
        nn.GELU(),
        nn.Linear(bottleneck, hidden_size),
    )
    return adapter


def build_server_models(config: dict):
    """
    Build adapter and loss_fn for server_app_llm.py.
    Hidden size is determined lazily; adapter is initialized after first activation.
    """
    ratio = float(config.get("adapter-hidden-ratio", 0.25))
    # adapter will be lazily initialized — placeholder for now
    adapter = None
    loss_fn = nn.MSELoss()
    return adapter, loss_fn


def server_initialize_adapter(adapter, hidden):
    """
    Initialize adapter lazily once hidden_dim is known.
    """
    if adapter is None:
        hidden_dim = hidden.shape[-1]
        return build_adapter(hidden_dim)
    return adapter


def server_forward_backward(adapter, loss_fn, hidden, optimizer):
    """
    Perform forward + backward pass on the adapter using activation from clients.
    """
    out = adapter(hidden)
    # Dummy self-reconstruction target for now (MobiLLM uses feature alignment)
    loss = loss_fn(out, hidden.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss