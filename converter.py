import torch
from safetensors.torch import load_file
import struct
import json
import os

input_dir = "/mnt/e/ModelosLLM/tinyllama1.1B"
output_path = "models/tinyllama_fp16.bin"

def convert():
    print("Carregando config e pesos...")

    with open(os.path.join(input_dir, "config.json"), "r") as f:
        config = json.load(f)

    state_dict = load_file(os.path.join(input_dir, "model.safetensors"))

    dim = config["hidden_size"]
    hidden_dim = config["intermediate_size"]
    n_layers = config["num_hidden_layers"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    vocab_size = config["vocab_size"]
    max_seq_len = 2048

    header = struct.pack(
        "iiiiiii",
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size,
        max_seq_len
    )

    print("Escrevendo modelo FP16...")

    with open(output_path, "wb") as f:
        f.write(header)

        def write(t):
            t.to(torch.float16).numpy().tofile(f)

        write(state_dict["model.embed_tokens.weight"])

        for i in range(n_layers):
            print(f"Layer {i+1}/{n_layers}", end="\r")

            write(state_dict[f"model.layers.{i}.input_layernorm.weight"])
            write(state_dict[f"model.layers.{i}.self_attn.q_proj.weight"])
            write(state_dict[f"model.layers.{i}.self_attn.k_proj.weight"])
            write(state_dict[f"model.layers.{i}.self_attn.v_proj.weight"])
            write(state_dict[f"model.layers.{i}.self_attn.o_proj.weight"])
            write(state_dict[f"model.layers.{i}.post_attention_layernorm.weight"])
            write(state_dict[f"model.layers.{i}.mlp.gate_proj.weight"])
            write(state_dict[f"model.layers.{i}.mlp.down_proj.weight"])
            write(state_dict[f"model.layers.{i}.mlp.up_proj.weight"])

        write(state_dict["model.norm.weight"])
        write(state_dict["lm_head.weight"])

    print("\n✅ Modelo FP16 salvo em:", output_path)

convert()