
import torch
from transformers import AutoTokenizer
import json
import math

def prefix_pad_prompt(prompt, target_len, tokenizer):
    input_ids = tokenizer(prompt)["input_ids"]
    prompt_len = len(input_ids)
 
    pad_token = tokenizer.eos_token or tokenizer.pad_token or "<pad>"
    pad_token_id = tokenizer(pad_token)["input_ids"][0]
 
    if prompt_len < target_len:
        padding = [pad_token_id] * (target_len - prompt_len - 1)
        padded_ids = padding + input_ids
    else:
        padded_ids = input_ids[-target_len:]
 
    return tokenizer.decode(padded_ids, skip_special_tokens=False)
 

def calculate_metrics(outputs):
    total_ttft = 0
    total_tpot = 0
    total_prefill_time = 0
    total_decode_time = 0
    total_prompt_token = 0
    total_generate_token = 0
    count = 0
    
    for output in outputs:
        metrics = output.metrics
        num_prompt_tokens = len(output.prompt_token_ids or [])
        total_prompt_token += num_prompt_tokens
        num_gen_tokens = sum(len(o.token_ids) for o in output.outputs)
        total_generate_token += num_gen_tokens
        if not metrics or metrics.first_token_time is None or metrics.last_token_time is None:
            continue

        decode_time = metrics.last_token_time - metrics.first_token_time
        ttft = metrics.first_token_time - metrics.arrival_time
        if num_gen_tokens > 1 and decode_time > 0:
            tpot = decode_time / (num_gen_tokens - 1)
        else:
            tpot = 0

        total_ttft += ttft
        total_tpot += tpot
        count += 1
        
    total_prefill_time = max(o.metrics.first_token_time for o in outputs if o.metrics.first_token_time != None) \
                            - min(o.metrics.first_scheduled_time for o in outputs)
    total_decode_time = max(o.metrics.last_token_time for o in outputs) \
                            - min(o.metrics.first_token_time for o in outputs if o.metrics.first_token_time != None)
    prefill_throughput = total_prompt_token / total_prefill_time
    decode_throughput = total_generate_token / total_decode_time
    avg_ttft = total_ttft / count
    avg_tpot = total_tpot / count
    return prefill_throughput, decode_throughput, avg_ttft, avg_tpot, total_prefill_time, total_decode_time


def calculated_kv_cache_size_GB(model_name, prompt_len, gen_len, batch_size):
    if model_name == "facebook/opt-6.7b":
        num_layers = 32
        hidden_size = 4096
        num_heads = 32
    elif model_name == "facebook/opt-13b":
        num_layers = 40
        hidden_size = 5120
        num_heads = 40
    elif model_name == "facebook/opt-30b":
        num_layers = 48
        hidden_size = 7168
        num_heads = 56
    elif model_name == "facebook/opt-66b":
        num_layers = 64
        hidden_size = 9216
        num_heads = 72
    elif model_name == "meta-llama/Llama-2-13b-hf":
        num_layers = 40
        hidden_size = 5120
        num_heads = 40
    else:
        raise ValueError(f"Model '{model_name}' is not supported in this function.")

    head_dim = hidden_size // num_heads
    total_tokens = prompt_len + gen_len
    dtype_bytes = 2  # FP16

    # Each token: 2 (key and value) * num_layers * num_heads * head_dim * dtype_bytes
    bytes_per_token = 2 * num_layers * num_heads * head_dim * dtype_bytes
    total_bytes = bytes_per_token * total_tokens * batch_size

    size_gb = total_bytes / (1024 ** 3)
    return size_gb

def modify_model_config(model_name, expected_model_len, tp=1):
    tp = 2
    if tp == 1: 
        if model_name == "facebook/opt-6.7b":
            _config_json = "/home/ubuntu/.cache/huggingface/models--facebook--opt-6.7b/blobs/bebe2424fb9fa4e2b5f0b24d7a12d6004553ee6e"
        if model_name == "facebook/opt-13b":
            _config_json = "/home/ubuntu/.cache/huggingface/models--facebook--opt-13b/blobs/d66132763e510905b39cbad4d7fd1b666a185e50"
        if model_name == "facebook/opt-30b":
            _config_json = "/dev/shm/.cache/huggingface/models--facebook--opt-30b/blobs/235a014b573b6a338c37f0058429bbf1f1b8a081"
            # _config_json = "/home/ubuntu/.cache/huggingface/models--facebook--opt-30b/blobs/235a014b573b6a338c37f0058429bbf1f1b8a081"
        if model_name == "facebook/opt-66b":
            _config_json = "/dev/shm/.cache/huggingface/models--facebook--opt-66b/blobs/bf3355ee2a48a23c8379441e0d0832f06118924d"

    if tp == 2:
        if model_name == "facebook/opt-6.7b":
            _config_json = "/dev/shm/.cache/huggingface/models--facebook--opt-6.7b/blobs/bebe2424fb9fa4e2b5f0b24d7a12d6004553ee6e"
        if model_name == "facebook/opt-13b":
            _config_json = "/dev/shm/.cache/huggingface/models--facebook--opt-13b/blobs/d66132763e510905b39cbad4d7fd1b666a185e50"
        if model_name == "facebook/opt-30b":
            _config_json = "/dev/shm/.cache/huggingface/models--facebook--opt-30b/blobs/235a014b573b6a338c37f0058429bbf1f1b8a081"
        if model_name == "facebook/opt-66b":
            _config_json = "/dev/shm/.cache/huggingface/models--facebook--opt-66b/blobs/bf3355ee2a48a23c8379441e0d0832f06118924d"
        if model_name == "meta-llama/Llama-2-13b-hf":
            _config_json = "/dev/shm/.cache/huggingface/models--meta-llama--Llama-2-13b-hf/blobs/374448aabc223983bce6e8127250846e2acf5cf2"
    with open(_config_json, "r") as f:
        config = json.load(f)
    if expected_model_len > 1024:
        config["max_position_embeddings"] = expected_model_len
    else:
        config["max_position_embeddings"] = 1024
    with open(_config_json, "w") as f:
        json.dump(config, f, indent=2)

def estimate_cpu_offload(model_name, kv_size, tp=1):
    """
    Estimate how much model weight needs to be offloaded to CPU,
    given the kv cache size (in GB), based on 90% utilization of 15.77GB GPU memory.
    """

    # example offload 2gb
    # model weights take 10.43GiB; 
    # non_torch_memory takes 0.07GiB; 
    # PyTorch activation peak memory takes 0.85GiB; 
    # the rest of the memory reserved for KV Cache is 2.84GiB.

    # example offload 1gb
    # model weights take 11.41GiB; 
    # non_torch_memory takes 0.07GiB; 
    # PyTorch activation peak memory takes 0.76GiB; 
    # the rest of the memory reserved for KV Cache is 1.95GiB.
    
    # example offload nothing
    # model weights take 12.40GiB; 
    # non_torch_memory takes 0.07GiB; 
    # PyTorch activation peak memory takes 0.38GiB; 
    # the rest of the memory reserved for KV Cache is 1.33GiB.
    
    _GPU_MEM = 15.77 * 0.9 * tp
    non_torch_memory = 0.07  # constant across examples
    if tp == 2:
        non_torch_memory = 0.17 * tp
    activation = 0
    model_size_gb = 0

    if model_name == "facebook/opt-6.7b":
        model_size_gb = 12.4
        activation = 0.38
    elif model_name == "facebook/opt-13b":
        model_size_gb = 24.9  # reported size for OPT-13B in vLLM logs
        activation = 0.6      # rough estimate, can tune based on observation
    elif model_name == "facebook/opt-30b":
        model_size_gb = 56
        activation = 1.0
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Compute available memory for model weights
    remaining_mem = _GPU_MEM - non_torch_memory - activation - kv_size
    _cpu_offload = max(0, model_size_gb - remaining_mem)

    _cpu_offload = math.ceil(_cpu_offload)
    return _cpu_offload
