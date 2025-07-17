
import torch
from transformers import AutoTokenizer



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
    if model_name.lower() == "facebook/opt-6.7b":
        num_layers = 32
        hidden_size = 4096
        num_heads = 32
        head_dim = hidden_size // num_heads
    else:
        raise ValueError(f"Model '{model_name}' is not supported in this function.")

    total_tokens = prompt_len + gen_len
    dtype_bytes = 2  # FP16

    # Each token: 2 (key and value) * num_layers * num_heads * head_dim * dtype_bytes
    bytes_per_token = 2 * num_layers * num_heads * head_dim * dtype_bytes
    total_bytes = bytes_per_token * total_tokens * batch_size

    size_gb = total_bytes / (1024 ** 3)
    return size_gb