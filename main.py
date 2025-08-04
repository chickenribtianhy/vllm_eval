import argparse
import torch
import time
import os
from itertools import product
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
import json
import math

from utils import *

os.environ["VLLM_USE_V1"] = "0"
base_prompt = "Paris is the capital of "

log_dir = "./vllm_tp_logs_bs1_tp2"

os.makedirs(log_dir, exist_ok=True)

_OFFLOAD_DEV = 0

def benchmark(model_name, prompt_len, gen_len, batch_size, tensor_parallelism):
    print(f"benchmarking {model_name.replace('/', '_')}_prompt{prompt_len}_gen{gen_len}_bs{batch_size}")
    log_file = os.path.join(log_dir, f"{model_name.replace('/', '_')}_prompt{prompt_len}_gen{gen_len}_bs{batch_size}.log")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = [prefix_pad_prompt(base_prompt, prompt_len, tokenizer) for _ in range(batch_size)]
    
    sampling_params = SamplingParams(temperature=0, max_tokens=gen_len, min_tokens=gen_len)

    expected_model_len = prompt_len + gen_len

    modify_model_config(model_name, expected_model_len, tensor_parallelism)

    calculated_kv_cache_size_per_req = calculated_kv_cache_size_GB(model_name, prompt_len, gen_len, 1)
    print(f"kv cache size per request: {calculated_kv_cache_size_per_req} GB")


    _cpu_offload = estimate_cpu_offload(model_name, calculated_kv_cache_size_per_req, tensor_parallelism) + _OFFLOAD_DEV
    _cpu_offload = 16.5
    print(f"offloading {_cpu_offload} GB model weights")

    llm = LLM(model=model_name,
                tensor_parallel_size=tensor_parallelism,
                gpu_memory_utilization=0.90,
                # max_num_batched_tokens=4096,
                max_num_seqs=32,
                disable_log_stats=False,
                swap_space=50,
                cpu_offload_gb=_cpu_offload,
                preemption_mode="swap",
                dtype="float16",
                load_format="dummy",
                )
 
    # _ = llm.generate(prompts, warm_up_params)
    with open(log_file, "w") as f:
        with redirect_stdout(f), redirect_stderr(f):
            f.write(f"Benchmarking {model_name} | Prompt Len: {prompt_len}, Gen Len: {gen_len}, Batch Size: {batch_size}\n")
            f.write(f"Start time: {datetime.now()}\n\n")
 
 
            start = time.perf_counter()
            responses = llm.generate(prompts, sampling_params)
            end = time.perf_counter()
            runtime = end - start

            f.write(f"\nRuntime: {runtime:.9f} seconds\n")
            f.write(f"Total throughput: {((prompt_len+gen_len)*batch_size)/runtime} tokens/sec\n")
            f.write(f"Generation throughput: {(gen_len*batch_size)/runtime} tokens/sec\n\n")
            

            prefill_throughput, decode_throughput, avg_ttft, avg_tpot, total_prefill_time, total_decode_time = calculate_metrics(outputs=responses)
            # print(count)
            f.write("============= Metrics =============\n")
            f.write(f"Prefill throughput: {prefill_throughput} tokens/s\n")
            f.write(f"Decode throughput: {decode_throughput} tokens/s\n")
            f.write(f"Average TTFT: {avg_ttft} s\n")
            f.write(f"Average TPOT: {avg_tpot} s\n")
            f.write(f"\ntotal prefill time: {total_prefill_time} \n")
            f.write(f"total decode time: {total_decode_time} \n")
            print("===================================\n")

            print(len(responses)) # 0
            for output in responses:
                metrics = output.metrics
                print(metrics)
                num_prompt_tokens = len(output.prompt_token_ids or [])
                # print(num_prompt_tokens)
            #     print(output)
            print("\n")

            f.write("\nOutputs:\n")

            f.write(f"[{batch_size-1}]: {responses[batch_size-1].outputs[0].text}")
            
            token_ids = responses[batch_size - 1].outputs[0].token_ids
            token_gen_length = len(token_ids)
 
            f.write(f" \n\ngeneration token length: {token_gen_length}\n")



            
 
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run VLLM benchmark for a single configuration.")
    parser.add_argument('--model', required=True, help='Model name (e.g., facebook/opt-1.3b)')
    parser.add_argument('--prompt_len', type=int, required=True, help='Prompt length')
    parser.add_argument('--gen_len', type=int, required=True, help='Generation length')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--tensor_parallelism', type=int, required=True, help='Tensor parallelism')
    args = parser.parse_args()
    # print(args)
    benchmark(args.model, args.prompt_len, args.gen_len, args.batch_size, args.tensor_parallelism)
    