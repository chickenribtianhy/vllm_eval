import argparse
import torch
import time
import os
from itertools import product
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr

# os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["VLLM_USE_V1"] = "0"
base_prompt = "Paris is the capital of "

log_dir = "./vllm_tp_logs"

os.makedirs(log_dir, exist_ok=True)
 
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


def benchmark(model_name, prompt_len, gen_len, batch_size, tensor_parallelism):
    print(f"benchmarking {model_name.replace('/', '_')}_prompt{prompt_len}_gen{gen_len}_bs{batch_size}")
    log_file = os.path.join(log_dir, f"{model_name.replace('/', '_')}_prompt{prompt_len}_gen{gen_len}_bs{batch_size}.log")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = [prefix_pad_prompt(base_prompt, prompt_len, tokenizer) for _ in range(batch_size)]
    
    # warm_up_params = SamplingParams(temperature=0.9, max_tokens=10, min_tokens=10)
    sampling_params = SamplingParams(temperature=0, max_tokens=gen_len, min_tokens=gen_len)
    # prefill_params = SamplingParams(temperature=0.9, max_tokens=1, best_of=3)
    # decode_params = SamplingParams(temperature=0.9, max_tokens=gen_len, min_tokens=gen_len, best_of=3)
    # test_params = SamplingParams(temperature=0.8, top_p=0.95)

    # num_gpus = torch.cuda.device_count()

    _cpu_offload = 0
    if model_name == "facebook/opt-30b":
        _cpu_offload = 19 # for TP=2
    llm = LLM(model=model_name,
                tensor_parallel_size=tensor_parallelism,
                gpu_memory_utilization=0.90,
                max_num_batched_tokens=4096,
                # max_num_batched_tokens=prompt_len*batch_size,
                max_num_seqs=32,
                disable_log_stats=False,
                swap_space=50,
                cpu_offload_gb=_cpu_offload,
                preemption_mode="swap",
                dtype="float16",
                # enable_chunked_prefill=False,
                # max_model_len=4096, 
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
    