import torch
import time
import os
from itertools import product
from vllm import LLM, SamplingParams
# from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Vector
from transformers import AutoTokenizer
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
 

os.environ["VLLM_USE_V1"] = "0"
# os.environ["VLLMHOME"] = "$PROJECT/vllm/vllmevaluation"

# prompt_lengths = [128, 1024, 2048]
# gen_lengths = [128, 512, 1024, 2048]
# batch_sizes = [1, 4, 8, 16, 32]

prompt_lengths = [128]
gen_lengths = [128]
batch_sizes = [1]
 
# models = ["facebook/opt-1.3b", "facebook/opt-13b", "facebook/opt-30b"]
models = ["facebook/opt-125m"]
 
base_prompt = "Paris is the capital of "
 
log_dir = "./vllm_tp_logs"
 
#log_dir =".vllm_logs"
 
os.makedirs(log_dir, exist_ok=True)
 
def prefix_pad_prompt(prompt, target_len, tokenizer):
    input_ids = tokenizer(prompt)["input_ids"]
    prompt_len = len(input_ids)
 
    pad_token = tokenizer.eos_token or tokenizer.pad_token or "<pad>"
    pad_token_id = tokenizer(pad_token)["input_ids"][0]
 
    if prompt_len < target_len:
        padding = [pad_token_id] * (target_len - prompt_len)
        padded_ids = padding + input_ids
    else:
        padded_ids = input_ids[-target_len:]
 
    return tokenizer.decode(padded_ids, skip_special_tokens=False)
 
def benchmark(model_name, prompt_len, gen_len, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = [prefix_pad_prompt(base_prompt, prompt_len, tokenizer) for _ in range(batch_size)]
 
    warm_up_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=10, min_tokens=10)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=gen_len, min_tokens=gen_len)
 
    log_file = os.path.join(log_dir, f"{model_name.replace('/', '_')}_prompt{prompt_len}_gen{gen_len}_bs{batch_size}.log")
    
    num_gpus = torch.cuda.device_count()
 
    llm = LLM(model=model_name,
                tensor_parallel_size=num_gpus,
                gpu_memory_utilization=0.10,
                max_num_batched_tokens=4096,
                max_num_seqs=32,
                disable_log_stats=False,
                swap_space=25,
                cpu_offload_gb=50.0,
                preemption_mode="swap",
                dtype="float16")
 
    _ = llm.generate(prompts, warm_up_params)
    
    with open(log_file, "w") as f:
        with redirect_stdout(f), redirect_stderr(f):
            f.write(f"Benchmarking {model_name} | Prompt Len: {prompt_len}, Gen Len: {gen_len}, Batch Size: {batch_size}\n")
            f.write(f"Start time: {datetime.now()}\n\n")
 
 
            start = time.perf_counter()
            responses = llm.generate(prompts, sampling_params)
            end = time.perf_counter()
            runtime = end - start
 
            f.write(f"Runtime: {runtime:.9f} seconds\n\n")
            f.write(f"throughput: {((prompt_len+gen_len)*batch_size)/runtime} tokens/sec")

            f.write("\nOutputs:\n")
 
            '''
            for output in outputs:
                f.write(f"{output.outputs.text}\n")
            '''
            f.write(f"[{batch_size-1}]: {responses[batch_size-1].outputs[0].text}")
            
            token_ids = responses[batch_size - 1].outputs[0].token_ids
            token_gen_length = len(token_ids)
 
            f.write(f" \n\ngeneration token length: {token_gen_length}\n")
 
if __name__ == "__main__":
    for model_name, prompt_len, gen_len, batch_size in product(models, prompt_lengths, gen_lengths, batch_sizes):
        try:
            benchmark(model_name, prompt_len, gen_len, batch_size)
        except Exception as e:
            err_log = os.path.join(log_dir, "errors.log")
            with open(err_log, "a") as ef:
                ef.write(f"Failed for {model_name} | prompt_len={prompt_len}, gen_len={gen_len}, bs={batch_size}\n")
                ef.write(f"{str(e)}\n\n")
 
