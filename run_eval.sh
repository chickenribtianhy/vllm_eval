#!/bin/bash
# set -x

# please specify cache directory
# export HUGGINGFACE_HUB_CACHE=/home/ubuntu/.cache/huggingface
# export VLLM_CACHE_DIR=/home/ubuntu/.cache/vllm

# please specify python
# PYTHON_EXEC=$(which python3) 
PYTHON_EXEC=/home/ubuntu/miniconda3/envs/vllm/bin/python


# configuration
# models=("facebook/opt-6.7b")
# prompt_lengths=(128 1024 2048)
# gen_lengths=(128 512 1024 2048)
# batch_sizes=(1 4 8 16 32)

models=("facebook/opt-6.7b")
prompt_lengths=(128)
gen_lengths=(128)
batch_sizes=(1)

tensor_parallelism=1

for model in "${models[@]}"; do
    model_safe=${model//\//_}
    for prompt_len in "${prompt_lengths[@]}"; do
        for gen_len in "${gen_lengths[@]}"; do
            for batch_size in "${batch_sizes[@]}"; do
                echo "Running configuration: model=$model, prompt_len=$prompt_len, gen_len=$gen_len, batch_size=$batch_size"
                OMP_NUM_THREADS=16 $PYTHON_EXEC main.py \
                    --model "$model" \
                    --prompt_len "$prompt_len" \
                    --gen_len "$gen_len" \
                    --batch_size "$batch_size" \
                    --tensor_parallelism "$tensor_parallelism" \
                    > vllm_run_auto/vllm_run_${model_safe}_${prompt_len}_${gen_len}_${batch_size}.log \
                    2>&1 
            done
        done
    done
done
