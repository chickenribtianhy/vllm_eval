"""
Benchmark DistServe on a batch of requests, which have identical input&output length
"""
import sys, os, math
import random
import torch

import numpy as np

from structs import *
from run_test_params import run_test_params
        
example_testing_params = [
    TestParamGroup(
        worker_param = WorkerParam(
            model_dir = "facebook/opt-13b",
            tp_world_size = 1,
            max_req_num = 1024,
            max_seq_len = 2048,
            use_dummy_weights = True
        ),
        input_params = [
            InputParam(
                batch_size = 1,
                input_len = 2000,
                output_len = 16,
            )
        ]
    )
]

def get_profiling_params() -> list[TestParamGroup]:
    return [
        TestParamGroup(
            worker_param = WorkerParam(
                model_dir = model,
                tp_world_size = tp_world_size,
                max_req_num = 256,
                max_seq_len = 2048,
                use_dummy_weights = True
            ),
            input_params = [
                InputParam(
                    batch_size = batch_size,
                    input_len = input_len,
                    output_len = 16
                )
                for (batch_size, input_len) in [
                    (batch_size, input_len)
                    for batch_size in [1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192]
                    for input_len in [4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 284, 512, 768, 1024, 1536, 2020]
                    if batch_size * ((input_len+15)//16*16) <= num_tokens_limit
                ]
            ]
        )
        for (model, tp_world_size, num_tokens_limit) in [
            ("facebook/opt-13b", 1, 49152),
            ("facebook/opt-13b", 2, 98304),
            ("facebook/opt-13b", 4, 250000),
            ("facebook/opt-66b", 2, 8192),
            ("facebook/opt-66b", 3, 36864),
            ("facebook/opt-66b", 4, 65000),
            # ("intlsy/opt-175b-hyperparam", 8, 40000)
        ]
    ]
    
def run_vllm(test_params: list[TestParamGroup], **kwargs):
    import ray
    from sut.sut_vllm import VLLMSUT
    ray.init(ignore_reinit_error=True)
    run_test_params(VLLMSUT, "db-identical-req-vllm.sqlite", test_params, **kwargs)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    
    if len(sys.argv) == 1:
        print(f"Usage: {sys.argv[0]} <test_group_name>")
        sys.exit(1)
    
    test_group_candidates = {
        "vllm-example": lambda : run_vllm(example_testing_params, warmup_rounds=1, measure_rounds=1, skip_duplicated=False, store_into_db=False),
        "vllm-profiling": lambda : run_vllm(get_profiling_params(), warmup_rounds=2, measure_rounds=3),
    }
    select_test_group = sys.argv[1]
    if select_test_group not in test_group_candidates:
        print(f"Wrong test group name! Available test group names: {list(test_group_candidates.keys())}")
        sys.exit(1)

    print(f"Selected test group: {select_test_group}")
    test_group_candidates[select_test_group]()
    