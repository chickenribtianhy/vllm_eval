# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams
import os
os.environ["HUGGINGFACE_HUB_CACHE"] = "/home/htian02/.cache/huggingface"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
gen_len = 2048
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=gen_len, min_tokens=gen_len)


def main():
    # Create an LLM.
    llm = LLM(model="facebook/opt-6.7b",
              max_model_len=4096, 
              load_format="dummy",
              )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()