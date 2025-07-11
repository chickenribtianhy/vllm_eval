from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    llm = LLM(model="facebook/opt-125m", disable_log_stats=False)
    outputs = llm.generate(prompts, sampling_params)
    print(outputs)

if __name__ == "__main__":
    main()