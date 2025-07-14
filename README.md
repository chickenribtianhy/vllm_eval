# vLLM Eval

vLLM offline inference evaluation

## Get Started

Create a new Python environment, requireing Python: 3.9 -- 3.12 \
Install vLLM with pip:

```bash
pip install vllm
```

Visit vLLM [documentation](https://docs.vllm.ai/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)

To run vllm_eval, execute the provided shell script:
```bash
./run_eval.sh
```

Modify python exec before running the script:
```bash
PYTHON_EXEC=<YOUR PYTHON EXEC>
```
Modify configuration in terms of model, prompt length, generation length, batch size and tensor parallelism to run evaluation.

Make sure the script has executable permissions. If needed, run:
```bash
chmod +x run_eval.sh
```
Runtime logs go to directory vllm\_run\_auto \
Calculated metrics go to directory vllm\_tp\_logs 