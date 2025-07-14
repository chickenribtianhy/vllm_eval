from transformers import AutoConfig

model_name = "facebook/opt-6.7b"

# Load and patch config
config = AutoConfig.from_pretrained(model_name)
config.max_position_embeddings = 4096  # increase maximum sequence length

# Save patched config locally
config.save_pretrained("./opt_config/opt-6.7b")