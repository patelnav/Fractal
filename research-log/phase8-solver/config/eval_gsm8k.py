# Config for GSM8K evaluation (Phase 8)
# Evaluates Ouroboros as a solution ranker

# Paths
checkpoint_path = 'checkpoints/ckpt.pt'
data_path = 'data/gsm8k/test.json'
output_dir = 'results/gsm8k_eval'
cache_dir = 'cache/generations'

# Evaluation settings
n_candidates = 5  # Number of candidates to generate per problem
max_problems = None  # None = all problems, set to int for subset
batch_size = 16  # Batch size for energy scoring

# Generator settings
generator = {
    # Options: 'vllm', 'huggingface', 'openai', 'dummy'
    'type': 'huggingface',  # Use HuggingFace with batched generation

    # HuggingFace settings
    'model_name': 'google/gemma-3-1b-it',  # 62.8% GSM8K, 1B params, fast
    'max_new_tokens': 256,
    'temperature': 0.7,
    'top_p': 0.9,
    'do_sample': True,

    # OpenAI settings (if type='openai')
    # 'model': 'gpt-3.5-turbo',
    # 'max_tokens': 256,

    # Caching
    'cache_enabled': True,
    'cache_dir': 'cache/generations',
}

# Device
device = 'auto'  # 'auto', 'cpu', 'cuda', 'mps'

# Reproducibility
seed = 42

# Logging
verbose = True
save_per_problem_results = True
