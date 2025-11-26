# Config for training on Lambda Labs A100 (80GB)
# Full-scale model for production training

# I/O
out_dir = 'checkpoints'
eval_interval = 500
log_interval = 10
eval_iters = 100
always_save_checkpoint = True

# wandb logging
wandb_log = True
wandb_project = 'ouroboros'
wandb_run_name = 'a100-full'

# data
data_dir = 'data/processed'
batch_size = 64  # A100 can handle larger batches

# model - full scale
n_layer = 12
n_head = 8
n_embd = 512
dropout = 0.1
max_seq_len = 512

# optimizer
learning_rate = 3e-4
max_iters = 5000
weight_decay = 0.1
grad_clip = 1.0

# learning rate decay
decay_lr = True
warmup_iters = 200
lr_decay_iters = 5000
min_lr = 3e-5

# system
device = 'cuda'
dtype = 'bfloat16'  # A100 supports bfloat16
compile_model = True  # Use torch.compile for speedup
