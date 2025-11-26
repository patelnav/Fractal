# Config for training on M2 Mac (MPS)
# Smaller model, shorter training for local testing

# I/O
out_dir = 'checkpoints'
eval_interval = 200
log_interval = 10
eval_iters = 50
always_save_checkpoint = True

# data
data_dir = 'data/processed'
batch_size = 16

# model - smaller for M2
n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.1
max_seq_len = 512

# optimizer
learning_rate = 3e-4
max_iters = 2000
weight_decay = 0.1
grad_clip = 1.0

# learning rate decay
decay_lr = True
warmup_iters = 100
lr_decay_iters = 2000
min_lr = 3e-5

# system
device = 'mps'
dtype = 'float32'  # MPS doesn't support bfloat16
compile_model = False  # torch.compile doesn't work well on MPS yet
