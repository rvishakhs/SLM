import torch
from contextlib import nullcontext
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

# Hyperparameters
learning_rate = 1e-4  #more stable training
max_iters = 20000
warmup_steps = 1000  #smoother initial train,
min_lr = 5e-4
eval_iters = 500
batch_size = 32
block_size = 123 

gradient_accumulation_steps = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32' : torch.float32, 'bfloat16': torch.bfloat16, 'float16' : torch.float16}[dtype]

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.set_default_device(device)
torch.manual_seed(42)


optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9) #weight decay for regularization

chedular_warmup = LinearLR(optimizer, total_iters = warmup_steps) #Implement linear warmup
chedular_decay = CosineAnnealingLR(optimizer, T_max= max_iters - warmup_steps, eta_min= min_lr)
chedular = SequentialLR(optimizer, schedulers=[chedular_warmup, chedular_decay], milestones=[warmup_steps])


# https://stackoverflow.com/questions/72534859/is-gradscaler-necessary-with-mixed-precision-training-with-pytorch
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))


