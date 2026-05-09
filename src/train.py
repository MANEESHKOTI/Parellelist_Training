import torch, argparse, time, deepspeed, os, json

from src.model import TransformerDecoder
from src.data import get_dataloader

parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, required=True, choices=['ddp', 'zero1', 'zero2'])
parser.add_argument("--deepspeed_config", type=str, default=None)
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--explode-grads", action="store_true")
parser.add_argument("--clip-grad-norm", type=float, default=None)
args = parser.parse_args()

# Setup paths based on strategy or explosion simulation
out_dir = f"outputs/{args.strategy}"
if args.explode_grads and args.clip_grad_norm: out_dir = "outputs/ddp_explode_clipped"
elif args.explode_grads: out_dir = "outputs/ddp_explode"
os.makedirs(out_dir, exist_ok=True)

# 1. Determine Local Rank & Device
local_rank = args.local_rank if args.local_rank != -1 else int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# 2. Initialize Strategy
if args.strategy == "ddp":
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    model = TransformerDecoder().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
else:
    model_base = TransformerDecoder().to(device)
    model, optimizer, _, _ = deepspeed.initialize(args=args, model=model_base, model_parameters=model_base.parameters())

loader = get_dataloader(batch_size=4)
metrics_history = []
log_file = open(f"{out_dir}/training.log", "w") if local_rank == 0 else None

def log_print(msg):
    if local_rank == 0:
        print(msg)
        log_file.write(msg + "\n")

model.train()
for step, (x, y) in enumerate(loader):
    if step >= 50: break 
    start = time.time()
    x, y = x.to(device), y.to(device)
    
    optimizer.zero_grad()
    
    # Forward Pass
    logits, loss = model(x, targets=y)
    
    # Simulate Explosion
    if args.explode_grads: loss = loss * 10000.0
    
    # Backward Pass & Sync Timing
    sync_start = time.time()
    if args.strategy == "ddp":
        loss.backward()
    else:
        model.backward(loss)
    comm_overhead = (time.time() - sync_start) * 1000 # ms
    
    # Gradient Norm & Clipping
    if args.strategy == "ddp":
        unclipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        if args.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()
    else:
        unclipped_norm = torch.tensor(0.0) # DeepSpeed handles clipping internally via config
        model.step()
    
    # Logging
    if local_rank == 0:
        grad_norm_val = float(unclipped_norm.item())
        metrics_history.append({
            "step": step,
            "loss": float(loss.item()),
            "grad_norm": grad_norm_val,
            "gpu_memory_mb": torch.cuda.max_memory_allocated() / 1e6,
            "comm_overhead_ms": comm_overhead,
            "step_time_s": time.time() - start
        })
        
        log_msg = f"Step {step} | Loss: {loss.item():.4f} | Grad Norm: {grad_norm_val:.2f}"
        if args.clip_grad_norm:
            log_msg = f"Step {step} | Grad Norm (pre-clip): {grad_norm_val:.2f} | Grad Norm (post-clip): {args.clip_grad_norm}"
        log_print(log_msg)

if local_rank == 0:
    with open(f"{out_dir}/metrics.json", 'w') as f:
        json.dump(metrics_history, f)
    log_file.close()
