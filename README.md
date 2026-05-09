# Transformer Distributed Training Benchmark
This project implements a 124M parameter Transformer from scratch and benchmarks PyTorch DDP against DeepSpeed ZeRO-1 and ZeRO-2.

## Setup
1. `docker-compose build train`
2. `docker-compose run --rm train python src/train.py --strategy=ddp`
