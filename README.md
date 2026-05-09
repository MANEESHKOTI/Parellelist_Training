# Swarm Architecture: Distributed Training Pipeline

## Project Overview
This repository contains a from-scratch PyTorch implementation of a 124M parameter Decoder-only Transformer (GPT-2 Small architecture). It features a production-grade distributed training and benchmarking pipeline designed to evaluate the performance, memory footprint, and throughput of standard PyTorch Distributed Data Parallel (DDP) versus Microsoft's DeepSpeed Zero Redundancy Optimizer (ZeRO) Stages 1 and 2.

## Repository Structure
* `src/`: Contains the core Transformer architecture (`model.py`), data loading utilities (`data.py`), instrumentation tracking (`metrics.py`), and the unified distributed training loop (`train.py`).
* `configs/`: DeepSpeed JSON configuration files for ZeRO-1 and ZeRO-2 optimization.
* `scripts/`: Python utilities for parsing benchmark telemetry and generating visualization plots.
* `outputs/`: Automatically generated directory containing run logs, `metrics.json` telemetry, and generated plots.
* `ENGINEERS_GUIDE.md`: The final performance analysis and decision framework based on the benchmark sprints.

## Prerequisites & Setup
This project is fully containerized to ensure reproducible execution across multi-GPU environments.
1. Ensure Docker and NVIDIA Container Toolkit are installed.
2. Build the training environment:
   ```bash
   docker-compose build trainExecution Guide
1. The Benchmark Sprints

To reproduce the core benchmarking data, run the following commands sequentially. Each run is capped at 50 steps for rapid profiling.

PyTorch DDP Baseline:

Bash
docker-compose run --rm train torchrun --nproc_per_node=2 src/train.py --strategy=ddp
DeepSpeed ZeRO Stage 1:

Bash
docker-compose run --rm train deepspeed --num_gpus=2 src/train.py --strategy=zero1 --deepspeed_config configs/ds_config_zero1.json
DeepSpeed ZeRO Stage 2:

Bash
docker-compose run --rm train deepspeed --num_gpus=2 src/train.py --strategy=zero2 --deepspeed_config configs/ds_config_zero2.json
2. Gradient Explosion & Mitigation Simulation

To test the pipeline's stability features, you can force a numerical explosion and verify the gradient clipping mitigation:

Force Explosion:

Bash
docker-compose run --rm train torchrun --nproc_per_node=2 src/train.py --strategy=ddp --explode-grads
Mitigate via Clipping:

Bash
docker-compose run --rm train torchrun --nproc_per_node=2 src/train.py --strategy=ddp --explode-grads --clip-grad-norm=1.0
3. Generate Telemetry Visualizations

Once all benchmarks have completed successfully, run the plotting script to generate the memory and throughput comparisons:

Bash
docker-compose run --rm train python scripts/generate_plots.py
Plots will be saved to outputs/plots/.
