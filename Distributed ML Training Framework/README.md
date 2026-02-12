# Distributed ML Training Framework

A research-grade, correctness-first distributed training framework in Python, inspired by the design space of PyTorch Distributed and Horovod.

This repository focuses on clear, testable implementations of core distributed training protocols before performance optimizations:
- deterministic single-node baseline
- synchronous Parameter Server training
- synchronous Ring All-Reduce training
- checkpointing and failure recovery
- benchmark harness with plots and interpretation

## Goals

- correctness over speed
- explicit failure handling and assumptions
- modular design that can evolve into production-grade components

## Current Features

### 1) Deterministic Single-Node Baseline
- MNIST MLP baseline training pipeline
- seeded deterministic behavior
- modular separation of config, model, and trainer

Key files:
- `src/dmlf/baseline/config.py`
- `src/dmlf/baseline/model.py`
- `src/dmlf/baseline/trainer.py`
- `scripts/train_mnist_baseline.py`

### 2) Dependency Guardrails
- explicit runtime guards for optional dependencies (`torch`, `torchvision`)
- actionable error messages when dependencies are missing

Key file:
- `src/dmlf/env.py`

### 3) Mock-Tensor Correctness Layer
- torch-free unit-testable tensor abstraction (`MockTensor`)
- parameter-server aggregation and ring all-reduce logic tests

Key files:
- `src/dmlf/distributed/mock_tensor.py`
- `src/dmlf/distributed/algorithms.py`

### 4) Gradient Interception via Autograd Hooks
- per-parameter gradient capture immediately after backward
- structured snapshots for later synchronization layers

Key file:
- `src/dmlf/distributed/gradient_interceptor.py`

### 5) Multiprocessing Worker Simulation
- single-machine process simulation with data sharding
- independent worker replicas and local gradient computation

Key file:
- `src/dmlf/baseline/trainer.py`

### 6) Synchronous Parameter Server Runtime
- one server process + multiple worker processes
- strict step/rank validation
- periodic checkpointing and restart-based recovery
- checksum validation to prevent silent checkpoint corruption

Key file:
- `src/dmlf/distributed/parameter_server_sync.py`

### 7) Synchronous Ring All-Reduce Runtime
- multi-process ring topology
- reduce-scatter + all-gather with chunk metadata checks
- strict message validation (step/phase/round/source)

Key file:
- `src/dmlf/distributed/ring_allreduce_sync.py`

### 8) Benchmarking Suite + Plots
- throughput
- scaling efficiency
- convergence vs serial baseline
- estimated communication overhead

Key files:
- `scripts/run_distributed_benchmarks.py`
- `experiments/benchmarks/results/*`
- `docs/benchmarks/benchmark_report.md`

## Repository Structure

```text
Distributed ML Training Framework/
  configs/
  docs/
  experiments/
  scripts/
  src/dmlf/
    baseline/
    distributed/
  tests/
```

## Getting Started

### Requirements
- Python 3.11+
- `matplotlib` for benchmark plotting
- Optional: `torch`, `torchvision` for baseline training and hook-based tests

### Run tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Notes:
- torch-dependent tests are skipped automatically when torch is not installed.

### Run baseline training

```bash
python scripts/train_mnist_baseline.py --epochs 3
```

### Run synchronous Parameter Server simulation

```bash
python scripts/run_sync_parameter_server.py --world-size 4 --num-steps 5 --parameter-dim 40 --shard-size 16
```

### Run synchronous Ring All-Reduce simulation

```bash
python scripts/run_sync_ring_allreduce.py --world-size 4 --num-steps 6 --parameter-dim 8 --shard-size 5
```

### Run benchmarks and generate plots

```bash
python scripts/run_distributed_benchmarks.py
```

Outputs are written to:
- `experiments/benchmarks/results/`
- `docs/benchmarks/benchmark_report.md`

## Safety and Correctness Guarantees (Current Scope)

- deterministic synthetic-data protocol checks for distributed algorithms
- strict message validation in distributed runtime paths
- checkpoint checksum validation to detect corruption
- explicit hard-failure behavior for malformed protocol states

## Known Limitations

- CPU-only simulated communication (multiprocessing queues)
- no real multi-node networking stack in current implementation
- no GPU/NCCL/Gloo transport integration yet
- convergence claims are bounded to synthetic objective tests and baseline checks

## Roadmap

- integrate real PyTorch model gradient synchronization path over current protocols
- add robust coordinator for elastic membership and retries
- introduce transport abstraction (local queue, TCP, and process-group backends)
- extend checkpoint state to optimizer/runtime metadata in PyTorch path

## License

Add your preferred license in `LICENSE` (currently not set).
