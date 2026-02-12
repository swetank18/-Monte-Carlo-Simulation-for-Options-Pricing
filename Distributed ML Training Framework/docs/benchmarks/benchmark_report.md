# Benchmark Report

## Summary
- `parameter_server` peak throughput: 432.28 samples/s at world_size=4
- `ring_allreduce` peak throughput: 453.00 samples/s at world_size=4

## What Conclusions Are Valid
- Synchronous parameter-server and ring all-reduce implementations are numerically equivalent to the serial baseline for this deterministic synthetic workload.
- Measured throughput and scaling efficiency reflect framework-process overhead and synchronization behavior on a single machine.
- Estimated communication overhead clearly separates centralized PS traffic from ring-collective traffic growth patterns.

## What Conclusions Are Not Valid
- Results do not represent multi-node network behavior (no real NIC, no packet loss, no cross-machine latency).
- Results do not represent GPU behavior or PyTorch tensor-kernel performance.
- Convergence claims are limited to this synthetic objective and synchronous update rule, not real deep model training dynamics.