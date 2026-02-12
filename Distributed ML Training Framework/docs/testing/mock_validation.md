# Distributed Aggregation Validation

## What Is Validated

- Deterministic arithmetic for tensor-like gradient containers (`MockTensor`).
- Parameter-server style averaging semantics in pure Python.
- Ring all-reduce synchronization and averaging semantics in pure Python.
- Torch-backed parity checks on CPU:
  - `parameter_server_average` matches `torch.mean` across worker gradients.
  - `ring_allreduce_average` rank outputs match `torch.mean`.
- Input validation paths (empty gradients, shape mismatch, invalid chunking).

## What Is Not Validated

- Autograd graph correctness and optimizer integration behavior in real training steps.
- CUDA execution behavior, kernel determinism, and GPU communication runtime effects.
- Backend communication stack behavior (NCCL, Gloo, RPC/TCP retries, partial failures).
- Multi-process timing behavior (stragglers, synchronization stalls, clock skew).

## Why This Layer Exists

Mock tests isolate algorithmic logic from framework/runtime dependencies and catch aggregation bugs early. Torch parity tests then anchor that logic to actual tensor arithmetic on CPU when torch is available.
