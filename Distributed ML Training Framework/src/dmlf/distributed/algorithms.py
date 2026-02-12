from __future__ import annotations

from .mock_tensor import MockTensor


def parameter_server_average(gradients: list[MockTensor]) -> MockTensor:
    """Aggregate gradients by arithmetic mean (parameter server style)."""
    if not gradients:
        raise ValueError("At least one gradient is required for aggregation.")

    running = gradients[0]
    for grad in gradients[1:]:
        running = running + grad

    return running.scale(1.0 / len(gradients))


def ring_allreduce_average(gradients: list[MockTensor]) -> list[MockTensor]:
    """Simulate ring all-reduce average over one tensor per rank."""
    if not gradients:
        raise ValueError("At least one gradient is required for all-reduce.")

    world_size = len(gradients)
    if world_size == 1:
        return [gradients[0]]

    chunks = [g.split(world_size) for g in gradients]

    # Reduce-scatter: each rank accumulates one chunk index over world_size-1 steps.
    reduce_buffers = [rank_chunks[:] for rank_chunks in chunks]
    for step in range(world_size - 1):
        next_buffers = [rank_chunks[:] for rank_chunks in reduce_buffers]
        for rank in range(world_size):
            src_rank = (rank - 1) % world_size
            chunk_idx = (rank - step - 1) % world_size
            next_buffers[rank][chunk_idx] = (
                reduce_buffers[rank][chunk_idx] + reduce_buffers[src_rank][chunk_idx]
            )
        reduce_buffers = next_buffers

    owned_chunk_idx = [
        (rank - (world_size - 1)) % world_size for rank in range(world_size)
    ]
    known_chunks: list[dict[int, MockTensor]] = []
    for rank in range(world_size):
        known_chunks.append(
            {
                owned_chunk_idx[rank]: reduce_buffers[rank][owned_chunk_idx[rank]].scale(
                    1.0 / world_size
                )
            }
        )

    # All-gather: circulate reduced chunks so every rank reconstructs full tensor.
    for step in range(world_size - 1):
        incoming: list[tuple[int, MockTensor]] = []
        for rank in range(world_size):
            src_rank = (rank - 1) % world_size
            recv_idx = (rank - (world_size - 1) - step - 1) % world_size
            incoming.append((recv_idx, known_chunks[src_rank][recv_idx]))

        for rank, (idx, chunk) in enumerate(incoming):
            known_chunks[rank][idx] = chunk

    result: list[MockTensor] = []
    for rank in range(world_size):
        ordered = [known_chunks[rank][idx] for idx in range(world_size)]
        result.append(MockTensor.concat(ordered))

    return result
