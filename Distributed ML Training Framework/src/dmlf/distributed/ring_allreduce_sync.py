from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from queue import Empty
from typing import Any

from .algorithms import parameter_server_average
from .mock_tensor import MockTensor


@dataclass(frozen=True)
class RingChunkMessage:
    step: int
    phase: str  # "reduce_scatter" or "all_gather"
    round_index: int
    chunk_index: int
    payload: MockTensor
    src_rank: int


@dataclass(frozen=True)
class RingWorkerReport:
    rank: int
    gradient_history: tuple[MockTensor, ...]
    parameter_history: tuple[MockTensor, ...]


@dataclass(frozen=True)
class RingAllReduceResult:
    final_parameters: MockTensor
    worker_reports: tuple[RingWorkerReport, ...]


def build_synthetic_shards(
    world_size: int, shard_size: int, parameter_dim: int
) -> tuple[tuple[MockTensor, ...], ...]:
    if world_size <= 0:
        raise ValueError("world_size must be >= 1")
    if shard_size <= 0:
        raise ValueError("shard_size must be >= 1")
    if parameter_dim <= 0:
        raise ValueError("parameter_dim must be >= 1")

    shards: list[tuple[MockTensor, ...]] = []
    for rank in range(world_size):
        samples: list[MockTensor] = []
        for sample_id in range(shard_size):
            base = float(rank * 100 + sample_id * 10)
            samples.append(
                MockTensor(tuple(base + float(dim) for dim in range(parameter_dim)))
            )
        shards.append(tuple(samples))
    return tuple(shards)


def _compute_local_gradient(parameters: MockTensor, shard: tuple[MockTensor, ...]) -> MockTensor:
    if not shard:
        raise ValueError("Worker shard must contain at least one sample")

    grads: list[MockTensor] = []
    for sample in shard:
        grads.append(parameters + sample.scale(-1.0))
    return parameter_server_average(grads)


def _validate_chunk_message(
    msg: RingChunkMessage,
    expected_step: int,
    expected_phase: str,
    expected_round: int,
    expected_src_rank: int,
) -> None:
    if msg.step != expected_step:
        raise RuntimeError(
            f"Ring message step mismatch: expected {expected_step}, got {msg.step}"
        )
    if msg.phase != expected_phase:
        raise RuntimeError(
            f"Ring message phase mismatch: expected {expected_phase}, got {msg.phase}"
        )
    if msg.round_index != expected_round:
        raise RuntimeError(
            f"Ring message round mismatch: expected {expected_round}, got {msg.round_index}"
        )
    if msg.src_rank != expected_src_rank:
        raise RuntimeError(
            f"Ring message src mismatch: expected {expected_src_rank}, got {msg.src_rank}"
        )


def _ring_allreduce_average_in_process(
    local_gradient: MockTensor,
    rank: int,
    world_size: int,
    step: int,
    send_queue: Any,
    recv_queue: Any,
) -> MockTensor:
    if len(local_gradient) % world_size != 0:
        raise ValueError(
            f"Gradient length {len(local_gradient)} must be divisible by world_size={world_size}"
        )

    chunks = local_gradient.split(world_size)

    # Reduce-scatter phase.
    for round_index in range(world_size - 1):
        send_idx = (rank - round_index) % world_size
        recv_idx = (rank - round_index - 1) % world_size

        send_queue.put(
            RingChunkMessage(
                step=step,
                phase="reduce_scatter",
                round_index=round_index,
                chunk_index=send_idx,
                payload=chunks[send_idx],
                src_rank=rank,
            )
        )

        msg = recv_queue.get()
        if not isinstance(msg, RingChunkMessage):
            raise RuntimeError("Worker received non-ring message during reduce-scatter")
        expected_src_rank = (rank - 1) % world_size
        _validate_chunk_message(
            msg,
            expected_step=step,
            expected_phase="reduce_scatter",
            expected_round=round_index,
            expected_src_rank=expected_src_rank,
        )
        if msg.chunk_index != recv_idx:
            raise RuntimeError(
                f"Reduce-scatter chunk mismatch: expected {recv_idx}, got {msg.chunk_index}"
            )

        chunks[recv_idx] = chunks[recv_idx] + msg.payload

    owned_idx = (rank - (world_size - 1)) % world_size
    chunks[owned_idx] = chunks[owned_idx].scale(1.0 / world_size)

    known_chunks: dict[int, MockTensor] = {owned_idx: chunks[owned_idx]}
    send_idx = owned_idx

    # All-gather phase.
    for round_index in range(world_size - 1):
        send_queue.put(
            RingChunkMessage(
                step=step,
                phase="all_gather",
                round_index=round_index,
                chunk_index=send_idx,
                payload=known_chunks[send_idx],
                src_rank=rank,
            )
        )

        msg = recv_queue.get()
        if not isinstance(msg, RingChunkMessage):
            raise RuntimeError("Worker received non-ring message during all-gather")
        expected_src_rank = (rank - 1) % world_size
        _validate_chunk_message(
            msg,
            expected_step=step,
            expected_phase="all_gather",
            expected_round=round_index,
            expected_src_rank=expected_src_rank,
        )

        known_chunks[msg.chunk_index] = msg.payload
        send_idx = msg.chunk_index

    if len(known_chunks) != world_size:
        raise RuntimeError(
            f"Rank {rank} reconstructed {len(known_chunks)} chunks, expected {world_size}"
        )

    ordered = [known_chunks[idx] for idx in range(world_size)]
    return MockTensor.concat(ordered)


def _ring_worker_process(
    rank: int,
    world_size: int,
    num_steps: int,
    learning_rate: float,
    initial_parameters: MockTensor,
    shard: tuple[MockTensor, ...],
    send_queue: Any,
    recv_queue: Any,
    report_queue: Any,
) -> None:
    parameters = initial_parameters
    gradient_history: list[MockTensor] = []
    parameter_history: list[MockTensor] = []

    for step in range(num_steps):
        local_gradient = _compute_local_gradient(parameters, shard)
        gradient_history.append(local_gradient)

        avg_gradient = _ring_allreduce_average_in_process(
            local_gradient=local_gradient,
            rank=rank,
            world_size=world_size,
            step=step,
            send_queue=send_queue,
            recv_queue=recv_queue,
        )

        parameters = parameters + avg_gradient.scale(-learning_rate)
        parameter_history.append(parameters)

    report_queue.put(
        RingWorkerReport(
            rank=rank,
            gradient_history=tuple(gradient_history),
            parameter_history=tuple(parameter_history),
        )
    )


def run_synchronous_ring_allreduce_training(
    world_size: int,
    num_steps: int,
    learning_rate: float,
    parameter_dim: int,
    shard_size: int,
    join_timeout_sec: float = 30.0,
) -> RingAllReduceResult:
    if world_size <= 0:
        raise ValueError("world_size must be >= 1")
    if num_steps <= 0:
        raise ValueError("num_steps must be >= 1")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if parameter_dim % world_size != 0:
        raise ValueError(
            f"parameter_dim={parameter_dim} must be divisible by world_size={world_size}"
        )

    shards = build_synthetic_shards(world_size, shard_size, parameter_dim)
    initial_parameters = MockTensor([0.0] * parameter_dim)

    ctx = mp.get_context("spawn")
    ring_queues = [ctx.Queue() for _ in range(world_size)]
    report_queue = ctx.Queue()

    workers: list[Any] = []
    for rank in range(world_size):
        send_queue = ring_queues[rank]
        recv_queue = ring_queues[(rank - 1) % world_size]
        proc = ctx.Process(
            target=_ring_worker_process,
            args=(
                rank,
                world_size,
                num_steps,
                learning_rate,
                initial_parameters,
                shards[rank],
                send_queue,
                recv_queue,
                report_queue,
            ),
        )
        workers.append(proc)

    for proc in workers:
        proc.start()

    for proc in workers:
        proc.join(timeout=join_timeout_sec)
        if proc.exitcode is None:
            proc.terminate()
            raise RuntimeError(f"Ring worker process {proc.pid} timed out")
        if proc.exitcode != 0:
            raise RuntimeError(f"Ring worker process {proc.pid} exited with code {proc.exitcode}")

    reports: list[RingWorkerReport] = []
    for _ in range(world_size):
        try:
            report = report_queue.get(timeout=2.0)
        except Empty as exc:
            raise RuntimeError("Did not receive all ring worker reports") from exc
        reports.append(report)

    reports.sort(key=lambda item: item.rank)

    reference = reports[0].parameter_history[-1]
    for report in reports[1:]:
        candidate = report.parameter_history[-1]
        if not candidate.almost_equal(reference, tol=1e-9):
            raise RuntimeError("Workers diverged after ring all-reduce")

    return RingAllReduceResult(final_parameters=reference, worker_reports=tuple(reports))


def serial_ring_reference(
    world_size: int,
    num_steps: int,
    learning_rate: float,
    parameter_dim: int,
    shard_size: int,
) -> MockTensor:
    shards = build_synthetic_shards(world_size, shard_size, parameter_dim)
    parameters = MockTensor([0.0] * parameter_dim)

    for _ in range(num_steps):
        gradients = [_compute_local_gradient(parameters, shard) for shard in shards]
        avg_gradient = parameter_server_average(gradients)
        parameters = parameters + avg_gradient.scale(-learning_rate)

    return parameters
