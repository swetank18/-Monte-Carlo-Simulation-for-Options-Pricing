from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import tempfile
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

from .algorithms import parameter_server_average
from .mock_tensor import MockTensor


@dataclass(frozen=True)
class ParameterMessage:
    step: int
    parameters: MockTensor
    phase: str  # "forward" or "updated"


@dataclass(frozen=True)
class GradientMessage:
    worker_rank: int
    step: int
    gradient: MockTensor


@dataclass(frozen=True)
class WorkerReport:
    rank: int
    gradient_history: tuple[MockTensor, ...]
    parameter_history: tuple[MockTensor, ...]


@dataclass(frozen=True)
class ServerReport:
    parameter_history: tuple[MockTensor, ...]
    written_checkpoints: tuple[str, ...]


@dataclass(frozen=True)
class SyncPSTrainingResult:
    final_parameters: MockTensor
    server_parameter_history: tuple[MockTensor, ...]
    worker_reports: tuple[WorkerReport, ...]
    restart_count: int
    last_checkpoint_path: str | None


@dataclass(frozen=True)
class CheckpointState:
    next_step: int
    parameters: MockTensor
    path: str


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


def _validate_gradient_batch(
    messages: list[GradientMessage], expected_step: int, world_size: int
) -> None:
    if len(messages) != world_size:
        raise RuntimeError(
            f"Expected {world_size} gradients, received {len(messages)} at step {expected_step}"
        )

    seen_ranks: set[int] = set()
    for msg in messages:
        if msg.step != expected_step:
            raise RuntimeError(
                f"Gradient step mismatch: expected {expected_step}, got {msg.step} from rank {msg.worker_rank}"
            )
        if msg.worker_rank in seen_ranks:
            raise RuntimeError(
                f"Duplicate gradient from rank {msg.worker_rank} at step {expected_step}"
            )
        seen_ranks.add(msg.worker_rank)


def _checkpoint_payload(
    next_step: int,
    parameters: MockTensor,
    world_size: int,
    parameter_dim: int,
    shard_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    return {
        "version": 1,
        "next_step": next_step,
        "parameters": list(parameters.values),
        "world_size": world_size,
        "parameter_dim": parameter_dim,
        "shard_size": shard_size,
        "learning_rate": learning_rate,
    }


def _checkpoint_checksum(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_checkpoint(
    checkpoint_dir: str,
    next_step: int,
    parameters: MockTensor,
    world_size: int,
    parameter_dim: int,
    shard_size: int,
    learning_rate: float,
) -> str:
    payload = _checkpoint_payload(
        next_step=next_step,
        parameters=parameters,
        world_size=world_size,
        parameter_dim=parameter_dim,
        shard_size=shard_size,
        learning_rate=learning_rate,
    )
    record = {
        "payload": payload,
        "checksum": _checkpoint_checksum(payload),
    }

    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"checkpoint_step_{next_step:08d}.json"
    tmp_target = target.with_suffix(".tmp")
    with tmp_target.open("w", encoding="utf-8") as f:
        json.dump(record, f, sort_keys=True)
    tmp_target.replace(target)
    return str(target)


def _parse_checkpoint_step(path: Path) -> int:
    stem = path.stem
    prefix = "checkpoint_step_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected checkpoint filename: {path.name}")
    return int(stem[len(prefix) :])


def _validate_checkpoint_record(
    record: dict[str, Any],
    world_size: int,
    parameter_dim: int,
    shard_size: int,
    learning_rate: float,
) -> CheckpointState:
    if "payload" not in record or "checksum" not in record:
        raise RuntimeError("Checkpoint missing payload/checksum fields")

    payload = record["payload"]
    checksum = record["checksum"]
    if not isinstance(payload, dict) or not isinstance(checksum, str):
        raise RuntimeError("Checkpoint format is invalid")

    expected_checksum = _checkpoint_checksum(payload)
    if checksum != expected_checksum:
        raise RuntimeError("Checkpoint checksum mismatch; possible corruption detected")

    if payload.get("world_size") != world_size:
        raise RuntimeError("Checkpoint world_size does not match current run")
    if payload.get("parameter_dim") != parameter_dim:
        raise RuntimeError("Checkpoint parameter_dim does not match current run")
    if payload.get("shard_size") != shard_size:
        raise RuntimeError("Checkpoint shard_size does not match current run")
    if float(payload.get("learning_rate")) != float(learning_rate):
        raise RuntimeError("Checkpoint learning_rate does not match current run")

    next_step = int(payload["next_step"])
    values = payload["parameters"]
    if not isinstance(values, list):
        raise RuntimeError("Checkpoint parameter payload is invalid")

    return CheckpointState(
        next_step=next_step,
        parameters=MockTensor(values),
        path="",
    )


def load_latest_checkpoint(
    checkpoint_dir: str,
    world_size: int,
    parameter_dim: int,
    shard_size: int,
    learning_rate: float,
) -> CheckpointState | None:
    path = Path(checkpoint_dir)
    if not path.exists():
        return None

    files = sorted(path.glob("checkpoint_step_*.json"), key=_parse_checkpoint_step)
    if not files:
        return None

    latest = files[-1]
    with latest.open("r", encoding="utf-8") as f:
        record = json.load(f)

    state = _validate_checkpoint_record(
        record=record,
        world_size=world_size,
        parameter_dim=parameter_dim,
        shard_size=shard_size,
        learning_rate=learning_rate,
    )
    return CheckpointState(
        next_step=state.next_step,
        parameters=state.parameters,
        path=str(latest),
    )


def _worker_process(
    rank: int,
    shard: tuple[MockTensor, ...],
    start_step: int,
    num_steps: int,
    fail_rank: int | None,
    fail_step: int | None,
    server_to_worker_queue: Any,
    worker_to_server_queue: Any,
    worker_report_queue: Any,
) -> None:
    gradient_history: list[MockTensor] = []
    parameter_history: list[MockTensor] = []

    for step in range(start_step, num_steps):
        forward_msg = server_to_worker_queue.get()
        if not isinstance(forward_msg, ParameterMessage) or forward_msg.phase != "forward":
            raise RuntimeError(f"Worker {rank} expected forward parameters at step {step}")
        if forward_msg.step != step:
            raise RuntimeError(
                f"Worker {rank} received forward step {forward_msg.step}, expected {step}"
            )

        if fail_rank is not None and fail_step is not None and rank == fail_rank and step == fail_step:
            raise RuntimeError(
                f"Injected worker failure at rank={rank}, step={step} for recovery test"
            )

        gradient = _compute_local_gradient(forward_msg.parameters, shard)
        gradient_history.append(gradient)
        worker_to_server_queue.put(
            GradientMessage(worker_rank=rank, step=step, gradient=gradient)
        )

        updated_msg = server_to_worker_queue.get()
        if not isinstance(updated_msg, ParameterMessage) or updated_msg.phase != "updated":
            raise RuntimeError(f"Worker {rank} expected updated parameters at step {step}")
        if updated_msg.step != step:
            raise RuntimeError(
                f"Worker {rank} received updated step {updated_msg.step}, expected {step}"
            )
        parameter_history.append(updated_msg.parameters)

    worker_report_queue.put(
        WorkerReport(
            rank=rank,
            gradient_history=tuple(gradient_history),
            parameter_history=tuple(parameter_history),
        )
    )


def _server_process(
    world_size: int,
    start_step: int,
    num_steps: int,
    learning_rate: float,
    parameter_dim: int,
    shard_size: int,
    initial_parameters: MockTensor,
    checkpoint_dir: str | None,
    checkpoint_interval_steps: int,
    server_to_worker_queues: list[Any],
    worker_to_server_queue: Any,
    server_report_queue: Any,
) -> None:
    parameters = initial_parameters
    parameter_history: list[MockTensor] = [parameters]
    written_checkpoints: list[str] = []

    if checkpoint_dir is not None and start_step == 0:
        written_checkpoints.append(
            _write_checkpoint(
                checkpoint_dir=checkpoint_dir,
                next_step=0,
                parameters=parameters,
                world_size=world_size,
                parameter_dim=parameter_dim,
                shard_size=shard_size,
                learning_rate=learning_rate,
            )
        )

    for step in range(start_step, num_steps):
        for queue in server_to_worker_queues:
            queue.put(ParameterMessage(step=step, parameters=parameters, phase="forward"))

        messages: list[GradientMessage] = []
        for _ in range(world_size):
            msg = worker_to_server_queue.get()
            if not isinstance(msg, GradientMessage):
                raise RuntimeError("Server received non-gradient message")
            messages.append(msg)

        _validate_gradient_batch(messages, expected_step=step, world_size=world_size)
        avg_gradient = parameter_server_average([msg.gradient for msg in messages])
        parameters = parameters + avg_gradient.scale(-learning_rate)
        parameter_history.append(parameters)

        for queue in server_to_worker_queues:
            queue.put(ParameterMessage(step=step, parameters=parameters, phase="updated"))

        completed_step = step + 1
        should_checkpoint = (
            checkpoint_dir is not None
            and checkpoint_interval_steps > 0
            and completed_step % checkpoint_interval_steps == 0
        ) or (checkpoint_dir is not None and completed_step == num_steps)

        if should_checkpoint and checkpoint_dir is not None:
            written_checkpoints.append(
                _write_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    next_step=completed_step,
                    parameters=parameters,
                    world_size=world_size,
                    parameter_dim=parameter_dim,
                    shard_size=shard_size,
                    learning_rate=learning_rate,
                )
            )

    server_report_queue.put(
        ServerReport(
            parameter_history=tuple(parameter_history),
            written_checkpoints=tuple(written_checkpoints),
        )
    )


def _terminate_processes(processes: list[Any]) -> None:
    for proc in processes:
        if proc.is_alive():
            proc.terminate()
    for proc in processes:
        proc.join(timeout=2.0)


def _run_single_attempt(
    world_size: int,
    start_step: int,
    num_steps: int,
    learning_rate: float,
    parameter_dim: int,
    shard_size: int,
    initial_parameters: MockTensor,
    checkpoint_dir: str | None,
    checkpoint_interval_steps: int,
    fail_rank: int | None,
    fail_step: int | None,
    join_timeout_sec: float,
) -> tuple[ServerReport, tuple[WorkerReport, ...]]:
    shards = build_synthetic_shards(world_size, shard_size, parameter_dim)

    ctx = mp.get_context("spawn")
    worker_to_server_queue = ctx.Queue()
    worker_report_queue = ctx.Queue()
    server_report_queue = ctx.Queue()
    server_to_worker_queues = [ctx.Queue() for _ in range(world_size)]

    server_proc = ctx.Process(
        target=_server_process,
        args=(
            world_size,
            start_step,
            num_steps,
            learning_rate,
            parameter_dim,
            shard_size,
            initial_parameters,
            checkpoint_dir,
            checkpoint_interval_steps,
            server_to_worker_queues,
            worker_to_server_queue,
            server_report_queue,
        ),
    )

    worker_procs: list[Any] = []
    for rank in range(world_size):
        proc = ctx.Process(
            target=_worker_process,
            args=(
                rank,
                shards[rank],
                start_step,
                num_steps,
                fail_rank,
                fail_step,
                server_to_worker_queues[rank],
                worker_to_server_queue,
                worker_report_queue,
            ),
        )
        worker_procs.append(proc)

    server_proc.start()
    for proc in worker_procs:
        proc.start()

    failure_reason: str | None = None
    for proc in worker_procs:
        proc.join(timeout=join_timeout_sec)
        if proc.exitcode is None:
            failure_reason = f"Worker process {proc.pid} timed out"
            break
        if proc.exitcode != 0:
            failure_reason = f"Worker process {proc.pid} exited with code {proc.exitcode}"
            break

    if failure_reason is None:
        server_proc.join(timeout=join_timeout_sec)
        if server_proc.exitcode is None:
            failure_reason = "Parameter server process timed out"
        elif server_proc.exitcode != 0:
            failure_reason = f"Parameter server process {server_proc.pid} exited with code {server_proc.exitcode}"

    if failure_reason is not None:
        _terminate_processes([server_proc, *worker_procs])
        raise RuntimeError(failure_reason)

    try:
        server_report = server_report_queue.get(timeout=2.0)
    except Empty as exc:
        raise RuntimeError("Did not receive server report") from exc

    worker_reports: list[WorkerReport] = []
    for _ in range(world_size):
        try:
            report = worker_report_queue.get(timeout=2.0)
        except Empty as exc:
            raise RuntimeError("Did not receive all worker reports") from exc
        worker_reports.append(report)

    worker_reports.sort(key=lambda item: item.rank)
    return server_report, tuple(worker_reports)


def run_synchronous_parameter_server_training(
    world_size: int,
    num_steps: int,
    learning_rate: float,
    parameter_dim: int,
    shard_size: int,
    join_timeout_sec: float = 30.0,
    checkpoint_dir: str | None = None,
    checkpoint_interval_steps: int = 0,
    max_restarts: int = 0,
    inject_failure_rank: int | None = None,
    inject_failure_step: int | None = None,
) -> SyncPSTrainingResult:
    if world_size <= 0:
        raise ValueError("world_size must be >= 1")
    if num_steps <= 0:
        raise ValueError("num_steps must be >= 1")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if parameter_dim <= 0:
        raise ValueError("parameter_dim must be >= 1")
    if checkpoint_interval_steps < 0:
        raise ValueError("checkpoint_interval_steps must be >= 0")
    if max_restarts < 0:
        raise ValueError("max_restarts must be >= 0")

    if max_restarts > 0 and checkpoint_interval_steps <= 0:
        raise ValueError("Recovery requires checkpoint_interval_steps > 0")

    active_checkpoint_dir = checkpoint_dir
    if active_checkpoint_dir is None and (checkpoint_interval_steps > 0 or max_restarts > 0):
        active_checkpoint_dir = tempfile.mkdtemp(prefix="dmlf_ps_checkpoints_")

    start_step = 0
    initial_parameters = MockTensor([0.0] * parameter_dim)
    restart_count = 0
    last_checkpoint_path: str | None = None

    while True:
        fail_rank = inject_failure_rank if restart_count == 0 else None
        fail_step = inject_failure_step if restart_count == 0 else None

        try:
            server_report, worker_reports = _run_single_attempt(
                world_size=world_size,
                start_step=start_step,
                num_steps=num_steps,
                learning_rate=learning_rate,
                parameter_dim=parameter_dim,
                shard_size=shard_size,
                initial_parameters=initial_parameters,
                checkpoint_dir=active_checkpoint_dir,
                checkpoint_interval_steps=checkpoint_interval_steps,
                fail_rank=fail_rank,
                fail_step=fail_step,
                join_timeout_sec=join_timeout_sec,
            )
            if server_report.written_checkpoints:
                last_checkpoint_path = server_report.written_checkpoints[-1]
            final_parameters = server_report.parameter_history[-1]
            return SyncPSTrainingResult(
                final_parameters=final_parameters,
                server_parameter_history=server_report.parameter_history,
                worker_reports=worker_reports,
                restart_count=restart_count,
                last_checkpoint_path=last_checkpoint_path,
            )
        except RuntimeError as exc:
            if restart_count >= max_restarts:
                raise RuntimeError(f"Training failed without recoverable checkpoint: {exc}") from exc
            if active_checkpoint_dir is None:
                raise RuntimeError("Recovery requested but checkpointing is disabled") from exc

            state = load_latest_checkpoint(
                checkpoint_dir=active_checkpoint_dir,
                world_size=world_size,
                parameter_dim=parameter_dim,
                shard_size=shard_size,
                learning_rate=learning_rate,
            )
            if state is None:
                raise RuntimeError("Worker failure detected but no checkpoint available") from exc

            start_step = state.next_step
            initial_parameters = state.parameters
            last_checkpoint_path = state.path
            restart_count += 1


def serial_synchronous_reference(
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
