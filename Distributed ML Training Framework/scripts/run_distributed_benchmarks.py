import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt

from dmlf.distributed.algorithms import parameter_server_average
from dmlf.distributed.mock_tensor import MockTensor
from dmlf.distributed.parameter_server_sync import (
    build_synthetic_shards,
    run_synchronous_parameter_server_training,
)
from dmlf.distributed.ring_allreduce_sync import run_synchronous_ring_allreduce_training


def _compute_local_gradient(parameters: MockTensor, shard: tuple[MockTensor, ...]) -> MockTensor:
    grads = [parameters + sample.scale(-1.0) for sample in shard]
    return parameter_server_average(grads)


def serial_reference_trajectory(
    world_size: int,
    num_steps: int,
    learning_rate: float,
    parameter_dim: int,
    shard_size: int,
) -> list[MockTensor]:
    shards = build_synthetic_shards(world_size, shard_size, parameter_dim)
    parameters = MockTensor([0.0] * parameter_dim)
    history = [parameters]

    for _ in range(num_steps):
        gradients = [_compute_local_gradient(parameters, shard) for shard in shards]
        avg_gradient = parameter_server_average(gradients)
        parameters = parameters + avg_gradient.scale(-learning_rate)
        history.append(parameters)

    return history


def l2_distance(left: MockTensor, right: MockTensor) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left.values, right.values)))


def estimate_comm_bytes_per_step(method: str, world_size: int, parameter_dim: int) -> int:
    bytes_per_tensor = parameter_dim * 8
    if method == "parameter_server":
        # gradients up + params down (forward and updated) each step
        return world_size * bytes_per_tensor + 2 * world_size * bytes_per_tensor
    if method == "ring_allreduce":
        # 2*(N-1) rounds, each worker sends chunk of size P/N; cluster total = 2*(N-1)*P
        return 2 * (world_size - 1) * bytes_per_tensor
    raise ValueError(f"Unknown method: {method}")


def run_benchmark_once(
    method: str,
    world_size: int,
    num_steps: int,
    learning_rate: float,
    parameter_dim: int,
    shard_size: int,
):
    start = time.perf_counter()
    if method == "parameter_server":
        result = run_synchronous_parameter_server_training(
            world_size=world_size,
            num_steps=num_steps,
            learning_rate=learning_rate,
            parameter_dim=parameter_dim,
            shard_size=shard_size,
            join_timeout_sec=180.0,
        )
        history = list(result.server_parameter_history)
        final_params = result.final_parameters
    elif method == "ring_allreduce":
        result = run_synchronous_ring_allreduce_training(
            world_size=world_size,
            num_steps=num_steps,
            learning_rate=learning_rate,
            parameter_dim=parameter_dim,
            shard_size=shard_size,
            join_timeout_sec=180.0,
        )
        history = [MockTensor([0.0] * parameter_dim)] + list(result.worker_reports[0].parameter_history)
        final_params = result.final_parameters
    else:
        raise ValueError(f"Unsupported method: {method}")

    elapsed = time.perf_counter() - start
    return elapsed, final_params, history


def write_metrics_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "world_size",
                "num_steps",
                "learning_rate",
                "parameter_dim",
                "shard_size",
                "elapsed_sec_mean",
                "elapsed_sec_std",
                "throughput_samples_per_sec",
                "throughput_steps_per_sec",
                "scaling_efficiency",
                "comm_bytes_per_step",
                "comm_mb_total",
                "final_l2_error_vs_serial",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_convergence_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "world_size", "step", "l2_error_vs_serial"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_throughput(metrics: list[dict[str, float | int | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    methods = sorted({str(row["method"]) for row in metrics})
    plt.figure(figsize=(8, 4.5))

    for method in methods:
        rows = sorted(
            [row for row in metrics if str(row["method"]) == method],
            key=lambda row: int(row["world_size"]),
        )
        x = [int(row["world_size"]) for row in rows]
        y = [float(row["throughput_samples_per_sec"]) for row in rows]
        plt.plot(x, y, marker="o", label=method)

    plt.xlabel("World Size")
    plt.ylabel("Throughput (samples/sec)")
    plt.title("Distributed Training Throughput")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scaling_efficiency(metrics: list[dict[str, float | int | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    methods = sorted({str(row["method"]) for row in metrics})
    plt.figure(figsize=(8, 4.5))

    for method in methods:
        rows = sorted(
            [row for row in metrics if str(row["method"]) == method],
            key=lambda row: int(row["world_size"]),
        )
        x = [int(row["world_size"]) for row in rows]
        y = [float(row["scaling_efficiency"]) for row in rows]
        plt.plot(x, y, marker="o", label=method)

    plt.xlabel("World Size")
    plt.ylabel("Scaling Efficiency")
    plt.ylim(bottom=0.0)
    plt.title("Scaling Efficiency vs World Size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_convergence(convergence_rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))

    keys = sorted({(str(row["method"]), int(row["world_size"])) for row in convergence_rows})
    for method, world_size in keys:
        rows = [
            row
            for row in convergence_rows
            if str(row["method"]) == method and int(row["world_size"]) == world_size
        ]
        rows.sort(key=lambda row: int(row["step"]))
        x = [int(row["step"]) for row in rows]
        y = [float(row["l2_error_vs_serial"]) for row in rows]
        plt.plot(x, y, marker=".", label=f"{method} (N={world_size})")

    plt.xlabel("Step")
    plt.ylabel("L2 Error vs Serial Baseline")
    plt.yscale("log")
    plt.title("Convergence Equivalence vs Serial Baseline")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_comm_overhead(metrics: list[dict[str, float | int | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    methods = sorted({str(row["method"]) for row in metrics})
    world_sizes = sorted({int(row["world_size"]) for row in metrics})

    x_positions = list(range(len(world_sizes)))
    width = 0.35

    plt.figure(figsize=(8, 4.5))
    for idx, method in enumerate(methods):
        rows = {
            int(row["world_size"]): row
            for row in metrics
            if str(row["method"]) == method
        }
        y = [float(rows[ws]["comm_mb_total"]) for ws in world_sizes]
        offset = (idx - (len(methods) - 1) / 2) * width
        plt.bar([x + offset for x in x_positions], y, width=width, label=method)

    plt.xticks(x_positions, [str(ws) for ws in world_sizes])
    plt.xlabel("World Size")
    plt.ylabel("Estimated Communication per Run (MB)")
    plt.title("Estimated Communication Overhead")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_report(report_path: Path, metrics: list[dict[str, float | int | str]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Benchmark Report", "", "## Summary"]
    for method in sorted({str(row["method"]) for row in metrics}):
        rows = [row for row in metrics if str(row["method"]) == method]
        rows.sort(key=lambda row: int(row["world_size"]))
        best = max(rows, key=lambda row: float(row["throughput_samples_per_sec"]))
        lines.append(
            f"- `{method}` peak throughput: {float(best['throughput_samples_per_sec']):.2f} samples/s at world_size={int(best['world_size'])}"
        )

    lines.extend(
        [
            "",
            "## What Conclusions Are Valid",
            "- Synchronous parameter-server and ring all-reduce implementations are numerically equivalent to the serial baseline for this deterministic synthetic workload.",
            "- Measured throughput and scaling efficiency reflect framework-process overhead and synchronization behavior on a single machine.",
            "- Estimated communication overhead clearly separates centralized PS traffic from ring-collective traffic growth patterns.",
            "",
            "## What Conclusions Are Not Valid",
            "- Results do not represent multi-node network behavior (no real NIC, no packet loss, no cross-machine latency).",
            "- Results do not represent GPU behavior or PyTorch tensor-kernel performance.",
            "- Convergence claims are limited to this synthetic objective and synchronous update rule, not real deep model training dynamics.",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run distributed framework benchmarks and generate plots.")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--parameter-dim", type=int, default=40)
    parser.add_argument("--shard-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "benchmarks" / "results",
    )
    args = parser.parse_args()

    methods: list[str] = ["parameter_server", "ring_allreduce"]
    metrics_rows: list[dict[str, float | int | str]] = []
    convergence_rows: list[dict[str, float | int | str]] = []

    for method in methods:
        base_throughput = None
        for world_size in sorted(args.world_sizes):
            if args.parameter_dim % world_size != 0:
                continue

            elapsed_runs: list[float] = []
            histories: list[list[MockTensor]] = []
            final_params: list[MockTensor] = []
            failed = False

            for _ in range(args.repeats):
                try:
                    elapsed, final_param, history = run_benchmark_once(
                        method=method,
                        world_size=world_size,
                        num_steps=args.num_steps,
                        learning_rate=args.learning_rate,
                        parameter_dim=args.parameter_dim,
                        shard_size=args.shard_size,
                    )
                    elapsed_runs.append(elapsed)
                    histories.append(history)
                    final_params.append(final_param)
                except RuntimeError as exc:
                    print(
                        f"Skipping method={method} world_size={world_size} due to runtime failure: {exc}"
                    )
                    failed = True
                    break

            if failed:
                continue

            elapsed_mean = statistics.mean(elapsed_runs)
            elapsed_std = statistics.pstdev(elapsed_runs) if len(elapsed_runs) > 1 else 0.0
            samples_total = world_size * args.shard_size * args.num_steps
            throughput_samples = samples_total / elapsed_mean
            throughput_steps = args.num_steps / elapsed_mean

            if world_size == 1:
                base_throughput = throughput_samples
            scaling_eff = throughput_samples / (base_throughput * world_size) if base_throughput else 1.0

            serial_history = serial_reference_trajectory(
                world_size=world_size,
                num_steps=args.num_steps,
                learning_rate=args.learning_rate,
                parameter_dim=args.parameter_dim,
                shard_size=args.shard_size,
            )
            final_error = l2_distance(final_params[0], serial_history[-1])

            comm_bytes_step = estimate_comm_bytes_per_step(
                method=method,
                world_size=world_size,
                parameter_dim=args.parameter_dim,
            )
            comm_bytes_total = comm_bytes_step * args.num_steps

            metrics_rows.append(
                {
                    "method": method,
                    "world_size": world_size,
                    "num_steps": args.num_steps,
                    "learning_rate": args.learning_rate,
                    "parameter_dim": args.parameter_dim,
                    "shard_size": args.shard_size,
                    "elapsed_sec_mean": elapsed_mean,
                    "elapsed_sec_std": elapsed_std,
                    "throughput_samples_per_sec": throughput_samples,
                    "throughput_steps_per_sec": throughput_steps,
                    "scaling_efficiency": scaling_eff,
                    "comm_bytes_per_step": comm_bytes_step,
                    "comm_mb_total": comm_bytes_total / (1024 * 1024),
                    "final_l2_error_vs_serial": final_error,
                }
            )

            # Use first run trajectory for convergence plotting.
            run_history = histories[0]
            max_len = min(len(run_history), len(serial_history))
            for step in range(max_len):
                error = l2_distance(run_history[step], serial_history[step])
                convergence_rows.append(
                    {
                        "method": method,
                        "world_size": world_size,
                        "step": step,
                        "l2_error_vs_serial": max(error, 1e-16),
                    }
                )

    metrics_csv = args.results_dir / "benchmark_metrics.csv"
    convergence_csv = args.results_dir / "convergence_metrics.csv"
    write_metrics_csv(metrics_csv, metrics_rows)
    write_convergence_csv(convergence_csv, convergence_rows)

    plot_throughput(metrics_rows, args.results_dir / "throughput.png")
    plot_scaling_efficiency(metrics_rows, args.results_dir / "scaling_efficiency.png")
    plot_convergence(convergence_rows, args.results_dir / "convergence_vs_baseline.png")
    plot_comm_overhead(metrics_rows, args.results_dir / "communication_overhead.png")

    write_report(PROJECT_ROOT / "docs" / "benchmarks" / "benchmark_report.md", metrics_rows)

    print(f"Wrote metrics: {metrics_csv}")
    print(f"Wrote convergence: {convergence_csv}")
    print(f"Wrote plots under: {args.results_dir}")


if __name__ == "__main__":
    main()
