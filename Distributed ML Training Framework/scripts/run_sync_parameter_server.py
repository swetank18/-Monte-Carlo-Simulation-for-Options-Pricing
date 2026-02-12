import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dmlf.distributed.parameter_server_sync import run_synchronous_parameter_server_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synchronous parameter-server simulation")
    parser.add_argument("--world-size", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--parameter-dim", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_synchronous_parameter_server_training(
        world_size=args.world_size,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        parameter_dim=args.parameter_dim,
        shard_size=args.shard_size,
    )
    print("Final parameters:", result.final_parameters.values)
    for worker in result.worker_reports:
        print(
            f"Worker {worker.rank}: gradients={len(worker.gradient_history)} updates={len(worker.parameter_history)}"
        )
