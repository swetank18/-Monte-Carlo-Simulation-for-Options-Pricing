import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dmlf.baseline import (
    BaselineConfig,
    load_config,
    simulate_independent_workers,
    train,
)


def parse_args() -> BaselineConfig:
    parser = argparse.ArgumentParser(description="Train deterministic MNIST MLP baseline")
    parser.add_argument("--config", type=str, default="configs/defaults/mnist_baseline.json")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--data-root", type=str)
    parser.add_argument("--log-level", type=str)
    parser.add_argument("--simulate-workers", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)

    overrides = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "num_workers": args.num_workers,
        "device": args.device,
        "data_root": args.data_root,
        "log_level": args.log_level,
        "simulate_workers": args.simulate_workers,
    }
    merged = {**cfg.__dict__, **{k: v for k, v in overrides.items() if v is not None}}
    return BaselineConfig(**merged)


if __name__ == "__main__":
    cfg = parse_args()
    if cfg.simulate_workers > 1:
        simulate_independent_workers(cfg, world_size=cfg.simulate_workers)
    else:
        train(cfg)
