"""Single-node correctness baseline for MNIST training."""

from .config import BaselineConfig, load_config


def train(cfg: BaselineConfig) -> list[float]:
    from .trainer import train as _train

    return _train(cfg)


def simulate_independent_workers(
    cfg: BaselineConfig, world_size: int
):
    from .trainer import simulate_independent_workers as _simulate

    return _simulate(cfg, world_size)


__all__ = ["BaselineConfig", "load_config", "train", "simulate_independent_workers"]
