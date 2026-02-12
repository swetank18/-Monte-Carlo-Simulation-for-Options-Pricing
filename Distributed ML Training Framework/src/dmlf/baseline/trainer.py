from __future__ import annotations

import logging
import multiprocessing as mp
import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from dmlf.env import require_torch, require_torchvision

from .config import BaselineConfig
from .model import MnistMLP


@dataclass(frozen=True)
class WorkerResult:
    rank: int
    num_batches: int
    num_samples: int
    epoch_losses: tuple[float, ...]
    gradient_l2_norms: dict[str, float]


def _seed_worker(base_seed: int, torch: Any) -> Callable[[int], None]:
    def _init_fn(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init_fn


def _get_training_stack() -> tuple[Any, Any, Any, Any, Any]:
    torch = require_torch("running MNIST baseline training")
    torchvision = require_torchvision("loading MNIST for baseline training")
    nn = torch.nn
    DataLoader = torch.utils.data.DataLoader
    datasets = torchvision.datasets
    transforms = torchvision.transforms
    return torch, nn, DataLoader, datasets, transforms


def set_determinism(seed: int) -> None:
    torch = require_torch("setting deterministic training behavior")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _build_dataset(cfg: BaselineConfig) -> Any:
    _, _, _, datasets, transforms = _get_training_stack()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return datasets.MNIST(
        root=cfg.data_root,
        train=True,
        download=True,
        transform=transform,
    )


def _rank_shard_indices(dataset_size: int, rank: int, world_size: int) -> list[int]:
    if world_size <= 0:
        raise ValueError("world_size must be >= 1")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size - 1}]")
    return list(range(rank, dataset_size, world_size))


def build_train_loader(cfg: BaselineConfig) -> Any:
    torch, _, DataLoader, datasets, transforms = _get_training_stack()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset = datasets.MNIST(
        root=cfg.data_root,
        train=True,
        download=True,
        transform=transform,
    )

    generator = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda") and torch.cuda.is_available()),
        worker_init_fn=_seed_worker(cfg.seed, torch),
        generator=generator,
    )


def build_sharded_train_loader(cfg: BaselineConfig, rank: int, world_size: int) -> Any:
    torch, _, DataLoader, _, _ = _get_training_stack()
    dataset = _build_dataset(cfg)
    shard_indices = _rank_shard_indices(len(dataset), rank, world_size)
    subset = torch.utils.data.Subset(dataset, shard_indices)
    generator = torch.Generator().manual_seed(cfg.seed + rank)

    return DataLoader(
        subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda") and torch.cuda.is_available()),
        worker_init_fn=_seed_worker(cfg.seed + rank, torch),
        generator=generator,
    )


def train(cfg: BaselineConfig) -> list[float]:
    torch, nn, _, _, _ = _get_training_stack()
    set_determinism(cfg.seed)

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("mnist-baseline")

    device = torch.device(
        cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    loader = build_train_loader(cfg)

    model = MnistMLP(hidden_dim=cfg.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    epoch_losses: list[float] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        num_samples = 0

        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

        epoch_loss = running_loss / max(1, num_samples)
        epoch_losses.append(epoch_loss)
        logger.info("Epoch %d/%d - loss: %.6f", epoch, cfg.epochs, epoch_loss)

    return epoch_losses


def _compute_gradient_l2_norms(model: Any, torch: Any) -> dict[str, float]:
    norms: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        norms[name] = float(torch.norm(parameter.grad.detach(), p=2).item())
    return norms


def _worker_train(rank: int, world_size: int, cfg: BaselineConfig, out_queue: Any) -> None:
    torch, nn, _, _, _ = _get_training_stack()
    set_determinism(cfg.seed + rank)

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format=f"%(asctime)s | worker={rank} | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(f"mnist-worker-{rank}")

    device = torch.device(
        cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    loader = build_sharded_train_loader(cfg, rank=rank, world_size=world_size)
    model = MnistMLP(hidden_dim=cfg.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()

    epoch_losses: list[float] = []
    num_batches = 0
    num_samples = 0
    gradient_l2_norms: dict[str, float] = {}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_samples = 0

        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            model.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            epoch_samples += batch_size
            num_batches += 1
            num_samples += batch_size
            gradient_l2_norms = _compute_gradient_l2_norms(model, torch)

        epoch_loss = running_loss / max(1, epoch_samples)
        epoch_losses.append(epoch_loss)
        logger.info(
            "Epoch %d/%d - local_loss: %.6f - shard_samples: %d",
            epoch,
            cfg.epochs,
            epoch_loss,
            epoch_samples,
        )

    out_queue.put(
        WorkerResult(
            rank=rank,
            num_batches=num_batches,
            num_samples=num_samples,
            epoch_losses=tuple(epoch_losses),
            gradient_l2_norms=gradient_l2_norms,
        )
    )


def simulate_independent_workers(cfg: BaselineConfig, world_size: int) -> list[WorkerResult]:
    if world_size < 1:
        raise ValueError("world_size must be >= 1")

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = []

    for rank in range(world_size):
        proc = ctx.Process(target=_worker_train, args=(rank, world_size, cfg, queue))
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"Worker process {proc.pid} failed with exit code {proc.exitcode}")

    results: list[WorkerResult] = []
    for _ in range(world_size):
        results.append(queue.get())
    results.sort(key=lambda item: item.rank)

    logger = logging.getLogger("mnist-multiprocess")
    for result in results:
        logger.info(
            "Worker %d complete - batches=%d samples=%d final_epoch_loss=%.6f gradients=%d",
            result.rank,
            result.num_batches,
            result.num_samples,
            result.epoch_losses[-1] if result.epoch_losses else float("nan"),
            len(result.gradient_l2_norms),
        )

    return results
