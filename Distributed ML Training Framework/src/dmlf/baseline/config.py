import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BaselineConfig:
    seed: int = 1234
    batch_size: int = 64
    test_batch_size: int = 1000
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    hidden_dim: int = 256
    num_workers: int = 0
    device: str = "cpu"
    data_root: str = "./data"
    log_level: str = "INFO"
    simulate_workers: int = 1

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "BaselineConfig":
        valid_keys = set(asdict(cls()).keys())
        filtered = {k: v for k, v in values.items() if k in valid_keys}
        return cls(**filtered)


def load_config(config_path: str | None = None) -> BaselineConfig:
    if config_path is None:
        return BaselineConfig()

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")

    return BaselineConfig.from_dict(data)
