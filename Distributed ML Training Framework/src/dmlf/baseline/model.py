from __future__ import annotations

from typing import Any

from dmlf.env import require_torch


def _get_torch_stack() -> tuple[Any, Any]:
    torch = require_torch("constructing the MNIST baseline model")
    nn = torch.nn
    return torch, nn


class MnistMLP:  # pragma: no cover - thin wrapper over torch module creation
    """Simple MLP baseline for 28x28 MNIST digits."""

    def __new__(cls, hidden_dim: int = 256) -> Any:
        torch, nn = _get_torch_stack()

        class _MnistMLP(nn.Module):
            def __init__(self, hidden_dim_value: int) -> None:
                super().__init__()
                self.network = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(28 * 28, hidden_dim_value),
                    nn.ReLU(),
                    nn.Linear(hidden_dim_value, hidden_dim_value),
                    nn.ReLU(),
                    nn.Linear(hidden_dim_value, 10),
                )

            def forward(self, x: Any) -> Any:
                return self.network(x)

        _ = torch  # explicit to show torch dependency is intentional here
        return _MnistMLP(hidden_dim)
