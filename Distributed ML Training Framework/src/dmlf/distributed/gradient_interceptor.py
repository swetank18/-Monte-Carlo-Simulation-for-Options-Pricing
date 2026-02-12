from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from dmlf.env import require_torch


@dataclass(frozen=True)
class ParameterGradient:
    """Captured gradient payload for a single parameter."""

    name: str
    gradient: Any
    shape: tuple[int, ...]
    dtype: str
    device: str


@dataclass(frozen=True)
class GradientSnapshot:
    """One backward pass worth of gradients in model parameter order."""

    backward_index: int
    gradients: OrderedDict[str, ParameterGradient | None]
    missing_parameters: tuple[str, ...]


class GradientInterceptor:
    """Intercepts and stores parameter gradients via autograd hooks."""

    def __init__(self, clone: bool = True, move_to_cpu: bool = False) -> None:
        self._clone = clone
        self._move_to_cpu = move_to_cpu
        self._registered_names: list[str] = []
        self._handles: list[Any] = []
        self._current: dict[str, ParameterGradient] = {}
        self._history: list[GradientSnapshot] = []
        self._backward_index = 0

    def attach(self, model: Any) -> None:
        """Register hooks on all trainable model parameters."""
        require_torch("attaching gradient autograd hooks")
        self.detach()

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            self._registered_names.append(name)
            self._handles.append(parameter.register_hook(self._make_hook(name)))

    def _make_hook(self, parameter_name: str):
        def _hook(grad: Any) -> Any:
            if grad is None:
                return grad

            captured = grad.detach()
            if self._clone:
                captured = captured.clone()
            if self._move_to_cpu:
                captured = captured.cpu()

            self._current[parameter_name] = ParameterGradient(
                name=parameter_name,
                gradient=captured,
                shape=tuple(captured.shape),
                dtype=str(captured.dtype),
                device=str(captured.device),
            )
            return grad

        return _hook

    def start_backward_capture(self) -> None:
        """Call immediately before backward() to start a fresh capture window."""
        self._current = {}
        self._backward_index += 1

    def finish_backward_capture(self) -> GradientSnapshot:
        """Call immediately after backward() to persist the captured gradients."""
        gradients = OrderedDict()
        missing: list[str] = []

        for name in self._registered_names:
            item = self._current.get(name)
            gradients[name] = item
            if item is None:
                missing.append(name)

        snapshot = GradientSnapshot(
            backward_index=self._backward_index,
            gradients=gradients,
            missing_parameters=tuple(missing),
        )
        self._history.append(snapshot)
        return snapshot

    def last_snapshot(self) -> GradientSnapshot | None:
        return self._history[-1] if self._history else None

    def history(self) -> list[GradientSnapshot]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history = []

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._registered_names = []
        self._current = {}

    def __enter__(self) -> "GradientInterceptor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.detach()
