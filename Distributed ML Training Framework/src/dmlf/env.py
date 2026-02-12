"""Execution environment dependency guards."""

from __future__ import annotations

import importlib
from types import ModuleType


class DependencyError(RuntimeError):
    """Raised when a required optional dependency is unavailable."""


def require_module(module_name: str, purpose: str, install_hint: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        message = (
            f"Missing required dependency '{module_name}' for {purpose}. "
            f"Install it with: {install_hint}. "
            f"Original import error: {exc}"
        )
        raise DependencyError(message) from exc


def require_torch(purpose: str) -> ModuleType:
    return require_module("torch", purpose=purpose, install_hint="pip install torch")


def require_torchvision(purpose: str) -> ModuleType:
    return require_module(
        "torchvision",
        purpose=purpose,
        install_hint="pip install torchvision",
    )
