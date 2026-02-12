from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MockTensor:
    """Minimal immutable tensor-like container for logic tests."""

    values: tuple[float, ...]

    def __init__(self, values: list[float] | tuple[float, ...]) -> None:
        object.__setattr__(self, "values", tuple(float(v) for v in values))

    def __len__(self) -> int:
        return len(self.values)

    def __add__(self, other: "MockTensor") -> "MockTensor":
        if len(self) != len(other):
            raise ValueError(
                f"Cannot add tensors with different lengths: {len(self)} vs {len(other)}"
            )
        return MockTensor(tuple(a + b for a, b in zip(self.values, other.values)))

    def scale(self, factor: float) -> "MockTensor":
        return MockTensor(tuple(v * factor for v in self.values))

    def split(self, chunks: int) -> list["MockTensor"]:
        if chunks <= 0:
            raise ValueError("chunks must be >= 1")
        if len(self) % chunks != 0:
            raise ValueError(
                f"Tensor length {len(self)} is not divisible by chunks={chunks}"
            )
        chunk_size = len(self) // chunks
        return [
            MockTensor(self.values[i * chunk_size : (i + 1) * chunk_size])
            for i in range(chunks)
        ]

    @staticmethod
    def concat(chunks: list["MockTensor"]) -> "MockTensor":
        values: list[float] = []
        for chunk in chunks:
            values.extend(chunk.values)
        return MockTensor(values)

    def almost_equal(self, other: "MockTensor", tol: float = 1e-9) -> bool:
        if len(self) != len(other):
            return False
        return all(abs(a - b) <= tol for a, b in zip(self.values, other.values))
