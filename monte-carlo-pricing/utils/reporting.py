"""Reporting utilities for saving Monte Carlo results."""
from __future__ import annotations

import csv
from typing import Iterable, Mapping


def write_csv(path: str, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    base_fields = ["Model", "Price", "StdError", "CI_Low", "CI_High", "Notes"]
    extra_fields = sorted({k for row in rows for k in row.keys() if k not in base_fields})
    fieldnames = base_fields + extra_fields
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_greeks_csv(path: str, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = ["Method", "Delta", "Gamma", "StdError", "Notes"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
