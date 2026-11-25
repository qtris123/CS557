#!/usr/bin/env python3
"""Summarize per-term reward magnitudes from a CSV export.

The script intentionally uses only the Python standard library so it can be
copied into an Isaac Lab project without adding dependencies.

Examples
--------

Auto-detect reward-like numeric columns::

    python scripts/analyze_rewards.py rewards.csv

Select columns explicitly::

    python scripts/analyze_rewards.py rewards.csv \
        --columns track_lin_vel track_ang_vel torque action_rate

Aggregate timestep rows into episodes::

    python scripts/analyze_rewards.py rewards.csv --episode-column episode_id

Apply weights to raw reward columns::

    python scripts/analyze_rewards.py rewards.csv \
        --weights-json '{"track_lin_vel": 2.0, "torque": -1e-5}' \
        --dt 0.02

Notes
-----

- If the CSV already contains weighted Isaac Lab reward contributions, do not
  pass ``--weights-json`` and normally leave ``--dt`` at 1.0.
- If the CSV contains raw reward functions, ``--weights-json`` and ``--dt`` can
  estimate weighted contributions.
- The script is a scale diagnostic, not an automatic reward tuner.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize reward-term scales and episodic contributions from CSV."
    )
    parser.add_argument("csv_file", type=Path, help="CSV file containing reward values")
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Reward columns to analyze. If omitted, reward-like numeric columns are detected.",
    )
    parser.add_argument(
        "--include-regex",
        default=r"(?i)(reward|rew)",
        help="Regex used for automatic column detection (default: %(default)s)",
    )
    parser.add_argument(
        "--episode-column",
        default=None,
        help="Optional column identifying episodes. Timestep values are summed per episode first.",
    )
    parser.add_argument(
        "--weights-json",
        default=None,
        help="JSON object mapping raw reward column names to weights.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Optional timestep multiplier applied with weights (default: %(default)s).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Maximum rows in the output, ranked by absolute total contribution.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Output format (default: %(default)s).",
    )
    return parser.parse_args()


def finite_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")
        rows = list(reader)
        return list(reader.fieldnames), rows


def is_numeric_column(rows: Iterable[dict[str, str]], column: str) -> bool:
    seen = False
    for row in rows:
        raw = row.get(column)
        if raw in (None, ""):
            continue
        if finite_float(raw) is None:
            return False
        seen = True
    return seen


def select_columns(
    fieldnames: list[str],
    rows: list[dict[str, str]],
    explicit: list[str] | None,
    include_regex: str,
    episode_column: str | None,
) -> list[str]:
    if explicit:
        missing = [column for column in explicit if column not in fieldnames]
        if missing:
            raise ValueError(f"Unknown columns: {', '.join(missing)}")
        columns = explicit
    else:
        pattern = re.compile(include_regex)
        columns = [
            column
            for column in fieldnames
            if column != episode_column
            and pattern.search(column)
            and is_numeric_column(rows, column)
        ]

    if not columns:
        raise ValueError(
            "No reward columns selected. Pass --columns explicitly or adjust --include-regex."
        )
    return columns


def parse_weights(raw: str | None) -> dict[str, float]:
    if raw is None:
        return {}
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise ValueError("--weights-json must decode to a JSON object")

    weights: dict[str, float] = {}
    for key, value in decoded.items():
        if not isinstance(key, str):
            raise ValueError("Weight keys must be strings")
        try:
            weights[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Weight for {key!r} is not numeric") from exc
    return weights


def transform_rows(
    rows: list[dict[str, str]],
    columns: list[str],
    weights: dict[str, float],
    dt: float,
) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {column: [] for column in columns}
    for row in rows:
        for column in columns:
            number = finite_float(row.get(column))
            if number is None:
                continue
            if column in weights:
                number *= weights[column] * dt
            values[column].append(number)
    return values


def aggregate_episodes(
    rows: list[dict[str, str]],
    columns: list[str],
    episode_column: str,
    weights: dict[str, float],
    dt: float,
) -> dict[str, list[float]]:
    episode_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row_number, row in enumerate(rows, start=2):
        episode = row.get(episode_column)
        if episode in (None, ""):
            raise ValueError(
                f"Missing episode id in column {episode_column!r} at CSV row {row_number}"
            )
        for column in columns:
            number = finite_float(row.get(column))
            if number is None:
                continue
            if column in weights:
                number *= weights[column] * dt
            episode_sums[episode][column] += number

    result: dict[str, list[float]] = {column: [] for column in columns}
    for sums in episode_sums.values():
        for column in columns:
            result[column].append(sums.get(column, 0.0))
    return result


def summarize(values_by_term: dict[str, list[float]]) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    totals = {term: sum(values) for term, values in values_by_term.items()}
    total_positive = sum(value for value in totals.values() if value > 0)
    total_negative_abs = abs(sum(value for value in totals.values() if value < 0))

    for term, values in values_by_term.items():
        if not values:
            continue
        total = totals[term]
        if total > 0 and total_positive > 0:
            budget_share = total / total_positive
        elif total < 0 and total_negative_abs > 0:
            budget_share = abs(total) / total_negative_abs
        else:
            budget_share = 0.0

        rows.append(
            {
                "term": term,
                "n": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "p95_abs": percentile([abs(value) for value in values], 0.95),
                "max_abs": max(abs(value) for value in values),
                "total": total,
                "budget_share": budget_share,
            }
        )

    rows.sort(key=lambda row: abs(float(row["total"])), reverse=True)
    return rows


def fmt_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value == 0:
        return "0"
    if abs_value >= 1e4 or abs_value < 1e-3:
        return f"{value:.3e}"
    return f"{value:.4f}"


def print_markdown(summary: list[dict[str, float | str | int]], top: int) -> None:
    print("| term | n | mean | std | p95 abs | max abs | total | budget share |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary[:top]:
        print(
            "| {term} | {n} | {mean} | {std} | {p95_abs} | {max_abs} | {total} | {share:.1%} |".format(
                term=row["term"],
                n=row["n"],
                mean=fmt_number(float(row["mean"])),
                std=fmt_number(float(row["std"])),
                p95_abs=fmt_number(float(row["p95_abs"])),
                max_abs=fmt_number(float(row["max_abs"])),
                total=fmt_number(float(row["total"])),
                share=float(row["budget_share"]),
            )
        )


def print_csv(summary: list[dict[str, float | str | int]], top: int) -> None:
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=("term", "n", "mean", "std", "p95_abs", "max_abs", "total", "budget_share"),
    )
    writer.writeheader()
    for row in summary[:top]:
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    try:
        if args.dt <= 0:
            raise ValueError("--dt must be positive")
        fieldnames, rows = load_rows(args.csv_file)
        if not rows:
            raise ValueError("CSV contains no data rows")

        if args.episode_column and args.episode_column not in fieldnames:
            raise ValueError(f"Unknown episode column: {args.episode_column}")

        columns = select_columns(
            fieldnames,
            rows,
            args.columns,
            args.include_regex,
            args.episode_column,
        )
        weights = parse_weights(args.weights_json)
        unknown_weights = sorted(set(weights) - set(columns))
        if unknown_weights:
            raise ValueError(
                "Weights provided for columns that are not selected: " + ", ".join(unknown_weights)
            )

        if args.episode_column:
            values = aggregate_episodes(
                rows, columns, args.episode_column, weights, args.dt
            )
        else:
            values = transform_rows(rows, columns, weights, args.dt)

        summary = summarize(values)
        if args.format == "markdown":
            print_markdown(summary, args.top)
        else:
            print_csv(summary, args.top)
        return 0
    except (OSError, ValueError, json.JSONDecodeError, re.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
