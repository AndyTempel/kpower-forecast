"""Reviewable scoring reports with explicit evidence limits and no full paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, cast

from pydantic import JsonValue

from ._backtest import BacktestResult
from ._report import summarize


def _table(records: list[dict[str, object]]) -> str:
    if not records:
        return "No eligible evidence.\n"
    columns = list(records[0])

    def cell(value: object) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.5g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(record.get(column)) for column in columns) + " |"
        for record in records
    )
    return "\n".join(lines) + "\n"


def write_report(
    result: BacktestResult,
    directory: Path,
    *,
    hard_gates: Mapping[str, bool] | None = None,
    scope: str = "development",
) -> dict[str, JsonValue]:
    """Create a new report directory; never overwrite or serialize minute paths.

    Args:
        result: Per-origin aggregate scoring tables.
        directory: New output directory; a failed partial directory is preserved.
        hard_gates: Separately verified cold/Pi evidence, absent by default.
        scope: Development or published-live evidence, stated prominently.

    Returns:
        Strict JSON metrics also written alongside Markdown and aggregate Parquet.

    Raises:
        FileExistsError: If output already exists.
        ValueError: For invalid scores or unsupported report scope.
    """
    if scope not in ("development", "published_live"):
        raise ValueError("unknown report scope")
    metrics = summarize(
        result.tables, n_paths=result.config.n_paths, hard_gates=hard_gates
    )
    metrics.update(
        {
            "scope": scope,
            "schedule": result.schedule,
            "horizon_h": result.horizon_h,
            "config": cast(JsonValue, result.config.model_dump(mode="json")),
        }
    )
    directory.mkdir(parents=True, exist_ok=False)
    for name, table in result.tables.items():
        frame = table.copy()
        for column in frame:
            if any(isinstance(value, dict | list | tuple) for value in frame[column]):
                frame[column] = frame[column].map(
                    lambda value: (
                        json.dumps(value, sort_keys=True, default=str)
                        if isinstance(value, dict | list | tuple)
                        else value
                    )
                )
        frame.to_parquet(directory / f"{name}.parquet", index=False)
    with (directory / "metrics.json").open("x") as stream:
        json.dump(metrics, stream, indent=2, allow_nan=False)
    lines = [
        "# HVAC forecast evaluation",
        "",
        f"Scope: **{scope}**. Schedule: **{result.schedule}**. "
        f"Horizon: {result.horizon_h} h.",
        "",
    ]
    if scope == "development":
        lines += [
            "All historical data is development data already seen during development. "
            "These scores detect bugs and regressions; they do not prove unseen "
            "forecasting skill. Weather approximations and source provenance are "
            "recorded per origin.",
            "",
            "B3 settlement is the count-checked development forecast proxy, not a "
            "historical MPC replay. Unknown quality, incomplete blocks and missing "
            "baseline pairs stay unscorable. Partial-energy diagnostics compare "
            "only observed minutes and cannot satisfy A4.",
            "",
        ]
    else:
        lines += [
            "Scores use archived published forecasts and original planner reserves. "
            "No model was rerun. Missing original reserve inputs, compressor-state "
            "evidence or exact legacy band flags stay unscorable.",
            "",
        ]
    gates = cast(dict[str, dict[str, object]], metrics["gates"])
    lines += [
        "## Registered gates",
        "",
        _table(
            [
                {
                    "gate": key,
                    "status": value["status"],
                    "hard": value.get("hard", False),
                }
                for key, value in gates.items()
            ]
        ),
        "A6 and A7 require independent evidence. Release remains the maintainer's "
        "decision; missing evidence is not a pass.",
        "",
    ]
    for key in ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "future_comfort_off_error"):
        value = metrics.get(key)
        records = cast(
            list[dict[str, object]],
            (
                value
                if isinstance(value, list)
                else [value]
                if isinstance(value, dict)
                else []
            ),
        )
        lines += [f"## {key}", "", _table(records), ""]
    lines += [
        "Detailed gate subcriteria, matched counts and reliability bins are in "
        "metrics.json. Per-origin Parquet contains statistics only; full minute "
        "paths are never serialized.",
        "",
    ]
    (directory / "report.md").write_text("\n".join(lines))
    return metrics
