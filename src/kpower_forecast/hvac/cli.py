"""Optional HVAC commands; numerical imports remain independent of CLI extras."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal, cast

import pandas as pd
import typer
from rich.console import Console

from ._artifacts import write_report
from ._backtest import backtest as run_backtest
from .bundle import HvacEvaluationBundle
from .evaluation import mixed_grid
from .model import KPowerHvacForecast
from .types import HvacForecastRequest, HvacModelConfig, utc

app = typer.Typer(help="Electrical HVAC forecasts and causal evaluation.")
logger = logging.getLogger(__name__)
console = Console()


@app.command("live-score")
def live_score_command(
    bundle: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    deploy: Annotated[
        str, typer.Option(help="Deployment time; a fix restarts the window.")
    ],
    until: Annotated[str | None, typer.Option(help="Exclusive origin cutoff.")] = None,
    out: Annotated[Path | None, typer.Option(help="New report directory.")] = None,
) -> None:
    """Score archived published forecasts and reserves without running a model."""
    from ._live import live_score

    directory = out or Path.cwd() / f"hvac-live-{datetime.now(UTC):%Y%m%dT%H%M%S}"
    if directory.exists():
        raise typer.BadParameter("output directory already exists")
    source = HvacEvaluationBundle.load(bundle)
    result = live_score(
        source,
        _config(source),
        deploy=_time(deploy),
        until=_time(until) if until else None,
    )
    write_report(result, directory, scope="published_live")
    console.print(f"Report: {directory / 'report.md'}")


def _time(value: str) -> datetime:
    return utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _config(bundle: HvacEvaluationBundle, path: Path | None = None) -> HvacModelConfig:
    if bundle.manifest.ingest_lag_min is None:
        raise ValueError("bundle has no measured ingestion lag")
    values = json.loads(path.read_text()) if path else {}
    values.setdefault("model_id", "hvac")
    values.setdefault("timezone", bundle.manifest.timezone)
    values.setdefault("ingest_lag_min", bundle.manifest.ingest_lag_min)
    return HvacModelConfig.model_validate(values)


@app.command("backtest")
def backtest_command(
    bundle: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    start: Annotated[str, typer.Option(help="Inclusive aware ISO origin.")],
    end: Annotated[str, typer.Option(help="Exclusive aware ISO origin.")],
    schedule: Annotated[str, typer.Option()] = "s0",
    config: Annotated[Path | None, typer.Option(exists=True)] = None,
    out: Annotated[Path | None, typer.Option(help="New report directory.")] = None,
    hours: Annotated[int, typer.Option(help="72 standard; 120 separate M6 pass.")] = 72,
) -> None:
    """Run registered rolling origins and write metrics, gate table and Parquet."""
    if schedule not in ("s0", "s1", "s2") or hours not in (72, 120):
        raise typer.BadParameter("schedule must be s0/s1/s2; hours must be 72/120")
    first, stop = _time(start), _time(end)
    if first >= stop:
        raise typer.BadParameter("start must precede end")
    directory = out or Path.cwd() / f"hvac-backtest-{datetime.now(UTC):%Y%m%dT%H%M%S}"
    if directory.exists():
        raise typer.BadParameter("output directory already exists")
    source = HvacEvaluationBundle.load(bundle)
    origins = tuple(
        stamp.to_pydatetime()
        for stamp in pd.date_range(
            first, stop, freq="h" if hours == 120 else "15min", inclusive="left"
        )
    )
    result = run_backtest(
        source,
        _config(source, config),
        origins,
        schedule=cast(Literal["s0", "s1", "s2"], schedule),
        horizon_h=hours,
        progress=lambda origin, stage: logger.info(
            "HVAC %s %s", origin.isoformat(), stage
        ),
    )
    write_report(result, directory)
    console.print(f"Report: {directory / 'report.md'}")


@app.command("fit")
def fit_command(
    bundle: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    at: Annotated[str, typer.Option(help="Aware ISO fit availability time.")],
    out: Annotated[Path, typer.Option(help="New local artifact directory.")],
    config: Annotated[Path | None, typer.Option(exists=True)] = None,
) -> None:
    """Fit, explicitly install locally, and atomically save an offline artifact."""
    if out.exists():
        raise typer.BadParameter("artifact directory already exists")
    source = HvacEvaluationBundle.load(bundle)
    view = source.available_at(_time(at), include_inputs=True)
    model = KPowerHvacForecast(_config(source, config), storage_dir=out)
    candidate = model.fit(view.training, as_of=view.origin)
    model.install(candidate)
    model.save()
    console.print(f"Saved local {candidate.tier.value} artifact: {model.artifact_path}")
    if candidate.diagnostics.get("t_bal_at_grid_edge"):
        console.print(
            "Balance-temperature boundary warning: maintainer review required."
        )


@app.command("plot")
def plot_command(
    bundle: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    origin: Annotated[str, typer.Option()],
    hours: Annotated[int, typer.Option(min=1, max=120)] = 24,
    schedule: Annotated[str, typer.Option()] = "s0",
    out: Annotated[Path | None, typer.Option(help="New Plotly HTML file.")] = None,
    config: Annotated[Path | None, typer.Option(exists=True)] = None,
) -> None:
    """Plot actuals, forecast means, display-only paths and run probabilities."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if schedule not in ("s0", "s1", "s2"):
        raise typer.BadParameter("schedule must be s0/s1/s2")
    directory = out or Path.cwd() / f"hvac-plot-{datetime.now(UTC):%Y%m%dT%H%M%S}.html"
    if directory.exists():
        raise typer.BadParameter("plot file already exists")
    source = HvacEvaluationBundle.load(bundle)
    view = source.available_at(_time(origin), include_inputs=True)
    if schedule == "s2" and view.schedule_payload is None:
        raise typer.BadParameter("no archived schedule is available")
    model = KPowerHvacForecast(_config(source, config))
    model.install(model.fit(view.training, as_of=view.origin))
    grid = mixed_grid(view.origin, hours=hours)
    forecast = model.predict(
        HvacForecastRequest(
            view.origin,
            view.training.minutes.loc[
                view.training.minutes.index
                >= pd.Timestamp(view.origin) - pd.Timedelta(hours=6)
            ],
            view.spans(
                cast(Literal["s0", "s1", "s2"], schedule),
                device_id=source.manifest.device_id,
                horizon_end=view.origin + timedelta(hours=hours),
            ),
            view.last_mode,
            view.weather_forecast,
            grid,
        )
    )
    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.05
    )
    stamps = [step.start for step in grid]
    for i, path in enumerate(forecast.paths_w):
        figure.add_trace(
            go.Scatter(
                x=stamps,
                y=path,
                mode="lines",
                line={"color": "rgba(90,160,250,0.18)", "width": 1},
                name="Display paths only",
                showlegend=i == 0,
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=stamps,
            y=[
                energy * 60000 / step.minutes
                for energy, step in zip(
                    forecast.intervals.expected_kwh, grid, strict=True
                )
            ],
            name="Expected HVAC W",
        ),
        row=1,
        col=1,
    )
    observed = view.outcomes["minutes.parquet"].loc[
        lambda frame: (
            (frame.index >= view.origin - timedelta(hours=3))
            & (frame.index < grid[-1].end)
        )
    ]
    figure.add_trace(
        go.Scatter(
            x=observed.index,
            y=observed.power_w,
            name="Qualified actual W",
            connectgaps=False,
            line={"color": "orange"},
        ),
        row=1,
        col=1,
    )
    if forecast.intervals.run_probability is not None:
        figure.add_trace(
            go.Scatter(
                x=stamps,
                y=forecast.intervals.run_probability,
                name="Run probability",
                line_shape="hv",
            ),
            row=2,
            col=1,
        )
    figure.update_yaxes(title_text="Power W", row=1, col=1)
    figure.update_yaxes(title_text="P(run)", range=[0, 1], row=2, col=1)
    figure.update_layout(
        title=f"Development forecast · {view.origin.isoformat()} · "
        f"{forecast.tier.value} · {schedule} (S1 is oracle)",
        template="plotly_dark",
        height=750,
    )
    with directory.open("x") as stream:
        stream.write(figure.to_html(include_plotlyjs=True))
    console.print(f"Plot: {directory}")
