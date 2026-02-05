"""
CLI for kpower-forecast.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import plotext as plt
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from .core import DataCategory, KPowerForecast, MeasurementUnit

# Setup logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("kpower-forecast")

app = typer.Typer(
    help="KPower Forecast CLI: Solar and Power Consumption Forecasting",
    add_completion=False,
)
console = Console()


class ForecastType(str, Enum):
    """Types of forecasts supported by the CLI."""

    solar = "solar"
    consumption = "consumption"


def parse_ha_csv(file_path: Path) -> pd.DataFrame:
    """
    Parses Home Assistant history CSV export.

    Heuristic: Expects 'last_changed' and 'state' columns.

    Args:
        file_path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with 'ds' and 'y' columns.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        console.print(f"[red]Error reading CSV file:[/red] {e}")
        raise typer.Exit(code=1) from None

    # HA Export columns usually include 'last_changed' and 'state'
    if "last_changed" not in df.columns or "state" not in df.columns:
        cols = {col.lower(): col for col in df.columns}
        if "last_changed" in cols:
            df = df.rename(columns={cols["last_changed"]: "ds"})
        elif "last_updated" in cols:
            df = df.rename(columns={cols["last_updated"]: "ds"})
        else:
            msg = "[red]Error:[/red] CSV must contain 'last_changed' or 'last_updated'."
            console.print(msg)
            raise typer.Exit(code=1) from None

        if "state" in cols:
            df = df.rename(columns={cols["state"]: "y"})
        else:
            console.print("[red]Error:[/red] CSV must contain 'state' column.")
            raise typer.Exit(code=1) from None
    else:
        df = df.rename(columns={"last_changed": "ds", "state": "y"})

    # Clean data
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    initial_len = len(df)
    df = df.dropna(subset=["ds", "y"])
    dropped = initial_len - len(df)

    if dropped > 0:
        logger.warning(
            f"Dropped {dropped} rows with invalid data (e.g., 'unavailable')."
        )

    if len(df) == 0:
        console.print("[red]Error:[/red] No valid data found in CSV after parsing.")
        raise typer.Exit(code=1) from None

    return df[["ds", "y"]]


@app.command()
def forecast(
    forecast_type: Annotated[ForecastType, typer.Argument(help="Type of forecast")],
    model_id: Annotated[str, typer.Argument(help="Unique ID for the model")],
    latitude: Annotated[float, typer.Argument(help="Latitude for the location")],
    longitude: Annotated[float, typer.Argument(help="Longitude for the location")],
    input_file: Annotated[
        Path, typer.Option("--input", "-i", help="Path to Home Assistant history CSV")
    ],
    category: Annotated[
        Optional[DataCategory],
        typer.Option("--category", "-c", help="Input data category"),
    ] = None,
    unit: Annotated[
        Optional[MeasurementUnit],
        typer.Option("--unit", "-u", help="Input measurement unit"),
    ] = None,
    horizon_days: Annotated[
        int, typer.Option("--horizon", "-h", help="Days to forecast")
    ] = 7,
    interval_min: Annotated[
        int, typer.Option("--interval", help="Resolution in minutes")
    ] = 60,
    force_retrain: Annotated[
        bool, typer.Option("--force", "-f", help="Force retrain")
    ] = False,
    show_graph: Annotated[
        bool, typer.Option("--graph/--no-graph", help="Display graph")
    ] = True,
    storage_path: Annotated[
        str, typer.Option("--storage", help="Path to store models")
    ] = "./data",
):
    """
    Run forecast for solar production or power consumption.
    """
    # Interactive menu for category if not specified
    if category is None:
        category_str = Prompt.ask(
            "Select input data category",
            choices=[c.value for c in DataCategory],
            default=DataCategory.INSTANT_ENERGY.value,
        )
        category = DataCategory(category_str)

    # Interactive menu for unit if not specified
    if unit is None:
        unit_str = Prompt.ask(
            "Select input measurement unit",
            choices=[u.value for u in MeasurementUnit],
            default=MeasurementUnit.KWH.value,
        )
        unit = MeasurementUnit(unit_str)

    forecaster = KPowerForecast(
        model_id=model_id,
        latitude=latitude,
        longitude=longitude,
        storage_path=storage_path,
        interval_minutes=interval_min,
        forecast_type=forecast_type.value,
        data_category=category,
        unit=unit,
    )

    # 1. Check Model / Training
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        if force_retrain or not forecaster.storage.exists(model_id):
            progress.add_task(description="Parsing input data...", total=None)
            history_df = parse_ha_csv(input_file)

            progress.add_task(description="Training model...", total=None)
            forecaster.train(history_df, force=True)
        else:
            progress.add_task(
                description="Loading model and fetching weather...", total=None
            )

        progress.add_task(description="Generating forecast...", total=None)
        try:
            forecast_df = forecaster.predict(days=horizon_days)
        except Exception as e:
            console.print(f"[red]Error during prediction:[/red] {e}")
            raise typer.Exit(code=1) from None

    # 2. Output Table
    forecast_df["date"] = forecast_df["ds"].dt.date
    daily = (
        forecast_df.groupby("date")
        .agg(
            total_kwh=("yhat", "sum"),
            peak_kw=("yhat", "max"),
            lower_kwh=("yhat_lower", "sum"),
            upper_kwh=("yhat_upper", "sum"),
        )
        .reset_index()
    )

    # Scale by interval to get energy (kWh)
    # Prophet yhat is kWh per interval_minutes
    scale = interval_min / 60.0
    daily["total_kwh"] *= scale
    daily["lower_kwh"] *= scale
    daily["upper_kwh"] *= scale

    table = Table(title=f"Forecast Summary: {forecast_type.value} ({model_id})")
    table.add_column("Date", style="cyan")
    table.add_column("Expected kWh", justify="right", style="green")
    table.add_column("Peak kW", justify="right", style="magenta")
    table.add_column("P10 kWh", justify="right", style="dim")
    table.add_column("P90 kWh", justify="right", style="dim")

    for _, row in daily.iterrows():
        table.add_row(
            str(row["date"]),
            f"{row['total_kwh']:.2f}",
            f"{row['peak_kw']:.2f}",
            f"{row['lower_kwh']:.2f}",
            f"{row['upper_kwh']:.2f}",
        )

    console.print(table)

    # 3. Output Graph
    if show_graph:
        plt.clf()
        plt.theme("dark")
        plt.date_form("Y-m-d H:M")
        x = [d.strftime("%Y-%m-%d %H:%M") for d in forecast_df["ds"]]
        y = forecast_df["yhat"].tolist()
        y_low = forecast_df["yhat_lower"].tolist()
        y_high = forecast_df["yhat_upper"].tolist()

        # Plot bounds first so prediction is on top
        plt.plot(x, y_low, label="P10 (Lower Bound)", color="blue", style="dim")
        plt.plot(x, y_high, label="P90 (Upper Bound)", color="blue", style="dim")
        plt.plot(x, y, label="Prediction (yhat)", color="yellow")

        plt.title(
            f"{forecast_type.value.capitalize()} Forecast: {latitude}, {longitude}"
        )
        plt.xlabel("Time")
        plt.ylabel("kW")
        plt.xfrequency(5)
        plt.show()


if __name__ == "__main__":
    app()
