"""Command-line rolling-origin benchmark for consumption history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from kpower_forecast.ml.baselines import (
    BASELINE_NAME,
    evaluate_local_slot_baseline,
)


def main() -> None:
    """Evaluate the standard naive baseline and print JSON metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history_csv", type=Path)
    parser.add_argument("--timezone", default="UTC")
    parser.add_argument("--interval-minutes", type=int, default=15)
    parser.add_argument("--origin-spacing-hours", type=int, default=6)
    parser.add_argument("--evaluation-days", type=int, default=28)
    args = parser.parse_args()

    history = pd.read_csv(args.history_csv)
    history["ds"] = pd.to_datetime(history["ds"], utc=True)
    end = history["ds"].max().floor(f"{args.interval_minutes}min")
    start = end - pd.Timedelta(days=args.evaluation_days)
    origins = pd.date_range(
        start=start,
        end=end - pd.Timedelta(days=1),
        freq=f"{args.origin_spacing_hours}h",
        tz="UTC",
    )
    periods_per_hour = 60 // args.interval_minutes
    horizons = [periods_per_hour, 6 * periods_per_hour, 24 * periods_per_hour]
    metrics = evaluate_local_slot_baseline(
        history,
        origins=[origin.to_pydatetime() for origin in origins],
        horizon_periods=horizons,
        interval_minutes=args.interval_minutes,
        timezone=args.timezone,
    )
    payload = {
        "baseline": BASELINE_NAME,
        "interval_minutes": args.interval_minutes,
        "timezone": args.timezone,
        "horizons": {
            str(periods * args.interval_minutes // 60): result.as_dict()
            for periods, result in metrics.items()
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
