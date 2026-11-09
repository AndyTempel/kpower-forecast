"""Rolling-origin orchestration through one causal bundle gateway per origin."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from time import perf_counter
from typing import Callable, Literal, Sequence, cast

import numpy as np
import pandas as pd

from ._baselines import legacy_baseline, r1c_probabilities, simple_baselines
from ._block_scores import block_scores
from ._scoring import cadence_scores, energy_scores, quarter_scores, run_scores
from .bundle import BundleSnapshot, HvacEvaluationBundle
from .evaluation import mixed_grid, regular_grid
from .frames import policy_number
from .model import KPowerHvacForecast, _predict_candidate, fingerprint
from .settlement import SettlementWindow
from .types import HvacError, HvacForecastRequest, HvacModelConfig, JsonValue, utc


@dataclass(frozen=True)
class BacktestResult:
    """Aggregate scoring rows only; minute paths never escape the scoring process."""

    tables: dict[str, pd.DataFrame]
    config: HvacModelConfig
    schedule: str
    horizon_h: int


def _windows(view: BundleSnapshot, end: datetime) -> tuple[SettlementWindow, ...]:
    rows = view.outcomes.get("tariff_quarters.parquet", pd.DataFrame())
    if rows.empty:
        return ()
    rows = rows.loc[
        (rows.window_start >= view.origin) & (rows.window_end <= end)
    ].sort_values("window_start")
    windows = []
    for row in rows.itertuples():
        start = cast(pd.Timestamp, row.window_start).to_pydatetime()
        finish = cast(pd.Timestamp, row.window_end).to_pydatetime()
        target = cast(float, row.import_target_w)
        hard = cast(float, row.hard_import_limit_w)
        windows.append(
            SettlementWindow(
                start,
                finish,
                str(row.block_id),
                (
                    float(target) * (finish - start).total_seconds() / 3600
                    if pd.notna(target)
                    else None
                ),
                float(hard) if pd.notna(hard) else None,
            )
        )
    return tuple(windows)


def _site_quality(view: BundleSnapshot, site: pd.DataFrame) -> pd.Series:
    # EMS emits base coverage only for attributable intervals with usable site
    # inputs. Apply its resolved full-row ratio, never observation-only validity.
    coverage = site.get("base_coverage_s", pd.Series(np.nan, index=site.index))
    required = policy_number(
        view.training.history_policy, "_FULL_ATTRIBUTION_COVERAGE_RATIO"
    )
    return cast(pd.Series, coverage.div(60).ge(required) & coverage.le(60))


def backtest(
    bundle: HvacEvaluationBundle,
    config: HvacModelConfig,
    origins: Sequence[datetime],
    *,
    schedule: Literal["s0", "s1", "s2"] = "s0",
    horizon_h: int = 72,
    phase_ablation: bool = True,
    progress: Callable[[datetime, str], None] | None = None,
) -> BacktestResult:
    """Fit every six hours and score causal forecasts on development data only.

    Args:
        bundle: Validated input inventory. Scoring accesses it exclusively through
            available_at(origin, include_inputs=True), once per origin.
        config: Frozen configuration, with the bundle's measured ingestion lag.
        origins: Strictly increasing quarter-aligned UTC origins; the registered
            standard run uses every quarter, the 120-hour M6 pass every hour.
        schedule: Held mode, explicitly oracle commands, or archived schedule.
        horizon_h: Standard 72 hours, or separate 120-hour energy-only pass.
        phase_ablation: Include B4 with all current-state observations hidden.
        progress: Optional bounded progress callback, without sensitive inputs.

    Returns:
        Per-origin scores and timings, fits and provenance. S2 missing a schedule
        is recorded as skipped rather than silently substituted with S0.

    Raises:
        ValueError: For invalid origins/contracts, fit failures or missing weather.
            In particular AFT nonconvergence is surfaced, never silently retried
            with relaxed settings or a different model.
    """
    if horizon_h not in (72, 120):
        raise HvacError("backtest requires the registered 72h or 120h pass")
    if schedule not in ("s0", "s1", "s2"):
        raise HvacError("unknown schedule variant")
    times = tuple(utc(origin) for origin in origins)
    if not times or any(
        t.minute % (60 if horizon_h == 120 else 15) or t.second or t.microsecond
        for t in times
    ):
        raise HvacError("backtest origins must follow the registered grid")
    if any(a >= b for a, b in zip(times, times[1:], strict=False)):
        raise HvacError("backtest origins must increase strictly")
    if (
        config.ingest_lag_min != bundle.manifest.ingest_lag_min
        or config.timezone != bundle.manifest.timezone
    ):
        raise HvacError("backtest config disagrees with bundle availability contract")
    model = KPowerHvacForecast(config)
    tables: dict[str, list[pd.DataFrame]] = {
        name: []
        for name in (
            "quarters",
            "runs",
            "energy",
            "cadence",
            "blocks",
            "runtime",
            "fits",
            "origins",
        )
    }
    fitted_at: datetime | None = None
    fitted_identity: str | None = None
    threshold: float | None = None
    for origin in times:
        before = {name: len(parts) for name, parts in tables.items()}
        if progress:
            progress(origin, "snapshot")
        view = bundle.available_at(origin, include_inputs=True)
        if schedule == "s2" and view.schedule_payload is None:
            tables["origins"].append(
                pd.DataFrame(
                    [
                        {
                            "origin": origin,
                            "status": "skipped_missing_schedule",
                            "schedule": schedule,
                        }
                    ]
                )
            )
            continue
        identity = fingerprint(view.training, config)
        fit_seconds = 0.0
        if (
            fitted_at is None
            or origin - fitted_at >= timedelta(hours=6)
            or identity != fitted_identity
        ):
            if progress:
                progress(origin, "fit")
            began = perf_counter()
            try:
                candidate = model.fit(view.training, as_of=origin)
            except HvacError as exc:
                if str(exc) not in {
                    "no_eligible_history",
                    "insufficient_energy_history",
                    "insufficient_weather_calibration",
                    "no_outdoor_history",
                }:
                    raise
                tables["origins"].append(
                    pd.DataFrame(
                        [
                            {
                                "origin": origin,
                                "status": "skipped_unavailable_fit",
                                "reason": str(exc),
                                "schedule": schedule,
                            }
                        ]
                    )
                )
                continue
            fit_seconds = perf_counter() - began
            model.install(candidate)
            fitted_at, fitted_identity = origin, candidate.fingerprint
            value = candidate.parameters.get("threshold_w")
            threshold = float(value) if isinstance(value, int | float) else None
            fit_record: dict[str, object] = {
                "origin": origin,
                "fingerprint": candidate.fingerprint,
                "tier": candidate.tier.value,
                "fallback_reason": candidate.fallback_reason,
                "fit_seconds": fit_seconds,
            }
            fit_record.update(dict(candidate.diagnostics))
            tables["fits"].append(pd.DataFrame([fit_record]))
        quarter_grid = regular_grid(origin, minutes=15, hours=horizon_h)
        baselines = simple_baselines(view, config, quarter_grid)
        baselines["B3"] = legacy_baseline(view, quarter_grid)
        request = HvacForecastRequest(
            origin,
            view.training.minutes.loc[
                view.training.minutes.index
                >= pd.Timestamp(origin) - pd.Timedelta(hours=6)
            ],
            view.spans(
                schedule,
                device_id=bundle.manifest.device_id,
                horizon_end=origin + timedelta(hours=horizon_h),
            ),
            view.last_mode,
            view.weather_forecast,
            mixed_grid(origin, hours=horizon_h),
        )
        if progress:
            progress(origin, "predict")
        began = perf_counter()
        forecast = model.predict(request)
        predict_seconds = perf_counter() - began
        actual = view.outcomes["minutes.parquet"]
        tables["energy"].append(
            energy_scores(forecast, actual, baselines, hours=horizon_h)
        )
        if horizon_h == 72:
            tables["quarters"].append(
                quarter_scores(forecast, actual, baselines, threshold_w=threshold)
            )
            b5 = r1c_probabilities(view, regular_grid(origin, minutes=5, hours=1))
            tables["runs"].append(
                run_scores(forecast, actual, threshold_w=threshold, r1c=b5)
            )
            tables["cadence"].append(
                cadence_scores(forecast, actual, threshold_w=threshold)
            )
            windows = _windows(view, origin + timedelta(hours=horizon_h))
            site = view.outcomes.get("site_minutes.parquet", pd.DataFrame())
            if windows and not site.empty:
                tables["blocks"].append(
                    block_scores(
                        forecast,
                        actual,
                        site,
                        _site_quality(view, site),
                        windows,
                        baselines,
                    )
                )
            if phase_ablation:
                active = model.active_fit
                assert active is not None
                unknown = _predict_candidate(
                    request, active, config, force_phase_unknown=True
                )
                for name, rows in (
                    (
                        "quarters",
                        quarter_scores(unknown, actual, {}, threshold_w=threshold),
                    ),
                    ("runs", run_scores(unknown, actual, threshold_w=threshold)),
                    ("energy", energy_scores(unknown, actual, {}, hours=horizon_h)),
                ):
                    rows["model"] = "B4"
                    tables[name].append(rows)
                if windows and not site.empty:
                    rows = block_scores(
                        unknown, actual, site, _site_quality(view, site), windows, {}
                    )
                    rows["model"] = "B4"
                    tables["blocks"].append(rows)
        events = view.outcomes.get("mode_events.parquet", pd.DataFrame())
        comfort_spans = []
        if not events.empty:
            ordered = events.sort_values("time", kind="stable")
            for i, event in enumerate(ordered.itertuples()):
                event_time = cast(pd.Timestamp, event.time).to_pydatetime()
                if (
                    event_time > origin
                    and event.mode == "OFF"
                    and event.source == "thermostat"
                ):
                    end = (
                        ordered.time.iloc[i + 1]
                        if i + 1 < len(ordered)
                        else origin + timedelta(hours=horizon_h)
                    )
                    comfort_spans.append((event_time, end))
        for name in ("quarters", "runs"):
            for rows in tables[name][before[name] :]:
                affected = np.zeros(len(rows), dtype=bool)
                for start, end in comfort_spans:
                    affected |= (
                        (rows.interval_start < end)
                        & (
                            rows.interval_start
                            + pd.to_timedelta(rows.minutes, unit="min")
                            > start
                        )
                    ).to_numpy()
                rows["future_comfort_off"] = affected
        tables["runtime"].append(
            pd.DataFrame(
                [
                    {
                        "origin": origin,
                        "fit_seconds": fit_seconds,
                        "predict_seconds": predict_seconds,
                        "peak_added_rss_mb": np.nan,
                        "rss_status": "not_measured",
                        "runtime_scope": "evaluation_host_not_pi_gate",
                    }
                ]
            )
        )
        provenance: dict[str, JsonValue] = dict(view.provenance)
        tables["origins"].append(
            pd.DataFrame(
                [
                    {
                        "origin": origin,
                        "status": "scored",
                        "schedule": schedule,
                        "weather_vintages": ",".join(view.vintage_kinds),
                        "fingerprint": forecast.fingerprint,
                        **provenance,
                    }
                ]
            )
        )
    return BacktestResult(
        {
            name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            for name, parts in tables.items()
        },
        config,
        schedule,
        horizon_h,
    )
