"""Independent numerical and censored-likelihood regression gates."""

from threading import Event

import numpy as np
import pytest
from scipy.stats import norm

from kpower_forecast.hvac._stats import (
    censored_lognormal_aft,
    huber_regression,
    normal_cdf,
    normal_logsf,
    normal_ppf,
    sample_crps,
    weighted_quantile,
)
from kpower_forecast.hvac.types import HvacCancelled, HvacTimeout


@pytest.fixture
def censored_process():
    """Generate a known AFT process and administrative 30-percent censoring."""
    rng = np.random.default_rng(921)
    x = np.column_stack([np.ones(16000), rng.normal(size=(16000, 2))])
    beta = np.array([2.5, 0.35, -0.2])
    duration = np.exp(x @ beta + 0.45 * rng.normal(size=len(x)))
    limit = np.quantile(duration, 0.7)
    return x, np.minimum(duration, limit), duration > limit, beta


def test_censored_aft_recovers_parameters_and_ignoring_censoring_is_biased(
    censored_process,
):
    x, duration, censored, truth = censored_process
    result = censored_lognormal_aft(x, duration, censored)
    naive = censored_lognormal_aft(x, duration, np.zeros(len(x), dtype=bool))
    assert result.converged
    np.testing.assert_allclose(result.coefficients, truth, rtol=0.05)
    assert result.sigma == pytest.approx(0.45, rel=0.05)
    assert abs(naive.coefficients[1] - truth[1]) > 0.05
    assert not result.coefficients.flags.writeable


def test_aft_offset_ridge_and_nonconvergence_are_explicit(censored_process):
    x, duration, censored, _ = censored_process
    offset = 0.3 * x[:, 1]
    adjusted = duration * np.exp(offset)
    base = censored_lognormal_aft(x, duration, censored)
    translated = censored_lognormal_aft(x, adjusted, censored, offset=offset)
    np.testing.assert_allclose(base.coefficients, translated.coefficients, atol=1e-7)
    shrunk = censored_lognormal_aft(x, duration, censored, ridge=[0, 1e7, 1e7])
    assert shrunk.converged and abs(shrunk.coefficients[1]) < 0.01
    assert not censored_lognormal_aft(x, duration, censored, max_iter=1).converged


def test_normal_probabilities_and_extreme_survival_match_independent_reference():
    z = np.linspace(-10, 10, 1001)
    np.testing.assert_allclose(normal_cdf(z), norm.cdf(z), atol=7.5e-8, rtol=0)
    p = np.r_[np.geomspace(1e-12, 0.01, 30), np.linspace(0.02, 0.98, 50), 1 - 1e-12]
    np.testing.assert_allclose(normal_ppf(p), norm.ppf(p), atol=1e-8, rtol=0)
    tail = np.array([0.0, 5.0, 10.0, 40.0])
    np.testing.assert_allclose(normal_logsf(tail), norm.logsf(tail), atol=2e-5, rtol=0)


def test_robust_regression_weighted_quantiles_and_crps_have_independent_answers():
    x = np.column_stack([np.ones(100), np.linspace(-1, 1, 100)])
    y = 2 + 3 * x[:, 1]
    y[::10] += 100
    np.testing.assert_allclose(huber_regression(x, y), [2, 3], atol=1e-6)
    np.testing.assert_allclose(
        weighted_quantile([1, 2, 1e9], [0, 0.5, 1], [1, 1, 0]), [1, 1.5, 2]
    )
    np.testing.assert_allclose(sample_crps([[0, 3], [2, 5]], [1, 4]), [0.5, 0.5])


def test_invalid_evidence_and_cooperative_cancellation_fail_loudly():
    with pytest.raises(ValueError):
        censored_lognormal_aft([[1], [1]], [1, 2], [True, True])
    with pytest.raises(ValueError):
        normal_ppf([0, 1])
    event = Event()
    event.set()
    with pytest.raises(HvacCancelled):
        censored_lognormal_aft([[1]], [1], [False], cancel=event)
    with pytest.raises(HvacTimeout):
        censored_lognormal_aft([[1]], [1], [False], deadline=0)


def test_offset_only_aft_estimates_scale_without_an_unidentified_intercept():
    rng = np.random.default_rng(45)
    durations = np.exp(2 + 0.4 * rng.normal(size=10000))
    result = censored_lognormal_aft(
        np.empty((len(durations), 0)),
        durations,
        np.zeros(len(durations), dtype=bool),
        offset=np.full(len(durations), 2.0),
    )
    assert result.converged and result.coefficients.size == 0
    assert result.sigma == pytest.approx(0.4, rel=0.03)
