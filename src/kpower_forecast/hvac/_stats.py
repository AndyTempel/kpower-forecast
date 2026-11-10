"""Small numerical tools for HVAC, without scipy or forecasting backends.

Normal probabilities use Abramowitz--Stegun 7.1.26; inverse probabilities use
Acklam's rational approximation. Censored AFT likelihood is optimized directly
by damped Newton steps, including the scale parameter and ridge penalties.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, pi, sqrt
from threading import Event
from time import monotonic

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .types import HvacCancelled, HvacError, HvacTimeout

FloatArray = NDArray[np.float64]
_LOG_2PI: float = float(np.log(2 * pi))
_SQRT_2: float = sqrt(2.0)


def check_budget(cancel: Event | None, deadline: float | None) -> None:
    """Check caller cancellation and an absolute monotonic deadline.

    Args:
        cancel: Optional caller-owned cooperative cancellation flag.
        deadline: Optional absolute monotonic deadline.

    Raises:
        HvacCancelled: If cancelled.
        HvacTimeout: If the deadline has expired.
    """
    if cancel is not None and cancel.is_set():
        raise HvacCancelled("HVAC operation cancelled")
    if deadline is not None and monotonic() >= deadline:
        raise HvacTimeout("HVAC operation exceeded its deadline")


def normal_sf(z: ArrayLike) -> FloatArray:
    """Return the normal upper tail without subtracting a tiny tail from one.

    Args:
        z: Standard normal variates.

    Returns:
        Array of upper-tail probabilities.
    """
    x = np.asarray(z, dtype=float)
    t = 1.0 / (1.0 + 0.2316419 * np.abs(x))
    poly = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    tail = np.exp(-0.5 * x * x) / np.sqrt(2 * pi) * poly
    return np.asarray(np.where(x >= 0, tail, 1 - tail))


def normal_cdf(z: ArrayLike) -> FloatArray:
    """Return normal CDF values (A&S absolute error below 7.5e-8).

    Args:
        z: Standard normal variates.

    Returns:
        Array of lower-tail probabilities.
    """
    return normal_sf(-np.asarray(z, dtype=float))


def normal_logsf(z: ArrayLike) -> FloatArray:
    """Evaluate log survival, retaining finite right-censored extreme tails.

    Args:
        z: Finite normal variates.

    Returns:
        Log upper-tail probabilities, with an asymptotic Mills expansion above 8.
    """
    x = np.asarray(z, dtype=float)
    safe = np.maximum(x, 8.0)
    inv2 = 1 / (safe * safe)
    correction = 1 - inv2 + 3 * inv2**2 - 15 * inv2**3 + 105 * inv2**4 - 945 * inv2**5
    asymptotic = -0.5 * safe**2 - 0.5 * _LOG_2PI - np.log(safe) + np.log(correction)
    # The CDF polynomial's absolute-error guarantee is insufficient for a
    # log-likelihood in the far tail. Use the stdlib erfc for log-survival;
    # the asymptotic branch then avoids erfc underflow at extreme censoring.
    tails = np.fromiter(
        (0.5 * erfc(float(v) / _SQRT_2) for v in x.ravel()),
        dtype=float,
        count=x.size,
    ).reshape(x.shape)
    with np.errstate(divide="ignore"):
        regular = np.log(tails)
    return np.asarray(np.where(x > 8, asymptotic, regular))


def normal_ppf(p: ArrayLike) -> FloatArray:
    """Return Acklam inverse-normal values for probabilities strictly inside (0,1).

    Args:
        p: Finite open-unit-interval probabilities.

    Returns:
        Standard normal quantiles.

    Raises:
        ValueError: If any probability is outside the open unit interval.
    """
    prob = np.asarray(p, dtype=float)
    if not np.isfinite(prob).all() or ((prob <= 0) | (prob >= 1)).any():
        raise HvacError("normal quantiles require 0 < probability < 1")
    a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239,
    ]
    b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
        1.0,
    ]
    c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ]
    d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996,
        3.754408661907416,
        1.0,
    ]
    tail_p = np.minimum(prob, 1 - prob)
    q = np.sqrt(-2 * np.log(tail_p))
    tail = np.polyval(c, q) / np.polyval(d, q)
    centered = prob - 0.5
    r = centered * centered
    middle = np.polyval(a, r) * centered / np.polyval(b, r)
    return np.asarray(
        np.where(tail_p < 0.02425, np.where(prob < 0.5, tail, -tail), middle)
    )


@dataclass(frozen=True)
class AftFit:
    """Censored log-normal coefficients and convergence evidence."""

    coefficients: FloatArray
    sigma: float
    log_likelihood: float
    converged: bool
    iterations: int


def _aft_objective(
    theta: FloatArray,
    x: FloatArray,
    y: FloatArray,
    censored: NDArray[np.bool_],
    offset: FloatArray,
    ridge: FloatArray,
    weights: FloatArray,
) -> tuple[float, FloatArray, FloatArray]:
    """Compute weighted penalized likelihood, score and Hessian analytically."""
    beta, eta = theta[:-1], theta[-1]
    sigma = np.exp(eta)
    z = (y - offset - x @ beta) / sigma
    lam = np.zeros_like(z)
    ll = -eta - 0.5 * z**2 - 0.5 * _LOG_2PI - y
    if censored.any():
        survival = normal_logsf(z[censored])
        lam[censored] = np.exp(-0.5 * z[censored] ** 2 - 0.5 * _LOG_2PI - survival)
        ll[censored] = survival
    prime = lam * (lam - z)
    w = weights
    ll = ll * w
    sb = w * np.where(censored, lam, z) / sigma
    se = w * np.where(censored, lam * z, z**2 - 1)
    hbb = -w * np.where(censored, prime, 1) / sigma**2
    hbe = -w * np.where(censored, prime * z + lam, 2 * z) / sigma
    hee = -w * np.where(censored, prime * z**2 + lam * z, 2 * z**2)
    grad = np.r_[x.T @ sb - ridge * beta, se.sum()]
    hess = np.empty((len(theta), len(theta)))
    hess[:-1, :-1] = (x.T * hbb) @ x - np.diag(ridge)
    hess[:-1, -1] = hess[-1, :-1] = x.T @ hbe
    hess[-1, -1] = hee.sum()
    return float(ll.sum() - 0.5 * np.dot(ridge, beta**2)), grad, hess


def censored_lognormal_aft(
    design: ArrayLike,
    durations: ArrayLike,
    censored: ArrayLike,
    *,
    offset: ArrayLike | None = None,
    ridge: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    max_iter: int = 100,
    tolerance: float = 1e-7,
    cancel: Event | None = None,
    deadline: float | None = None,
) -> AftFit:
    """Fit log duration = offset + design @ beta + normal noise, with censoring.

    Args:
        design: Finite [observation, coefficient] matrix, including any intercept.
        durations: Positive observed durations, including right-censoring times.
        censored: Boolean vector; True contributes survival rather than density.
        offset: Optional known log-duration location per observation.
        ridge: Nonnegative per-coefficient penalties (none on sigma).
        weights: Optional positive per-observation weights (e.g. recency).
        max_iter: Maximum damped Newton steps.
        tolerance: Relative score convergence tolerance.
        cancel: Caller cancellation flag.
        deadline: Absolute monotonic deadline.

    Returns:
        Coefficients, sigma and explicit convergence evidence. Nonconvergence is
        never silently treated as a successful fit.

    Raises:
        ValueError: If inputs are malformed or no complete duration exists.
        HvacCancelled: If cancelled.
        HvacTimeout: If the deadline expires.
    """
    check_budget(cancel, deadline)
    x, t = np.asarray(design, dtype=float), np.asarray(durations, dtype=float)
    c = np.asarray(censored)
    if (
        x.ndim != 2
        or t.ndim != 1
        or x.shape[0] != len(t)
        or c.shape != t.shape
        or c.dtype != bool
        or not len(t)
        or not np.isfinite(x).all()
        or not np.isfinite(t).all()
        or (t <= 0).any()
        or c.all()
    ):
        raise HvacError(
            "AFT requires finite design, positive durations and complete spells"
        )
    off = np.zeros_like(t) if offset is None else np.asarray(offset, dtype=float)
    penalty = np.zeros(x.shape[1]) if ridge is None else np.asarray(ridge, dtype=float)
    wt = np.ones_like(t) if weights is None else np.asarray(weights, dtype=float)
    if wt.shape != t.shape or not np.isfinite(wt).all() or (wt <= 0).any():
        raise HvacError("AFT weights must be positive and finite per observation")
    if (
        off.shape != t.shape
        or penalty.shape != (x.shape[1],)
        or not np.isfinite(off).all()
        or not np.isfinite(penalty).all()
        or (penalty < 0).any()
        or max_iter < 1
        or tolerance <= 0
    ):
        raise HvacError("invalid AFT offset, penalties or convergence settings")
    y = np.log(t)
    beta = np.linalg.lstsq(x, y - off, rcond=None)[0]
    sigma = max(float(np.std(y - off - x @ beta)), 0.1)
    theta = np.r_[beta, np.log(sigma)]
    converged = False
    iteration = 0
    for _iteration in range(1, max_iter + 1):
        iteration = _iteration
        check_budget(cancel, deadline)
        value, grad, hess = _aft_objective(theta, x, y, c, off, penalty, wt)
        if np.max(np.abs(grad)) <= tolerance * (1 + len(t)):
            converged = True
            break
        information = -hess
        minimum = float(np.linalg.eigvalsh(information)[0])
        regularization = max(1e-8, -minimum + 1e-6)
        delta = np.linalg.solve(information + np.eye(len(theta)) * regularization, grad)
        # Step halving is also the safeguard against an extreme scale excursion.
        accepted = False
        for half in range(30):
            candidate = theta + delta * 0.5**half
            if not -12 < candidate[-1] < 5:
                continue
            new_value = _aft_objective(candidate, x, y, c, off, penalty, wt)[0]
            if np.isfinite(
                new_value
            ) and new_value >= value + 1e-4 * 0.5**half * np.dot(grad, delta):
                theta, accepted = candidate, True
                break
        if not accepted:
            break
        # At the optimum, floating-point noise can keep the score just above
        # the gradient tolerance; a negligible objective change is convergence.
        if abs(new_value - value) <= 1e-12 * (1 + abs(value)) and np.max(
            np.abs(grad)
        ) <= 1e-3 * (1 + len(t)):
            converged = True
            break
    value = _aft_objective(theta, x, y, c, off, penalty, wt)[0]
    coefficients = theta[:-1].copy()
    coefficients.setflags(write=False)
    return AftFit(coefficients, float(np.exp(theta[-1])), value, converged, iteration)


def huber_regression(
    design: ArrayLike, target: ArrayLike, *, max_iter: int = 100
) -> FloatArray:
    """Fit a Huber regression with delta = 1.345 times the residual MAD.

    Args:
        design: Finite two-dimensional predictor matrix, including any intercept.
        target: Finite response vector.
        max_iter: Maximum IRLS steps.

    Returns:
        Regression coefficients.

    Raises:
        ValueError: For malformed or empty data.
    """
    x, y = np.asarray(design, dtype=float), np.asarray(target, dtype=float)
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or not len(y)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise HvacError("Huber regression requires a finite matrix and target")
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    for _ in range(max_iter):
        residual = y - x @ beta
        mad = float(np.median(np.abs(residual - np.median(residual))))
        delta = max(1.345 * mad, 1e-9)
        weights = np.minimum(1.0, delta / np.maximum(np.abs(residual), 1e-12))
        root = np.sqrt(weights)
        updated = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)[0]
        if np.max(np.abs(updated - beta)) < 1e-8:
            beta = updated
            break
        beta = updated
    return np.asarray(beta)


def weighted_quantile(
    values: ArrayLike, probabilities: ArrayLike, weights: ArrayLike
) -> FloatArray:
    """Interpolate quantiles at the midpoints of positive empirical weight masses.

    Args:
        values: Finite one-dimensional sample values.
        probabilities: Quantiles in the closed unit interval.
        weights: Nonnegative finite sample weights with positive total mass.

    Returns:
        Quantiles with the probabilities' shape.

    Raises:
        ValueError: For invalid samples, weights or probabilities.
    """
    v, q, w = (
        np.asarray(values, dtype=float),
        np.asarray(probabilities, dtype=float),
        np.asarray(weights, dtype=float),
    )
    if (
        v.ndim != 1
        or w.shape != v.shape
        or not np.isfinite(v).all()
        or not np.isfinite(w).all()
        or (w < 0).any()
        or w.sum() <= 0
        or not np.isfinite(q).all()
        or ((q < 0) | (q > 1)).any()
    ):
        raise HvacError("invalid weighted quantile inputs")
    keep = w > 0
    order = np.argsort(v[keep], kind="stable")
    v, w = v[keep][order], w[keep][order]
    positions = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.asarray(np.interp(q, positions, v))


def sample_crps(samples: ArrayLike, observed: ArrayLike) -> FloatArray:
    """Compute empirical CRPS across axis zero in O(n log n) time.

    Args:
        samples: Finite ensemble, with paths along axis zero.
        observed: Finite actual values, broadcastable over the other axes.

    Returns:
        CRPS for each output, with the same units as the samples.

    Raises:
        ValueError: For empty or nonfinite evidence.
    """
    x, y = np.asarray(samples, dtype=float), np.asarray(observed, dtype=float)
    if x.ndim < 1 or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise HvacError("CRPS requires nonempty finite samples and actuals")
    ordered = np.sort(x, axis=0)
    n = len(x)
    ranks = (2 * np.arange(1, n + 1) - n - 1).reshape((n,) + (1,) * (x.ndim - 1))
    return np.asarray(
        np.mean(np.abs(x - y), axis=0) - np.sum(ranks * ordered, axis=0) / n**2
    )


def conditional_lognormal(
    mu: ArrayLike,
    sigma: ArrayLike,
    elapsed: ArrayLike,
    rng: np.random.Generator,
) -> FloatArray:
    """Sample remaining duration conditional on survival through elapsed time.

    Args:
        mu: Log-median duration, possibly +inf for a non-ending state.
        sigma: Positive log-scale dispersion.
        elapsed: Nonnegative observed age in minutes.
        rng: Caller-owned deterministic generator.

    Returns:
        Nonnegative remaining minutes. Log-survival inversion avoids rounding
        extreme conditional probabilities to one.

    Raises:
        ValueError: For malformed parameters or elapsed ages.
    """
    m, s, e = np.broadcast_arrays(
        np.asarray(mu, dtype=float),
        np.asarray(sigma, dtype=float),
        np.asarray(elapsed, dtype=float),
    )
    if (
        np.isnan(m).any()
        or np.isneginf(m).any()
        or not np.isfinite(s).all()
        or (s <= 0).any()
        or not np.isfinite(e).all()
        or (e < 0).any()
    ):
        raise HvacError("invalid conditional lognormal parameters")
    result = np.full(m.shape, np.inf)
    use = np.isfinite(m)
    if not use.any():
        return result
    mm, ss, ee = m[use], s[use], e[use]
    z_age = (np.log(np.maximum(ee, 1e-300)) - mm) / ss
    log_survival = np.where(ee > 0, normal_logsf(z_age), 0.0)
    target = log_survival + np.log(rng.uniform(np.finfo(float).tiny, 1.0, len(mm)))
    # Inverse CDF is accurate in the central region; asymptotic initialization
    # remains finite when 1-exp(target) would round to exactly one.
    z = np.sqrt(np.maximum(-2 * target, 0.0))
    central = target > -30
    z[central] = normal_ppf(-np.expm1(target[central]))
    for _ in range(6):
        survival = normal_logsf(z)
        mills = np.exp(-0.5 * z * z - 0.5 * _LOG_2PI - survival)
        z += (survival - target) / mills
    result[use] = np.maximum(np.exp(np.minimum(mm + ss * z, 700)) - ee, 0.0)
    return result
