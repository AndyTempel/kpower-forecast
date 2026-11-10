"""Forecast APIs, loading optional heavy implementations only when requested."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import KPowerForecast

__all__ = ["KPowerForecast"]
__version__ = "2026.11.3"


def __getattr__(name: str) -> object:
    """Resolve the existing root API without importing it for every subpackage.

    Args:
        name: Requested public attribute.

    Returns:
        The existing forecasting class.

    Raises:
        AttributeError: If the requested name is not a public lazy export.
    """
    if name == "KPowerForecast":
        from .core import KPowerForecast

        globals()[name] = KPowerForecast
        return KPowerForecast
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
