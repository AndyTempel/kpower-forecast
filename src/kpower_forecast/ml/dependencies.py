"""Optional dependency checks for ML backends."""

from importlib.util import find_spec


class MissingMLDependencyError(RuntimeError):
    """Raised when an optional ML backend dependency is unavailable."""


def ensure_optional_dependencies(
    import_names: tuple[str, ...], backend_name: str
) -> None:
    """Ensure optional dependencies for a backend are importable.

    Args:
        import_names: Top-level import names required by the backend.
        backend_name: Human-readable backend name for error messages.

    Returns:
        None.

    Raises:
        MissingMLDependencyError: If any import is unavailable.
    """
    missing = [name for name in import_names if find_spec(name) is None]
    if missing:
        packages = ", ".join(sorted(missing))
        raise MissingMLDependencyError(
            f"{backend_name} requires optional ML dependencies: {packages}. "
            'Install them with: pip install "kpower-forecast[ml]"'
        )
