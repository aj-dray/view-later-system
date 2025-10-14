from .base import SearchItem

# Avoid eager imports of agent implementations to prevent circular imports.
# Expose them lazily via __getattr__ (PEP 562) when accessed.

__all__ = [
    "SearchItem",
    "StructuredAgent",
    "UnstructuedAgent",
]


def __getattr__(name: str):  # pragma: no cover - thin import shim
    if name == "StructuredAgent":
        from .structured_agent import (
            StructuredAgent as _StructuredAgent,
        )

        return _StructuredAgent
    if name == "UnstructuedAgent":
        from .unstructued_agent import (
            UnstructuedAgent as _UnstructuedAgent,
        )

        return _UnstructuedAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
