"""Utilities package exports.

Keep heavy optional dependencies lazy so importing unrelated modules
(e.g. scientific validators) does not require sentence-transformers.
"""

from typing import Any

__all__ = ["remove_formatting_from_text", "is_text_similar"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .text_similarity import is_text_similar, remove_formatting_from_text

        return {
            "remove_formatting_from_text": remove_formatting_from_text,
            "is_text_similar": is_text_similar,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
