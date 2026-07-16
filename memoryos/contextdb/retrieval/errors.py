"""Typed failures shared by bounded online retrieval components."""


class CatalogCandidateBoundExceeded(RuntimeError):
    """A structured online query cannot be completed within its hard scan bound."""


__all__ = ["CatalogCandidateBoundExceeded"]
