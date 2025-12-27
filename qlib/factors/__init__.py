"""Custom factor library.

Qlib's factor expressions are typically represented as strings in its expression DSL,
then evaluated by `QlibDataLoader` at runtime based on the underlying base fields
(e.g. `$open/$high/$low/$close/$volume`).

This package is a lightweight place to define and reuse such factor expressions.
"""

from .mean_reversion import mrz20_expr, get_feature_config
from .alpha158_plus import Alpha158PlusMRZ

__all__ = [
    "mrz20_expr",
    "get_feature_config",
    "Alpha158PlusMRZ",
]
