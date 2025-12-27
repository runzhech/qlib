"""Mean-reversion factor definitions.

This module defines a single factor using Qlib's expression DSL.

Factor: MRZ20 (Mean-Reversion Z-Score, 20-day)

Intuition
---------
- If the current close is far above its recent mean, it tends to mean-revert downward.
- If the current close is far below its recent mean, it tends to mean-revert upward.

Formula (Qlib DSL)
------------------
Let:
- $close be the close price
- Mean($close, 20) be the 20-day moving average
- Std($close, 20) be the 20-day standard deviation

We define a z-score and take a negative sign to align with mean-reversion alpha:

    MRZ20 = ( Mean($close, 20) - $close ) / ( Std($close, 20) + 1e-12 )

Notes
-----
- The small constant 1e-12 prevents division by zero.
- This factor is computed at runtime from base fields, and does not require dumping
  precomputed factors into Qlib's bin storage.

Example usage
-------------
Use it directly in a `QlibDataLoader` config:

    from qlib.utils import init_instance_by_config

    dl_cfg = {
        "class": "QlibDataLoader",
        "module_path": "qlib.data.dataset.loader",
        "kwargs": {"config": {"feature": get_feature_config()}},
    }

Or inside a handler config (e.g. `DataHandlerLP`).
"""

from __future__ import annotations

from typing import List, Tuple


def mrz20_expr(window: int = 20, eps: float = 1e-12) -> str:
    """Return the MRZ factor expression.

    Parameters
    ----------
    window
        Rolling window length.
    eps
        Numerical stability constant.
    """

    # Keep the string in Qlib's expression DSL.
    # NOTE: Qlib's expression AST may not support unary minus on expression nodes.
    # Use an equivalent form to avoid leading `-(...)`.
    w = int(window)
    e = float(eps)
    return f"(Mean($close,{w})-$close)/(Std($close,{w})+{e})"


def get_feature_config(window: int = 20, eps: float = 1e-12) -> Tuple[List[str], List[str]]:
    """Return (fields, names) for plugging into `QlibDataLoader`.

    Returns
    -------
    (fields, names)
        `fields` is a list of DSL expressions, and `names` is a list of column names.
    """

    return [mrz20_expr(window=window, eps=eps)], [f"MRZ{int(window)}"]
