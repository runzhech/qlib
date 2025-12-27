"""Extensions to built-in Alpha158.

This module provides a handler that *adds* custom factors on top of `Alpha158`
without replacing Alpha158's original feature set.

Handler: Alpha158PlusMRZ
------------------------
- Base features: all Alpha158 features (same as `qlib.contrib.data.handler.Alpha158`)
- Extra feature: MRZ{window}

Extra factor formula (Qlib DSL)
------------------------------
    MRZ{window} = - ( $close - Mean($close, window) ) / ( Std($close, window) + eps )

This is a classic mean-reversion z-score; the negative sign makes it an alpha that
prefers prices below their recent mean.
"""

from __future__ import annotations

from typing import Tuple, List

from qlib.contrib.data.handler import Alpha158

from .mean_reversion import mrz20_expr


class Alpha158PlusMRZ(Alpha158):
    """Alpha158 with an additional mean-reversion z-score feature."""

    def __init__(self, *args, mrz_window: int = 20, mrz_eps: float = 1e-12, **kwargs):
        self._mrz_window = int(mrz_window)
        self._mrz_eps = float(mrz_eps)
        super().__init__(*args, **kwargs)

    def get_feature_config(self) -> Tuple[List[str], List[str]]:
        fields, names = super().get_feature_config()

        # Append one extra factor.
        fields = list(fields)
        names = list(names)
        fields.append(mrz20_expr(window=self._mrz_window, eps=self._mrz_eps))
        names.append(f"MRZ{self._mrz_window}")
        return fields, names
