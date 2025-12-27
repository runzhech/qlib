"""JoinQuant small-volume strategy adapted for Qlib.

This module provides a strategy usable by `test_ground/daily_backtest.py`.

What is preserved from the JoinQuant version
-------------------------------------------
- Two regimes:
  1) Stock regime: weekly rebalance on Tuesday.
  2) ETF rotation regime: during specific calendar windows, liquidate non-ETF
     holdings and hold top momentum ETFs.

What is intentionally simplified
--------------------------------
- JoinQuant's original implementation relies on platform fundamentals
  (e.g. market cap) and intraday callbacks (11:00, 14:50, etc).
- In Qlib's daily backtest loop, we keep the *regime switch + rebalance rhythm*
  and use the model signal for stock selection (instead of market-cap ranking).

You can still apply additional filters on the `signal` before backtesting
(e.g. limit-up not buy) in `test_ground/daily_backtest.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import math

import numpy as np
import pandas as pd

from qlib.data import D
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy


def _normalize_cn_code(code: str) -> str:
    """Normalize common CN instrument codes to Qlib-like upper form.

    Examples
    --------
    - "518880.XSHG" -> "SH.518880"
    - "159949.XSHE" -> "SZ.159949"
    - "sh.518880"   -> "SH.518880"
    - "SH.518880"   -> "SH.518880"
    """

    s = code.strip()
    if not s:
        return s

    up = s.upper()
    if up.endswith(".XSHG"):
        return f"SH.{up.split('.')[0].zfill(6)}"
    if up.endswith(".XSHE"):
        return f"SZ.{up.split('.')[0].zfill(6)}"

    if up.startswith("SH.") or up.startswith("SZ."):
        parts = up.split(".")
        if len(parts) == 2:
            return f"{parts[0]}.{parts[1].zfill(6)}"
        return up

    if up.startswith("SH") and len(up) == 8 and up[2] == ".":
        return up

    if up.startswith("SH") and up[2:].isdigit():
        return f"SH.{up[2:].zfill(6)}"
    if up.startswith("SZ") and up[2:].isdigit():
        return f"SZ.{up[2:].zfill(6)}"

    # last resort: keep as-is but upper
    return up


def _is_rotation_period(dt: pd.Timestamp) -> bool:
    """ETF rotation period.

    Matches JoinQuant logic:
    - 12/15 ~ 01/30 (cross-year)
    - 04/04 ~ 04/28
    """

    md = dt.strftime("%m-%d")
    dec_jan = (md >= "12-15") or (md <= "01-30")
    april = (md >= "04-04") and (md <= "04-28")
    return dec_jan or april


def _momentum_score(close: np.ndarray, annual_days: int = 250) -> float:
    """JoinQuant-like momentum score.

    Uses weighted log-price regression slope -> annualized return, multiplied by weighted R^2.
    """

    if close.size < 2:
        return 0.0

    y = np.log(close.astype(float))
    n = y.size
    x = np.arange(n, dtype=float)
    weights = np.linspace(1.0, 2.0, n)

    # weighted linear regression via polyfit
    slope, intercept = np.polyfit(x, y, 1, w=weights)

    # annualized return
    annualized_returns = math.pow(math.exp(slope), annual_days) - 1.0

    # weighted R^2
    y_hat = slope * x + intercept
    residuals = y - y_hat
    wss_res = float(np.sum(weights * residuals**2))
    denom = float(np.sum(weights * (y - float(np.mean(y))) ** 2))
    if denom <= 0:
        r2 = 0.0
    else:
        r2 = 1.0 - wss_res / denom

    return float(annualized_returns * r2)


def _fetch_close_series(code: str, *, end_time: pd.Timestamp, m_days: int) -> Optional[np.ndarray]:
    """Fetch last m_days close prices up to end_time (inclusive)."""

    # Overshoot the lookback in calendar days to tolerate non-trading days.
    start_time = (pd.Timestamp(end_time).normalize() - pd.Timedelta(days=m_days * 4)).strftime("%Y-%m-%d")
    end_str = pd.Timestamp(end_time).normalize().strftime("%Y-%m-%d")

    try:
        df = D.features([code], ["$close"], start_time=start_time, end_time=end_str, freq="day")
    except Exception:
        return None

    if df is None or len(df) == 0:
        return None

    # df: MultiIndex(instrument, datetime) -> column '$close'
    try:
        s = df["$close"]
    except Exception:
        # some providers may name it differently; fallback to first column
        s = df.iloc[:, 0]

    # keep last m_days
    vals = np.asarray(s.values, dtype=float)
    if vals.size < m_days:
        return None
    return vals[-m_days:]


class SmallCapEtfRotationStrategy(TopkDropoutStrategy):
    """A Qlib strategy inspired by JoinQuant smallvolume.py.

    - Non-rotation period: weekly Tuesday rebalance using model `signal`.
      Internally reuses TopkDropoutStrategy logic.
    - Rotation period: ETF momentum rotation. Sell non-ETF holdings and hold
      top momentum ETFs.

    Parameters (stock regime)
    -------------------------
    topk, n_drop, ...
        Same as `TopkDropoutStrategy`.

    Parameters (ETF regime)
    -----------------------
    etf_pool
        ETF codes. Accepts JoinQuant style (e.g. "518880.XSHG") or Qlib style.
    etf_target_num
        Number of ETFs to hold.
    etf_hold_ratio
        Capital ratio (within risk_degree) allocated to ETFs.
    m_days
        Momentum lookback window (trading days).
    """

    def __init__(
        self,
        *,
        etf_pool: Optional[List[str]] = None,
        etf_target_num: int = 2,
        etf_hold_ratio: float = 0.8,
        m_days: int = 25,
        weekly_rebalance_weekday: int = 1,  # Tuesday (Mon=0)
        **kwargs,
    ):
        super().__init__(**kwargs)

        pool = etf_pool or [
            "518880.XSHG",  # Gold ETF
            "513100.XSHG",  # Nasdaq100
            "159949.XSHE",  # ChiNext50
            "588080.XSHG",  # STAR50
            "510180.XSHG",  # SSE180
            "513880.XSHG",  # Nikkei225
            "513030.XSHG",  # DAX30
        ]
        self.etf_pool = [_normalize_cn_code(x) for x in pool]
        self.etf_target_num = int(etf_target_num)
        self.etf_hold_ratio = float(etf_hold_ratio)
        self.m_days = int(m_days)
        self.weekly_rebalance_weekday = int(weekly_rebalance_weekday)

    def _pick_etfs(self, *, pred_end_time: pd.Timestamp) -> List[str]:
        scores = []
        for code in self.etf_pool:
            close = _fetch_close_series(code, end_time=pred_end_time, m_days=self.m_days)
            if close is None:
                continue
            score = _momentum_score(close)
            scores.append((code, float(score)))

        if len(scores) == 0:
            return []

        df = pd.DataFrame(scores, columns=["code", "score"]).set_index("code")
        df = df.sort_values("score", ascending=False)
        positive = df[df["score"] > 0]
        if len(positive) == 0:
            # fallback: keep the full pool
            chosen = df.index.tolist()
        else:
            chosen = positive.index.tolist()

        return chosen[: self.etf_target_num]

    def _generate_etf_trade_decision(self, *, trade_start_time: pd.Timestamp, trade_end_time: pd.Timestamp) -> TradeDecisionWO:
        # Use previous day close information to avoid future leak.
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(self.trade_calendar.get_trade_step(), shift=1)

        target_etfs = self._pick_etfs(pred_end_time=pred_end_time)
        if len(target_etfs) == 0:
            # No ETF data available in provider, skip ETF regime safely.
            return TradeDecisionWO([], self)

        buy_list = [
            c
            for c in target_etfs
            if self.trade_exchange.is_stock_tradable(
                stock_id=c,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.BUY,
            )
        ]

        if len(buy_list) == 0:
            # If we cannot buy any ETFs, do not liquidate existing holdings.
            return TradeDecisionWO([], self)

        target_norm = set(_normalize_cn_code(x) for x in target_etfs)

        current_temp = self.trade_position.__class__(cash=0)  # placeholder, overwritten by deepcopy below
        import copy
        current_temp = copy.deepcopy(self.trade_position)
        cash = current_temp.get_cash()

        # Determine what we currently hold
        current_codes = list(current_temp.get_stock_list())
        current_norm_map = {c: _normalize_cn_code(c) for c in current_codes}

        current_etfs = [c for c in current_codes if current_norm_map[c] in target_norm]
        non_etfs = [c for c in current_codes if current_norm_map[c] not in target_norm]

        # If already aligned and no other holdings, skip trading.
        if len(non_etfs) == 0 and set(current_norm_map[c] for c in current_etfs) == target_norm:
            return TradeDecisionWO([], self)

        sell_orders: List[Order] = []
        buy_orders: List[Order] = []

        # 1) Sell all non-target holdings
        for code in non_etfs:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.SELL,
            ):
                continue
            sell_amount = current_temp.get_stock_amount(code=code)
            if sell_amount <= 0:
                continue
            o = Order(
                stock_id=code,
                amount=sell_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.SELL,
            )
            if self.trade_exchange.check_order(o):
                sell_orders.append(o)
                trade_val, trade_cost, _trade_price = self.trade_exchange.deal_order(o, position=current_temp)
                cash += trade_val - trade_cost

        # 2) Buy target ETFs (equal-weight within etf_hold_ratio * risk_degree)
        per_value = cash * self.get_risk_degree() * self.etf_hold_ratio / len(buy_list)

        for code in buy_list:
            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.BUY,
            )
            if buy_price is None or buy_price <= 0:
                continue
            buy_amount = per_value / buy_price
            factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            if buy_amount <= 0:
                continue
            buy_orders.append(
                Order(
                    stock_id=code,
                    amount=buy_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.BUY,
                )
            )

        return TradeDecisionWO(sell_orders + buy_orders, self)

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)

        # Regime switch
        if _is_rotation_period(pd.Timestamp(trade_start_time)):
            return self._generate_etf_trade_decision(trade_start_time=trade_start_time, trade_end_time=trade_end_time)

        # Stock regime: weekly rebalance (Tuesday by default)
        if pd.Timestamp(trade_start_time).weekday() != self.weekly_rebalance_weekday:
            return TradeDecisionWO([], self)

        return super().generate_trade_decision(execute_result=execute_result)
