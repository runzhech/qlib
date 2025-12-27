from __future__ import annotations

from typing import Optional

import pandas as pd

from qlib.data import D


def is_gem(code: str) -> bool:
    """创业板：sz.300xxx"""
    if not isinstance(code, str) or "." not in code:
        return False
    exch, sym = code.split(".", 1)
    return exch.lower() == "sz" and sym.startswith("300")


def is_star(code: str) -> bool:
    """科创板：sh.688xxx"""
    if not isinstance(code, str) or "." not in code:
        return False
    exch, sym = code.split(".", 1)
    return exch.lower() == "sh" and sym.startswith("688")


def filter_pred_score_board(
    pred_score: pd.Series,
    *,
    allow_gem: bool,
    allow_star: bool,
) -> pd.Series:
    if pred_score is None or pred_score.empty:
        return pred_score
    inst = pred_score.index.get_level_values(1).astype(str)
    mask = pd.Series(True, index=pred_score.index)
    if not allow_gem:
        mask &= ~inst.map(is_gem).values
    if not allow_star:
        mask &= ~inst.map(is_star).values
    return pred_score[mask.values]


def filter_next_day_limit_up_for_buy(
    pred_score: pd.Series,
    *,
    limit_threshold: float,
    freq: str,
    close_field: str = "$close",
) -> pd.Series:
    """过滤掉“下一交易日涨停”的标的，使 TopK 排序时跳过它们。

    TopkDropoutStrategy 会用上一交易日的 signal 来决定下一交易日的买入列表（shift=1）。
    这里使用回测数据判断交易日 t 是否涨停：

        close_t / close_{t-1} - 1 >= limit_threshold

    若某股票在交易日 t 涨停，则将 pred_score 在 t-1 的对应记录剔除（不参与买入候选）。

    注意：D.features 的索引为 (instrument, datetime)。
    """
    if pred_score is None or pred_score.empty:
        return pred_score

    dates = pd.to_datetime(pred_score.index.get_level_values(0))
    min_pred_dt = pd.Timestamp(dates.min()).normalize()
    max_pred_dt = pd.Timestamp(dates.max()).normalize()

    cal = D.calendar(start_time=min_pred_dt, end_time=max_pred_dt, freq=freq)
    if cal is None or len(cal) < 2:
        return pred_score
    cal = [pd.Timestamp(x).normalize() for x in cal]
    prev_date_map = {cal[i + 1]: cal[i] for i in range(len(cal) - 1)}
    max_trade_dt = cal[-1]

    instruments = sorted(set(pred_score.index.get_level_values(1).astype(str)))
    if not instruments:
        return pred_score

    close_df = D.features(instruments, [close_field], start_time=min_pred_dt, end_time=max_trade_dt, freq=freq)
    if close_df is None or len(close_df) == 0:
        return pred_score

    close_col = close_df.columns[0]
    close_s = close_df[close_col]

    # close_s index: (instrument, datetime)
    prev_close_s = close_s.groupby(level=0).shift(1)
    up_ratio = (close_s / prev_close_s) - 1.0
    limit_up_trade = (up_ratio >= float(limit_threshold)).fillna(False)

    trade_dt = pd.to_datetime(limit_up_trade.index.get_level_values(1)).map(lambda x: pd.Timestamp(x).normalize())
    pred_dt = trade_dt.map(lambda x: prev_date_map.get(x, pd.NaT))
    inst = limit_up_trade.index.get_level_values(0).astype(str)

    limit_up_pred = pd.Series(limit_up_trade.values, index=pd.MultiIndex.from_arrays([pred_dt, inst]))
    limit_up_pred = limit_up_pred[limit_up_pred.index.get_level_values(0).notna()]
    limit_up_pred = limit_up_pred.groupby(level=[0, 1]).max()

    mask = ~limit_up_pred.reindex(pred_score.index, fill_value=False)
    return pred_score[mask.values]


def apply_signal_filters(
    pred_score: pd.Series,
    *,
    allow_gem: bool,
    allow_star: bool,
    skip_next_day_limit_up: bool,
    limit_threshold: float,
    freq: str,
) -> pd.Series:
    out = filter_pred_score_board(pred_score, allow_gem=allow_gem, allow_star=allow_star)
    if skip_next_day_limit_up:
        out = filter_next_day_limit_up_for_buy(out, limit_threshold=limit_threshold, freq=freq)
    return out
