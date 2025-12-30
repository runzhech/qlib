#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""\
全A股日频数据下载脚本 (TuShare版)

功能：使用 TuShare Pro 批量下载全A股日K数据并保存为 Qlib 目录结构。
说明：脚本结构参考 `download_all_stocks_daily_baostock.py`，支持增量更新。

依赖：
- tushare（TuShare Pro）

使用前准备：
- 配置 TuShare Token（任选其一）：
  - 环境变量：TUSHARE_TOKEN 或 TUSHARE_PRO_TOKEN
  - 命令行参数：--token <YOUR_TOKEN>

注意：
- TuShare Pro 存在接口频率限制；全量跑全市场会很慢，建议先用 --limit 测试。
- 默认下载最近 365 天数据。

"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from data.qlib_converter import QlibConverter
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保在项目根目录下运行此脚本")
    raise


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_daily_stocks_tushare.log', encoding='utf-8'),
        logging.StreamHandler(),
    ],
    force=True,
)
logger = logging.getLogger(__name__)


def _normalize_date_arg(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        return None if v == '' else v
    return value


def _to_ymd(date_str: str) -> str:
    """YYYY-MM-DD -> YYYYMMDD"""
    return pd.to_datetime(date_str).strftime('%Y%m%d')


def _to_qlib_symbol(ts_code: str) -> str:
    """TuShare ts_code -> qlib symbol.

    例如：
    - 600000.SH -> sh.600000
    - 000001.SZ -> sz.000001
    """
    ts_code = str(ts_code).strip()
    if '.' not in ts_code:
        raise ValueError(f"非法 ts_code: {ts_code}")
    code, exch = ts_code.split('.', 1)
    exch = exch.upper()
    if exch == 'SH':
        return f"sh.{code}"
    if exch == 'SZ':
        return f"sz.{code}"
    if exch == 'BJ':
        return f"bj.{code}"
    return f"{exch.lower()}.{code}"


def _parse_ts_codes(value: Optional[str]) -> List[str]:
    """解析逗号分隔的 ts_code 列表。"""
    if value is None:
        return []
    if not isinstance(value, str):
        return []
    raw = [x.strip() for x in value.split(',')]
    codes = [x for x in raw if x]
    seen = set()
    out: List[str] = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


class TuShareDailyDownloader:
    """全A股日频数据下载器 (基于 TuShare Pro)"""

    def __init__(
        self,
        output_dir: str = "./data/qlib_data_daily",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "qfq",
        format: str = "csv",
        token: Optional[str] = None,
        sleep_seconds: float = 0.2,
        download_etf: bool = True,
        download_index: bool = True,
        index_codes: Optional[List[str]] = None,
    ):
        self.output_dir = Path(output_dir)

        start_date = _normalize_date_arg(start_date)
        end_date = _normalize_date_arg(end_date)

        now = datetime.now()
        self.start_date = start_date or (now - timedelta(days=365)).strftime('%Y-%m-%d')
        self.end_date = end_date or now.strftime('%Y-%m-%d')

        self.adjust = adjust
        self.format = format
        self.sleep_seconds = float(sleep_seconds)

        self.download_etf = bool(download_etf)
        self.download_index = bool(download_index)
        self.index_codes = list(index_codes) if index_codes else []

        # token
        self.token = (
            _normalize_date_arg(token)
            or os.environ.get('TUSHARE_TOKEN')
            or os.environ.get('TUSHARE_PRO_TOKEN')
        )
        if not self.token:
            raise RuntimeError(
                "未检测到 TuShare Token。请设置环境变量 TUSHARE_TOKEN/TUSHARE_PRO_TOKEN 或传入 --token。"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.qlib_converter = QlibConverter({
            'qlib_data_path': str(self.output_dir),
            'qlib_cache_path': str(self.output_dir.parent / 'cache'),
        })

        self.stats: Dict[str, object] = {
            'total_stocks': 0,
            'success_count': 0,
            'failed_count': 0,
            'start_time': None,
            'end_time': None,
            'failed_stocks': [],
            'success_stocks': [],
        }

        logger.info("TuShare日频数据下载器初始化完成")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"时间范围: {self.start_date} 到 {self.end_date}")
        logger.info(f"复权方式: {self.adjust}")
        logger.info(f"输出格式: {self.format}")
        logger.info(f"sleep_seconds: {self.sleep_seconds}")
        logger.info(f"下载ETF数据: {self.download_etf}")
        logger.info(f"下载指数数据: {self.download_index}")
        if self.index_codes:
            logger.info(f"指数代码: {','.join(self.index_codes)}")

        self._pro = None

    def _get_pro(self):
        if self._pro is not None:
            return self._pro

        try:
            import tushare as ts
        except Exception as e:
            raise RuntimeError(f"tushare 不可用: {e}. 请安装: pip install tushare")

        ts.set_token(self.token)
        self._pro = ts.pro_api(self.token)
        return self._pro

    def get_all_stocks(self) -> List[str]:
        """获取所有A股股票列表（ts_code 列表，如 600000.SH）。"""
        logger.info("正在获取A股股票列表 (TuShare stock_basic)...")
        pro = self._get_pro()

        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,market,list_date')
        if df is None or df.empty:
            raise RuntimeError("TuShare stock_basic 返回为空，无法获取股票列表")

        # 仅保留沪深北
        df['ts_code'] = df['ts_code'].astype(str)
        df = df[df['ts_code'].str.endswith(('.SH', '.SZ', '.BJ'))]

        codes = df['ts_code'].dropna().astype(str).tolist()
        # 去重保持顺序
        seen = set()
        unique: List[str] = []
        for c in codes:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        logger.info(f"A股股票标的合计: {len(unique)}")
        return unique

    def get_all_etfs(self) -> List[str]:
        """获取所有可交易ETF列表（TuShare fund_basic）。

        说明：TuShare 的 ETF 多数可通过 fund_basic(market='E') 获取。
        本方法做了尽量宽松的筛选：优先识别 fund_type/type/invest_type 含 ETF，
        若缺失则按常见代码段（5xxxxx/1xxxxx）兜底。
        """
        logger.info("正在获取ETF列表 (TuShare fund_basic)...")
        pro = self._get_pro()

        try:
            df = pro.fund_basic(market='E', status='L')
        except Exception as e:
            logger.warning(f"fund_basic 调用失败: {e}")
            return []

        if df is None or df.empty or 'ts_code' not in df.columns:
            logger.warning("TuShare fund_basic 返回为空或缺少 ts_code，跳过ETF下载")
            return []

        df = df.copy()
        df['ts_code'] = df['ts_code'].astype(str)
        df = df[df['ts_code'].str.endswith(('.SH', '.SZ'))]

        def _is_etf_row(row: pd.Series) -> bool:
            for col in ('fund_type', 'type', 'invest_type'):
                if col in row and pd.notna(row[col]):
                    if 'ETF' in str(row[col]).upper():
                        return True
            code = str(row.get('ts_code', ''))
            if '.' in code:
                code6 = code.split('.', 1)[0]
                if len(code6) == 6 and code6.isdigit() and (code6.startswith('5') or code6.startswith('1')):
                    return True
            name = str(row.get('name', ''))
            if 'ETF' in name.upper():
                return True
            return False

        mask = df.apply(_is_etf_row, axis=1)
        df = df[mask]

        codes = df['ts_code'].dropna().astype(str).tolist()
        seen = set()
        unique: List[str] = []
        for c in codes:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        logger.info(f"ETF标的合计: {len(unique)}")
        return unique

    def get_indices(self) -> List[str]:
        """获取指数列表。

        - 若用户通过 index_codes 指定，则使用指定列表
        - 否则默认下载：沪深300(000300.SH)、中证500(000905.SH)、上证50(000016.SH)
        """
        if self.index_codes:
            return self.index_codes
        return ['000300.SH', '000905.SH', '000016.SH']

    def _download_daily_by_asset(self, ts_code: str, start_date: str, end_date: str, asset: str) -> pd.DataFrame:
        """使用 ts.pro_bar 下载日K（asset: E=股票, FD=基金/ETF, I=指数）。"""
        start_ymd = _to_ymd(start_date)
        end_ymd = _to_ymd(end_date)

        try:
            import tushare as ts
        except Exception as e:
            raise RuntimeError(f"tushare 不可用: {e}. 请安装: pip install tushare")

        adj = None
        # 指数通常不支持复权；为避免报错，这里仅对股票/ETF传 adj
        if asset in ('E', 'FD') and self.adjust in ('qfq', 'hfq'):
            adj = self.adjust

        df = ts.pro_bar(
            ts_code=ts_code,
            start_date=start_ymd,
            end_date=end_ymd,
            adj=adj,
            freq='D',
            asset=asset,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        return df

    def download_stock_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """下载单只股票日K（优先 pro_bar，失败回退 pro.daily）。"""
        pro = self._get_pro()
        try:
            return self._download_daily_by_asset(ts_code, start_date, end_date, asset='E')
        except Exception as e:
            logger.warning(f"ts.pro_bar 失败 {ts_code}: {e}，回退到 pro.daily")
            start_ymd = _to_ymd(start_date)
            end_ymd = _to_ymd(end_date)
            df = pro.daily(ts_code=ts_code, start_date=start_ymd, end_date=end_ymd)
            return df if df is not None else pd.DataFrame()

    def download_etf_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """下载单个ETF日K（asset='FD'）。"""
        try:
            df = self._download_daily_by_asset(ts_code, start_date, end_date, asset='FD')
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning(f"ETF pro_bar(asset='FD') 失败 {ts_code}: {e}，尝试回退 fund_daily")

        pro = self._get_pro()
        start_ymd = _to_ymd(start_date)
        end_ymd = _to_ymd(end_date)
        try:
            df = pro.fund_daily(ts_code=ts_code, start_date=start_ymd, end_date=end_ymd)
        except Exception as e:
            logger.warning(f"ETF fund_daily 失败 {ts_code}: {e}")
            return pd.DataFrame()

        return df if df is not None else pd.DataFrame()

    def download_index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """下载单个指数日K（asset='I'）。"""
        try:
            df = self._download_daily_by_asset(ts_code, start_date, end_date, asset='I')
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning(f"指数 pro_bar(asset='I') 失败 {ts_code}: {e}，尝试回退 index_daily")

        pro = self._get_pro()
        start_ymd = _to_ymd(start_date)
        end_ymd = _to_ymd(end_date)
        try:
            df = pro.index_daily(ts_code=ts_code, start_date=start_ymd, end_date=end_ymd)
        except Exception as e:
            logger.warning(f"指数 index_daily 失败 {ts_code}: {e}")
            return pd.DataFrame()

        return df if df is not None else pd.DataFrame()

    def _preprocess_data(self, df: pd.DataFrame, qlib_symbol: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        # ts.pro_bar/pro.daily: trade_date 为 YYYYMMDD
        date_col = 'trade_date' if 'trade_date' in df.columns else 'date'
        if date_col not in df.columns:
            return pd.DataFrame()

        out = df.copy()
        out[date_col] = pd.to_datetime(out[date_col].astype(str), format='%Y%m%d', errors='coerce')
        out = out.dropna(subset=[date_col]).sort_values(date_col)

        # 统一字段
        rename_map = {
            'vol': 'volume',
        }
        for k, v in rename_map.items():
            if k in out.columns and v not in out.columns:
                out = out.rename(columns={k: v})

        keep_cols = ['open', 'close', 'high', 'low', 'volume', 'amount']
        for c in keep_cols:
            if c not in out.columns:
                out[c] = np.nan

        for c in keep_cols:
            out[c] = pd.to_numeric(out[c], errors='coerce')

        # 清理明显异常的行情行（避免后续校验/训练失败）
        before_rows = len(out)
        bad_price = (out['high'] < out['low']) | (out['close'] < 0)
        bad_volume = out['volume'] < 0
        bad_mask = bad_price | bad_volume
        if bad_mask.any():
            out = out[~bad_mask]
            removed = before_rows - len(out)
            logger.warning(
                f"{qlib_symbol}: 发现并移除异常K线 {removed} 行 (high<low 或 close<0 或 volume<0)"
            )

        out = out[[date_col] + keep_cols]
        out = out.rename(columns={date_col: 'date'})
        out['symbol'] = qlib_symbol
        return out

    def _load_existing(self, qlib_symbol: str) -> Optional[pd.DataFrame]:
        file_path = self.output_dir / qlib_symbol / f"{qlib_symbol}.{self.format}"
        if not file_path.exists():
            return None

        try:
            if self.format == 'csv':
                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True, compression='gzip')
                except Exception:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            elif self.format == 'parquet':
                df = pd.read_parquet(file_path)
            else:
                df = self.qlib_converter.load_qlib_data(qlib_symbol, self.format)
        except Exception as e:
            logger.warning(f"读取本地数据失败，忽略增量逻辑: {file_path} | err={e}")
            return None

        if df is None or df.empty:
            return None

        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df = df.copy()
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date']).set_index('date')
            else:
                df.index = pd.to_datetime(df.index, errors='coerce')

        df = df[~df.index.isna()]
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        return df

    @staticmethod
    def _next_day(date_like: pd.Timestamp) -> str:
        """给定最后一个交易日，返回下一自然日（YYYY-MM-DD）。"""
        dt = pd.to_datetime(date_like)
        return (dt.normalize() + timedelta(days=1)).strftime('%Y-%m-%d')

    def _process_stock(self, ts_code: str) -> int:
        downloaded_bytes = 0
        try:
            qlib_symbol = _to_qlib_symbol(ts_code)

            existing = self._load_existing(qlib_symbol)
            download_start = self.start_date
            need_append = False

            if existing is not None and not existing.empty:
                last_dt = pd.to_datetime(existing.index.max())
                target_end = pd.to_datetime(self.end_date)
                if last_dt.normalize() >= target_end.normalize():
                    self.stats['success_count'] += 1
                    return 0

                # 增量更新：严格从本地最新日期的下一天开始下载
                download_start = self._next_day(last_dt)
                need_append = True

            raw = self.download_stock_daily(ts_code, start_date=download_start, end_date=self.end_date)
            if raw is None or raw.empty:
                if need_append:
                    self.stats['success_count'] += 1
                    return 0
                self.stats['failed_count'] += 1
                self.stats['failed_stocks'].append({'symbol': ts_code, 'error': 'Empty data returned'})
                return 0

            downloaded_bytes = int(raw.memory_usage(deep=True).sum())

            processed = self._preprocess_data(raw, qlib_symbol)
            if processed is None or processed.empty:
                if need_append:
                    self.stats['success_count'] += 1
                    return 0
                self.stats['failed_count'] += 1
                self.stats['failed_stocks'].append({'symbol': ts_code, 'error': 'Preprocess produced empty data'})
                return 0

            new_data = processed.set_index('date')

            if need_append and existing is not None and not existing.empty:
                final_data = pd.concat([existing, new_data])
                final_data = final_data[~final_data.index.duplicated(keep='last')]
                final_data = final_data.sort_index()
            else:
                final_data = new_data

            self.qlib_converter.save_to_qlib_data(final_data, qlib_symbol, self.format)

            self.stats['success_count'] += 1
            self.stats['success_stocks'].append(ts_code)

            # 频率控制
            if self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)

            return downloaded_bytes

        except Exception as e:
            self.stats['failed_count'] += 1
            self.stats['failed_stocks'].append({'symbol': ts_code, 'error': str(e)})
            return 0

    def _process_etf(self, ts_code: str) -> int:
        downloaded_bytes = 0
        try:
            qlib_symbol = _to_qlib_symbol(ts_code)

            existing = self._load_existing(qlib_symbol)
            download_start = self.start_date
            need_append = False

            if existing is not None and not existing.empty:
                last_dt = pd.to_datetime(existing.index.max())
                target_end = pd.to_datetime(self.end_date)
                if last_dt.normalize() >= target_end.normalize():
                    self.stats['success_count'] += 1
                    return 0

                download_start = self._next_day(last_dt)
                need_append = True

            raw = self.download_etf_daily(ts_code, start_date=download_start, end_date=self.end_date)
            if raw is None or raw.empty:
                if need_append:
                    self.stats['success_count'] += 1
                    return 0
                self.stats['failed_count'] += 1
                self.stats['failed_stocks'].append({'symbol': ts_code, 'error': 'Empty ETF data returned'})
                return 0

            downloaded_bytes = int(raw.memory_usage(deep=True).sum())

            processed = self._preprocess_data(raw, qlib_symbol)
            if processed is None or processed.empty:
                if need_append:
                    self.stats['success_count'] += 1
                    return 0
                self.stats['failed_count'] += 1
                self.stats['failed_stocks'].append({'symbol': ts_code, 'error': 'Preprocess produced empty ETF data'})
                return 0

            new_data = processed.set_index('date')

            if need_append and existing is not None and not existing.empty:
                final_data = pd.concat([existing, new_data])
                final_data = final_data[~final_data.index.duplicated(keep='last')]
                final_data = final_data.sort_index()
            else:
                final_data = new_data

            self.qlib_converter.save_to_qlib_data(final_data, qlib_symbol, self.format)

            self.stats['success_count'] += 1
            self.stats['success_stocks'].append(ts_code)

            if self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)

            return downloaded_bytes

        except Exception as e:
            self.stats['failed_count'] += 1
            self.stats['failed_stocks'].append({'symbol': ts_code, 'error': str(e)})
            return 0

    def _process_index(self, ts_code: str) -> int:
        downloaded_bytes = 0
        try:
            qlib_symbol = _to_qlib_symbol(ts_code)

            existing = self._load_existing(qlib_symbol)
            download_start = self.start_date
            need_append = False

            if existing is not None and not existing.empty:
                last_dt = pd.to_datetime(existing.index.max())
                target_end = pd.to_datetime(self.end_date)
                if last_dt.normalize() >= target_end.normalize():
                    self.stats['success_count'] += 1
                    return 0

                download_start = self._next_day(last_dt)
                need_append = True

            raw = self.download_index_daily(ts_code, start_date=download_start, end_date=self.end_date)
            if raw is None or raw.empty:
                if need_append:
                    self.stats['success_count'] += 1
                    return 0
                self.stats['failed_count'] += 1
                self.stats['failed_stocks'].append({'symbol': ts_code, 'error': 'Empty index data returned'})
                return 0

            downloaded_bytes = int(raw.memory_usage(deep=True).sum())

            processed = self._preprocess_data(raw, qlib_symbol)
            if processed is None or processed.empty:
                if need_append:
                    self.stats['success_count'] += 1
                    return 0
                self.stats['failed_count'] += 1
                self.stats['failed_stocks'].append({'symbol': ts_code, 'error': 'Preprocess produced empty index data'})
                return 0

            new_data = processed.set_index('date')

            if need_append and existing is not None and not existing.empty:
                final_data = pd.concat([existing, new_data])
                final_data = final_data[~final_data.index.duplicated(keep='last')]
                final_data = final_data.sort_index()
            else:
                final_data = new_data

            self.qlib_converter.save_to_qlib_data(final_data, qlib_symbol, self.format)

            self.stats['success_count'] += 1
            self.stats['success_stocks'].append(ts_code)

            if self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)

            return downloaded_bytes

        except Exception as e:
            self.stats['failed_count'] += 1
            self.stats['failed_stocks'].append({'symbol': ts_code, 'error': str(e)})
            return 0

    def run(self, limit: int = None):
        self.stats['start_time'] = datetime.now()

        try:
            stock_codes = self.get_all_stocks()
            if limit:
                stock_codes = stock_codes[:limit]
                logger.info(f"测试模式: 限制下载前 {limit} 只股票")

            etf_codes: List[str] = []
            if self.download_etf:
                etf_codes = self.get_all_etfs()
                if limit:
                    etf_codes = etf_codes[:limit]
                    logger.info(f"测试模式: 限制下载前 {limit} 只ETF")
            else:
                logger.info("按参数跳过ETF下载")

            index_codes: List[str] = []
            if self.download_index:
                index_codes = self.get_indices()
            else:
                logger.info("按参数跳过指数下载")

            self.stats['total_stocks'] = len(stock_codes) + len(etf_codes) + len(index_codes)
            logger.info(
                f"待下载合计: {self.stats['total_stocks']} (股票 {len(stock_codes)} + ETF {len(etf_codes)} + 指数 {len(index_codes)})"
            )

            completed = 0
            total_bytes = 0
            start_time = time.time()
            last_time = start_time

            for ts_code in stock_codes:
                downloaded = self._process_stock(ts_code)
                total_bytes += downloaded
                completed += 1

                now = time.time()
                if now - last_time >= 1.0 or completed % 20 == 0:
                    elapsed = now - start_time
                    speed = (total_bytes / 1024 / 1024) / elapsed if elapsed > 0 else 0
                    print(f"\r进度: {completed}/{self.stats['total_stocks']} | 速度: {speed:.2f} MB/s", end='', flush=True)
                    last_time = now

            for ts_code in etf_codes:
                downloaded = self._process_etf(ts_code)
                total_bytes += downloaded
                completed += 1

                now = time.time()
                if now - last_time >= 1.0 or completed % 20 == 0:
                    elapsed = now - start_time
                    speed = (total_bytes / 1024 / 1024) / elapsed if elapsed > 0 else 0
                    print(f"\r进度: {completed}/{self.stats['total_stocks']} | 速度: {speed:.2f} MB/s", end='', flush=True)
                    last_time = now

            for ts_code in index_codes:
                downloaded = self._process_index(ts_code)
                total_bytes += downloaded
                completed += 1

                now = time.time()
                if now - last_time >= 1.0 or completed % 20 == 0:
                    elapsed = now - start_time
                    speed = (total_bytes / 1024 / 1024) / elapsed if elapsed > 0 else 0
                    print(f"\r进度: {completed}/{self.stats['total_stocks']} | 速度: {speed:.2f} MB/s", end='', flush=True)
                    last_time = now

            print()
            self.stats['end_time'] = datetime.now()
            self._generate_summary()

        except Exception as e:
            logger.error(f"主流程异常: {e}")

    def _generate_summary(self):
        duration = self.stats['end_time'] - self.stats['start_time']
        logger.info("=" * 60)
        logger.info("下载完成摘要")
        logger.info(f"总耗时: {duration}")
        logger.info(f"总数: {self.stats['total_stocks']}")
        logger.info(f"成功: {self.stats['success_count']}")
        logger.info(f"失败: {self.stats['failed_count']}")
        logger.info("=" * 60)

        if self.stats['failed_stocks']:
            with open(self.output_dir / "failed_downloads_daily_tushare.json", 'w', encoding='utf-8') as f:
                json.dump(self.stats['failed_stocks'], f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='TuShare A股日频数据下载器')
    parser.add_argument('--output-dir', type=str, default='./data/qlib_data_daily')
    parser.add_argument('--start-date', type=str, help='YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, help='YYYY-MM-DD')
    parser.add_argument('--adjust', type=str, default='qfq', choices=['qfq', 'hfq', ''])
    parser.add_argument('--format', type=str, default='csv', choices=['csv', 'parquet', 'binary'])
    parser.add_argument('--limit', type=int, help='限制下载股票数量(测试用)')
    parser.add_argument('--token', type=str, help='TuShare Token（也可用环境变量 TUSHARE_TOKEN/TUSHARE_PRO_TOKEN）')
    parser.add_argument('--sleep', type=float, default=0.2, help='每只股票下载后 sleep 秒数（用于限频）')
    parser.add_argument('--with-etf', dest='download_etf', action='store_true', default=True, help='下载ETF数据（默认开启）')
    parser.add_argument('--no-etf', dest='download_etf', action='store_false', help='不下载ETF数据')
    parser.add_argument('--with-index', dest='download_index', action='store_true', default=True, help='下载指数数据（默认开启）')
    parser.add_argument('--no-index', dest='download_index', action='store_false', help='不下载指数数据')
    parser.add_argument(
        '--index-codes',
        type=str,
        default='',
        help='指定要下载的指数ts_code（逗号分隔），如 "000300.SH,000905.SH"；为空则默认下载沪深300/中证500/上证50',
    )

    args = parser.parse_args()

    # 允许用户显式传入空字符串（例如 --end-date ""），视为未传
    if isinstance(args.start_date, str) and args.start_date.strip() == "":
        args.start_date = None
    if isinstance(args.end_date, str) and args.end_date.strip() == "":
        args.end_date = None
    if isinstance(args.token, str) and args.token.strip() == "":
        args.token = None

    index_codes = _parse_ts_codes(args.index_codes)

    downloader = TuShareDailyDownloader(
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        adjust=args.adjust,
        format=args.format,
        token=args.token,
        sleep_seconds=args.sleep,
        download_etf=args.download_etf,
        download_index=args.download_index,
        index_codes=index_codes,
    )

    downloader.run(limit=args.limit)


if __name__ == '__main__':
    main()
