from loguru import logger
import os
from typing import Optional

import fire
import numpy as np
import pandas as pd
import qlib
from tqdm import tqdm

from qlib.data import D


class DataHealthChecker:
    """Checks a dataset for data completeness and correctness. The data will be converted to a pd.DataFrame and checked for the following problems:
    - any of the columns ["open", "high", "low", "close", "volume"] are missing
    - any data is missing
    - any step change in the OHLCV columns is above a threshold (default: 0.5 for price, 3 for volume)
    - any factor is missing
    """

    def __init__(
        self,
        csv_path=None,
        qlib_dir=None,
        freq="day",
        instruments="all",
        start_time=None,
        end_time=None,
        include_factor=False,
        large_step_threshold_price=0.5,
        large_step_threshold_volume=3,
        missing_data_num=0,
    ):
        assert csv_path or qlib_dir, "One of csv_path or qlib_dir should be provided."
        assert not (csv_path and qlib_dir), "Only one of csv_path or qlib_dir should be provided."

        self.data = {}
        self.problems = {}
        self.freq = freq
        self.instruments = instruments
        self.start_time = start_time
        self.end_time = end_time
        self.include_factor = include_factor
        self.large_step_threshold_price = large_step_threshold_price
        self.large_step_threshold_volume = large_step_threshold_volume
        self.missing_data_num = missing_data_num

        if csv_path:
            assert os.path.isdir(csv_path), f"{csv_path} should be a directory."
            files = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
            for filename in tqdm(files, desc="Loading data"):
                df = pd.read_csv(os.path.join(csv_path, filename))
                self.data[filename] = df

        elif qlib_dir:
            # Prefer mapping provider_uri with explicit freq; this is required for non-day data like 5min.
            # Also makes behavior consistent across day/1d datasets.
            provider_uri = {self.freq: qlib_dir}
            # daily datasets sometimes store calendars as 1d.txt instead of day.txt
            try:
                cal_dir = os.path.join(qlib_dir, "calendars")
                if self.freq == "day" and not os.path.exists(os.path.join(cal_dir, "day.txt")) and os.path.exists(
                    os.path.join(cal_dir, "1d.txt")
                ):
                    provider_uri = {"1d": qlib_dir}
                    self.freq = "1d"
                elif self.freq == "1d" and not os.path.exists(os.path.join(cal_dir, "1d.txt")) and os.path.exists(
                    os.path.join(cal_dir, "day.txt")
                ):
                    provider_uri = {"day": qlib_dir}
                    self.freq = "day"
            except Exception:
                pass

            qlib.init(provider_uri=provider_uri)
            self.load_qlib_data()

    def load_qlib_data(self):
        inst_cfg = D.instruments(market="all")
        all_instruments = D.list_instruments(
            instruments=inst_cfg, as_list=True, freq=self.freq, start_time=self.start_time, end_time=self.end_time
        )

        if isinstance(self.instruments, str):
            if self.instruments.lower() == "all":
                instrument_list = all_instruments
            else:
                instrument_list = [x.strip() for x in self.instruments.split(",") if x.strip()]
        else:
            instrument_list = list(self.instruments)

        required_fields = ["$open", "$close", "$low", "$high", "$volume"]
        if self.include_factor:
            required_fields.append("$factor")

        df = D.features(
            instrument_list,
            required_fields,
            start_time=self.start_time,
            end_time=self.end_time,
            freq=self.freq,
        )

        rename_map = {
            "$open": "open",
            "$close": "close",
            "$low": "low",
            "$high": "high",
            "$volume": "volume",
            "$factor": "factor",
        }
        df = df.rename(columns=rename_map)

        # Split by instrument for downstream checks.
        if isinstance(df.index, pd.MultiIndex) and "instrument" in df.index.names:
            for inst, inst_df in df.groupby(level="instrument", sort=False):
                if isinstance(inst_df.index, pd.MultiIndex):
                    inst_df = inst_df.droplevel("instrument")
                self.data[inst] = inst_df
        else:
            # fallback
            self.data["all"] = df

    def check_missing_data(self) -> Optional[pd.DataFrame]:
        """Check if any data is missing in the DataFrame."""
        result_dict = {
            "instruments": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }
        for filename, df in self.data.items():
            missing_data_columns = df.isnull().sum()[df.isnull().sum() > self.missing_data_num].index.tolist()
            if len(missing_data_columns) > 0:
                result_dict["instruments"].append(filename)
                result_dict["open"].append(df.isnull().sum()["open"])
                result_dict["high"].append(df.isnull().sum()["high"])
                result_dict["low"].append(df.isnull().sum()["low"])
                result_dict["close"].append(df.isnull().sum()["close"])
                result_dict["volume"].append(df.isnull().sum()["volume"])

        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            return result_df
        else:
            logger.info(f"✅ There are no missing data.")
            return None

    def check_large_step_changes(self) -> Optional[pd.DataFrame]:
        """Check if there are any large step changes above the threshold in the OHLCV columns."""
        result_dict = {
            "instruments": [],
            "col_name": [],
            "date": [],
            "pct_change": [],
        }
        for filename, df in self.data.items():
            affected_columns = []
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce")
                    prev = s.shift(1)
                    # use safe pct change; avoid +/-inf when prev==0
                    pct_change = (s - prev).abs() / prev.abs().replace(0, np.nan)
                    pct_change = pct_change.replace([np.inf, -np.inf], np.nan)
                    threshold = self.large_step_threshold_volume if col == "volume" else self.large_step_threshold_price
                    max_change = pct_change.max(skipna=True)
                    if pd.notna(max_change) and max_change > threshold:
                        result_dict["instruments"].append(filename)
                        result_dict["col_name"].append(col)
                        max_idx = pct_change.idxmax()
                        if isinstance(max_idx, tuple) and len(max_idx) >= 2:
                            dt = max_idx[1]
                        else:
                            dt = max_idx
                        dt = pd.Timestamp(dt)
                        result_dict["date"].append(dt.strftime("%Y-%m-%d"))
                        result_dict["pct_change"].append(float(max_change))
                        affected_columns.append(col)

        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            return result_df
        else:
            logger.info(f"✅ There are no large step changes in the OHLCV column above the threshold.")
            return None

    def check_required_columns(self) -> Optional[pd.DataFrame]:
        """Check if any of the required columns (OLHCV) are missing in the DataFrame."""
        required_columns = ["open", "high", "low", "close", "volume"]
        result_dict = {
            "instruments": [],
            "missing_col": [],
        }
        for filename, df in self.data.items():
            if not all(column in df.columns for column in required_columns):
                missing_required_columns = [column for column in required_columns if column not in df.columns]
                result_dict["instruments"].append(filename)
                result_dict["missing_col"] += missing_required_columns

        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            return result_df
        else:
            logger.info(f"✅ The columns (OLHCV) are complete and not missing.")
            return None

    def check_missing_factor(self) -> Optional[pd.DataFrame]:
        """Check if the 'factor' column is missing in the DataFrame."""
        if not self.include_factor:
            logger.info("Skip factor check (include_factor=False).")
            return None

        result_dict = {
            "instruments": [],
            "missing_factor_col": [],
            "missing_factor_data": [],
        }
        for filename, df in self.data.items():
            if "000300" in filename or "000903" in filename or "000905" in filename:
                continue
            if "factor" not in df.columns:
                result_dict["instruments"].append(filename)
                result_dict["missing_factor_col"].append(True)
                # no factor column; can't check emptiness
                continue
            if df["factor"].isnull().all():
                if filename in result_dict["instruments"]:
                    result_dict["missing_factor_data"].append(True)
                else:
                    result_dict["instruments"].append(filename)
                    result_dict["missing_factor_col"].append(False)
                    result_dict["missing_factor_data"].append(True)

        result_df = pd.DataFrame(result_dict).set_index("instruments")
        if not result_df.empty:
            return result_df
        else:
            logger.info(f"✅ The `factor` column already exists and is not empty.")
            return None

    def check_data(self):
        check_missing_data_result = self.check_missing_data()
        check_large_step_changes_result = self.check_large_step_changes()
        check_required_columns_result = self.check_required_columns()
        check_missing_factor_result = self.check_missing_factor()
        if (
            check_large_step_changes_result is not None
            or check_large_step_changes_result is not None
            or check_required_columns_result is not None
            or check_missing_factor_result is not None
        ):
            print(f"\nSummary of data health check ({len(self.data)} files checked):")
            print("-------------------------------------------------")
            if isinstance(check_missing_data_result, pd.DataFrame):
                logger.warning(f"There is missing data.")
                print(check_missing_data_result)
            if isinstance(check_large_step_changes_result, pd.DataFrame):
                logger.warning(f"The OHLCV column has large step changes.")
                print(check_large_step_changes_result)
            if isinstance(check_required_columns_result, pd.DataFrame):
                logger.warning(f"Columns (OLHCV) are missing.")
                print(check_required_columns_result)
            if isinstance(check_missing_factor_result, pd.DataFrame):
                logger.warning(f"The factor column does not exist or is empty")
                print(check_missing_factor_result)


if __name__ == "__main__":
    fire.Fire(DataHealthChecker)
