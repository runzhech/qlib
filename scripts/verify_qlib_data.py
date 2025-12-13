#!/usr/bin/env python

import argparse
from pathlib import Path
from typing import List

import qlib
from qlib.data import D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Qlib .bin data by reading with qlib.data.D")
    parser.add_argument(
        "--provider_uri",
        default="/home/rainjoe/ws/qlib_data/cn_data_5min",
        help="Qlib data directory (contains calendars/, instruments/, features/)",
    )
    parser.add_argument("--region", default="cn", help="Qlib region passed to qlib.init (e.g. cn/us)")
    parser.add_argument("--freq", default="5min", help="Data frequency used by D.calendar/D.features")
    parser.add_argument(
        "--fields",
        default="$close",
        help="Comma-separated fields, e.g. '$close,$volume' (must be prefixed with $)",
    )
    parser.add_argument("--n", type=int, default=5, help="How many instruments to sample")
    parser.add_argument("--start_time", default=None, help="Optional start time (e.g. '2024-12-02')")
    parser.add_argument("--end_time", default=None, help="Optional end time (e.g. '2025-12-11')")
    return parser.parse_args()


def split_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    args = parse_args()

    provider_uri = args.provider_uri
    fields = split_csv(args.fields)

    # Validate path early to avoid confusing qlib errors.
    if isinstance(provider_uri, str):
        provider_path = Path(provider_uri).expanduser()
        if not provider_path.exists():
            raise FileNotFoundError(
                f"provider_uri not found: {provider_path}\n"
                f"Run dump first, e.g.\n"
                f"  python scripts/dump_bin.py dump_all --data_path ... --qlib_dir {provider_path} --freq {args.freq} --file_suffix .parquet"
            )
        provider_uri = str(provider_path)

    # For non-daily data, qlib expects provider_uri to include a freq key.
    # If user passes a plain path string, map it to the requested freq.
    if isinstance(provider_uri, str) and args.freq and args.freq != "day":
        provider_uri = {"__DEFAULT_FREQ": provider_uri, args.freq: provider_uri}

    qlib.init(provider_uri=provider_uri, region=args.region, expression_cache=None, dataset_cache=None)

    cal = D.calendar(freq=args.freq)
    print(f"calendar[{args.freq}] size={len(cal)} range=({cal[0]} -> {cal[-1]})")

    inst = D.list_instruments(D.instruments("all"), freq=args.freq, as_list=True)
    print(f"instruments[{args.freq}] count={len(inst)}")
    if not inst:
        raise RuntimeError("No instruments found. Check instruments/all.txt and freq.")

    sample = inst[: max(1, args.n)]
    print("sample instruments:", sample)

    df = D.features(sample, fields, freq=args.freq, start_time=args.start_time, end_time=args.end_time)
    print("features shape:", df.shape)
    print(df.head(10))


if __name__ == "__main__":
    main()
