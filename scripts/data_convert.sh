#!/usr/bin/env bash

set -euo pipefail


# convert to qlib bin
python /home/rainjoe/ws/qlibC/qlib/scripts/dump_bin.py dump_all \
  --data_path /home/rainjoe/ws/quant_project/data/qlib_data_min \
  --qlib_dir  /home/rainjoe/ws/qlib_data/cn_data_5min \
  --freq 5min \
  --file_suffix .parquet \
  --exclude_fields date,symbol

# verify
python /home/rainjoe/ws/qlibC/qlib/scripts/verify_qlib_data.py \
  --provider_uri /home/rainjoe/ws/qlib_data/cn_data_5min \
  --freq 5min \
  --fields '$close' \
  --n 3
