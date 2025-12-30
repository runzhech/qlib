#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取并打印 Parquet 格式数据脚本
功能：读取指定路径的 Parquet 文件并打印数据预览和信息
"""

import argparse
import pandas as pd
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='读取并打印 Parquet 格式数据')
    
    # 互斥组：可以直接提供文件路径，或者提供目录+股票代码
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('file_path', nargs='?', help='Parquet 文件路径')
    group.add_argument('--symbol', help='股票代码 (例如: sh.600519)')
    
    parser.add_argument('--dir', default='./data/qlib_data_min', help='数据目录 (配合 --symbol 使用)')
    parser.add_argument('--head', type=int, default=10, help='显示前 N 行')
    parser.add_argument('--tail', type=int, default=10, help='显示后 N 行')
    parser.add_argument('--no-info', action='store_true', help='不显示 DataFrame info')
    
    args = parser.parse_args()
    
    file_path = None
    if args.file_path:
        file_path = Path(args.file_path)
    elif args.symbol:
        # 尝试构建路径，通常是 dir/symbol.parquet
        base_dir = Path(args.dir)
        # 尝试几种常见的文件名模式
        candidates = [
            base_dir / args.symbol / f"{args.symbol}.parquet", # Baostock downloader 默认格式
            base_dir / f"{args.symbol}.parquet",
            base_dir / args.symbol / "data.parquet", # Qlib 风格
        ]
        
        for p in candidates:
            if p.exists():
                file_path = p
                break
        
        if file_path is None:
            print(f"错误: 在 {base_dir} 下未找到 {args.symbol} 的 parquet 文件")
            print(f"尝试过的路径: {[str(p) for p in candidates]}")
            sys.exit(1)

    if not file_path.exists():
        print(f"错误: 文件不存在: {file_path}")
        sys.exit(1)
        
    print(f"正在读取文件: {file_path}")
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"读取失败: {e}")
        sys.exit(1)
        
    print("-" * 50)
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print("-" * 50)
    
    if not args.no_info:
        print("数据信息:")
        df.info()
        print("-" * 50)
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", None,
        "display.max_colwidth", None,
    ):
        print(f"前 {args.head} 行:")
        print(df.head(args.head))
        
        print("-" * 50)
        print(f"后 {args.tail} 行:")
        print(df.tail(args.tail))

if __name__ == "__main__":
    main()
