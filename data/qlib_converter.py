"""
Qlib数据格式转换模块
实现股票数据转换为Qlib格式，支持Alpha158因子准备
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 尝试导入qlib相关模块
try:
    import qlib
    from qlib.data import D
    from qlib.data.data import CalendarProvider
    from qlib.data import features
    from qlib.config import REG_CN
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    logging.warning("Qlib not available. Install with: pip install qlib")

from datetime import datetime, timedelta
import pickle


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_stock_data(df: pd.DataFrame) -> bool:
        """验证股票数据格式"""
        required_columns = ['open', 'close', 'high', 'low', 'volume']
        
        # 检查必需列
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"缺少必需列: {missing_columns}")
            return False
        
        # 检查数据类型
        if not all(df[col].dtype in ['float64', 'int64'] for col in required_columns):
            logging.error("数据列必须是数值类型")
            return False
        
        # 检查缺失值
        missing_ratio = df[required_columns].isnull().sum() / len(df)
        if (missing_ratio > 0.1).any():
            logging.warning(f"部分列缺失值比例较高: {missing_ratio[missing_ratio > 0.1].to_dict()}")
        
        # 检查价格数据合理性
        if (df['high'] < df['low']).any():
            logging.error("存在high < low的数据")
            return False
        
        if (df['close'] < 0).any() or (df['volume'] < 0).any():
            logging.error("存在负数价格或成交量")
            return False
        
        return True
    
    @staticmethod
    def validate_index(df: pd.DataFrame, index_name: str = 'date') -> bool:
        """验证索引格式"""
        if index_name not in df.index.names:
            logging.error(f"缺少索引: {index_name}")
            return False
        
        # 检查索引类型和排序
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            logging.error(f"索引必须是datetime类型")
            return False
        
        if not df.index.is_monotonic_increasing:
            logging.warning("索引未排序，已自动排序")
            df.sort_index(inplace=True)
        
        return True


class Alpha158FactorCalculator:
    """Alpha158因子计算器"""
    
    def __init__(self):
        self.factor_groups = {
            'price_momentum': [
                'return(close, 1)', 'return(close, 3)', 'return(close, 5)',
                'return(close, 10)', 'return(close, 20)', 'return(close, 60)',
                'return(close, 120)', 'return(close, 250)'
            ],
            'volatility': [
                'volatility(return(close, 1), 5)', 'volatility(return(close, 1), 10)',
                'volatility(return(close, 1), 20)', 'volatility(return(close, 1), 30)'
            ],
            'volume': [
                'return(volume, 1)', 'return(volume, 3)', 'return(volume, 5)',
                'volatility(return(volume, 1), 10)', 'volume_change_ratio'
            ],
            'price_relative': [
                'return(high, 1)', 'return(low, 1)', 'return(open, 1)',
                'close/open', 'high/low', 'return(close/open, 1)'
            ]
        }
    
    def calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算Alpha158因子"""
        factors_df = pd.DataFrame(index=df.index)
        
        # 价格动量因子
        for factor in self.factor_groups['price_momentum']:
            factors_df[factor] = self._calculate_return(df['close'], factor)
        
        # 波动率因子
        for factor in self.factor_groups['volatility']:
            period = self._extract_period_from_factor(factor)
            factors_df[factor] = self._calculate_volatility(df['close'], period)
        
        # 成交量因子
        for factor in self.factor_groups['volume']:
            if 'return(volume' in factor:
                period = int(factor.split(',')[1].split(')')[0])
                factors_df[factor] = self._calculate_return(df['volume'], factor, period)
            elif 'volatility' in factor:
                factors_df[factor] = self._calculate_volatility(df['volume'], 10)
            else:
                factors_df['volume_change_ratio'] = df['volume'].pct_change()
        
        # 价格相对因子
        factors_df['close/open'] = df['close'] / df['open']
        factors_df['high/low'] = df['high'] / df['low']
        factors_df['return(close/open, 1)'] = (df['close'] / df['open']).pct_change()
        factors_df['return(high, 1)'] = df['high'].pct_change()
        factors_df['return(low, 1)'] = df['low'].pct_change()
        factors_df['return(open, 1)'] = df['open'].pct_change()
        
        return factors_df
    
    def _calculate_return(self, series: pd.Series, factor_name: str, period: int = 1) -> pd.Series:
        """计算收益率"""
        if 'return(close' in factor_name:
            return series.pct_change(period)
        elif 'return(volume' in factor_name:
            return series.pct_change(period)
        else:
            return series.pct_change(period)
    
    def _calculate_volatility(self, series: pd.Series, window: int) -> pd.Series:
        """计算波动率"""
        returns = series.pct_change()
        return returns.rolling(window).std()
    
    def _extract_period_from_factor(self, factor_name: str) -> int:
        """从因子名称中提取周期参数"""
        try:
            # 查找最后一个括号内的内容
            if '(' in factor_name and ')' in factor_name:
                # 找到最后一个开括号的位置
                last_open_paren = factor_name.rfind('(')
                # 提取括号内的内容
                content = factor_name[last_open_paren+1:factor_name.rfind(')')]
                # 如果包含逗号，取第一个数字
                if ',' in content:
                    period_str = content.split(',')[0].strip()
                else:
                    period_str = content.strip()
                return int(period_str)
            return 1
        except (ValueError, IndexError):
            return 1


class QlibConverter:
    """Qlib数据格式转换器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化转换器
        
        Args:
            config: 配置字典
        """
        self.config = self._default_config()
        if config:
            self.config.update(config)
            
        self.logger = logging.getLogger(__name__)
        self.validator = DataValidator()
        self.factor_calculator = Alpha158FactorCalculator()
        
        # qlib路径配置
        self.qlib_data_path = Path(self.config.get('qlib_data_path', './data/qlib_data'))
        self.qlib_cache_path = Path(self.config.get('qlib_cache_path', './data/cache'))
        
        # 创建必要目录
        self.qlib_data_path.mkdir(parents=True, exist_ok=True)
        self.qlib_cache_path.mkdir(parents=True, exist_ok=True)
        
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'qlib_data_path': './data/qlib_data',
            'qlib_cache_path': './data/cache',
            'date_format': '%Y-%m-%d',
            'datetime_format': '%Y-%m-%d %H:%M:%S',
            'index_column': 'date',
            'columns_mapping': {
                'open': 'open',
                'close': 'close',
                'high': 'high',
                'low': 'low',
                'volume': 'volume',
                'amount': 'amount'
            }
        }
    
    def _init_qlib(self):
        """初始化qlib"""
        try:
            qlib.init(
                base_dir=str(self.qlib_data_path),
                region=REG_CN
            )
            self.logger.info(f"Qlib初始化成功，数据路径: {self.qlib_data_path}")
        except Exception as e:
            self.logger.error(f"Qlib初始化失败: {e}")
    
    def convert_to_qlib_format(self, 
                              data: Union[pd.DataFrame, str], 
                              symbol: str,
                              columns_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        转换为Qlib格式
        
        Args:
            data: 原始数据（DataFrame或文件路径）
            symbol: 股票代码
            columns_mapping: 列映射字典
            
        Returns:
            转换后的DataFrame
        """
        # 加载数据
        if isinstance(data, str):
            df = self._load_data_from_file(data)
        else:
            df = data.copy()
        
        # 列映射
        if columns_mapping:
            df = self._map_columns(df, columns_mapping)
        else:
            df = self._map_columns(df, self.config['columns_mapping'])
        
        # 验证数据
        if not self.validator.validate_stock_data(df):
            raise ValueError("数据验证失败")
        
        # 确保日期索引
        df = self._ensure_datetime_index(df)
        
        # 添加股票代码
        if 'symbol' not in df.columns and 'instrument' not in df.columns:
            df['symbol'] = symbol
            df['instrument'] = symbol
        
        # 计算Alpha158因子
        alpha_factors = self.factor_calculator.calculate_factors(df)
        
        # 合并原始数据和因子数据
        qlib_df = pd.concat([df, alpha_factors], axis=1)
        
        # 添加必要的qlib列
        qlib_df = self._add_qlib_columns(qlib_df)
        
        self.logger.info(f"成功转换数据格式，股票代码: {symbol}, 数据形状: {qlib_df.shape}")
        return qlib_df
    
    def save_to_qlib_data(self, 
                          data: pd.DataFrame, 
                          symbol: str,
                          format: str = 'binary',
                          compression: str = 'gzip') -> str:
        """
        保存数据到qlib数据目录
        
        Args:
            data: 股票数据
            symbol: 股票代码
            format: 保存格式 ('binary', 'csv', 'parquet')
            compression: 压缩方式
            
        Returns:
            保存的文件路径
        """
        symbol_path = self.qlib_data_path / symbol
        symbol_path.mkdir(exist_ok=True)
        
        filename = f"{symbol}.{format}"
        file_path = symbol_path / filename
        
        try:
            if format == 'csv':
                data.to_csv(file_path, index=True, compression=compression if compression != 'none' else None)
            elif format == 'parquet':
                data.to_parquet(file_path, compression=compression)
            else:  # binary
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.logger.info(f"数据保存成功: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"保存数据失败: {e}")
            raise
    
    def load_qlib_data(self, symbol: str, format: str = 'binary') -> Optional[pd.DataFrame]:
        """
        加载qlib数据
        
        Args:
            symbol: 股票代码
            format: 数据格式
            
        Returns:
            加载的数据DataFrame
        """
        symbol_path = self.qlib_data_path / symbol
        
        # 尝试不同格式
        formats_to_try = [format]
        if format == 'binary':
            formats_to_try = ['binary', 'csv', 'parquet']
        
        for fmt in formats_to_try:
            file_path = symbol_path / f"{symbol}.{fmt}"
            if file_path.exists():
                try:
                    if fmt == 'csv':
                        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    elif fmt == 'parquet':
                        data = pd.read_parquet(file_path)
                    else:  # binary
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                    
                    self.logger.info(f"数据加载成功: {file_path}")
                    return data
                    
                except Exception as e:
                    self.logger.warning(f"格式{fmt}加载失败: {e}")
                    continue
        
        self.logger.warning(f"未找到股票{symbol}的数据文件")
        return None
    
    def create_qlib_dataset(self, 
                           symbols: List[str],
                           start_date: str,
                           end_date: str,
                           format: str = 'binary') -> Dict[str, Union[str, pd.DataFrame]]:
        """
        创建qlib数据集
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            format: 数据格式
            
        Returns:
            数据集信息字典
        """
        dataset_info = {
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'data_path': str(self.qlib_data_path),
            'created_at': datetime.now().isoformat(),
            'data_summary': {}
        }
        
        valid_symbols = []
        
        for symbol in symbols:
            data = self.load_qlib_data(symbol, format)
            if data is not None:
                # 过滤日期范围
                mask = (data.index >= start_date) & (data.index <= end_date)
                filtered_data = data.loc[mask]
                
                if len(filtered_data) > 0:
                    valid_symbols.append(symbol)
                    dataset_info['data_summary'][symbol] = {
                        'rows': len(filtered_data),
                        'columns': len(filtered_data.columns),
                        'start_date': filtered_data.index.min().strftime('%Y-%m-%d'),
                        'end_date': filtered_data.index.max().strftime('%Y-%m-%d')
                    }
        
        dataset_info['valid_symbols'] = valid_symbols
        dataset_info['total_symbols'] = len(valid_symbols)
        
        self.logger.info(f"数据集创建完成，包含{len(valid_symbols)}只股票")
        return dataset_info
    
    def _load_data_from_file(self, file_path: str) -> pd.DataFrame:
        """从文件加载数据"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
    
    def _map_columns(self, df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        """映射列名"""
        df_mapped = df.copy()
        rename_dict = {}
        
        for orig_col, new_col in mapping.items():
            if orig_col in df_mapped.columns:
                rename_dict[orig_col] = new_col
        
        if rename_dict:
            df_mapped.rename(columns=rename_dict, inplace=True)
        
        return df_mapped
    
    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保日期时间索引"""
        df_copy = df.copy()
        
        # 检查是否有日期列
        date_columns = [col for col in df_copy.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if date_columns:
            date_col = date_columns[0]
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            df_copy.set_index(date_col, inplace=True)
        elif not isinstance(df_copy.index, pd.DatetimeIndex):
            # 尝试转换索引
            try:
                df_copy.index = pd.to_datetime(df_copy.index)
            except Exception as e:
                self.logger.warning(f"无法转换索引为datetime: {e}")
        
        # 排序索引
        df_copy.sort_index(inplace=True)
        
        return df_copy
    
    def _add_qlib_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加qlib必需的列"""
        df_copy = df.copy()
        
        # 添加money列（如果没有）
        if 'money' not in df_copy.columns and 'amount' in df_copy.columns:
            df_copy['money'] = df_copy['amount']
        elif 'money' not in df_copy.columns:
            # 计算money = close * volume
            if 'close' in df_copy.columns and 'volume' in df_copy.columns:
                df_copy['money'] = df_copy['close'] * df_copy['volume']
        
        # 添加factor列（为空，qlib会自动计算）
        df_copy['factor'] = 1.0
        
        return df_copy
    
    def get_data_statistics(self, symbol: str) -> Dict:
        """获取数据统计信息"""
        data = self.load_qlib_data(symbol)
        
        if data is None:
            return {}
        
        stats = {
            'symbol': symbol,
            'total_rows': len(data),
            'columns': list(data.columns),
            'date_range': {
                'start': data.index.min().strftime('%Y-%m-%d') if len(data) > 0 else None,
                'end': data.index.max().strftime('%Y-%m-%d') if len(data) > 0 else None
            },
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict()
        }
        
        # 添加数值列的统计信息
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric_stats'] = data[numeric_cols].describe().to_dict()
        
        return stats
    
    def batch_convert(self, 
                     data_files: Dict[str, str],
                     output_format: str = 'binary') -> Dict[str, str]:
        """
        批量转换数据文件
        
        Args:
            data_files: {symbol: file_path} 字典
            output_format: 输出格式
            
        Returns:
            转换结果字典 {symbol: output_path}
        """
        results = {}
        
        for symbol, file_path in data_files.items():
            try:
                # 转换数据
                qlib_data = self.convert_to_qlib_format(file_path, symbol)
                
                # 保存数据
                output_path = self.save_to_qlib_data(qlib_data, symbol, output_format)
                results[symbol] = output_path
                
                self.logger.info(f"成功转换并保存: {symbol}")
                
            except Exception as e:
                self.logger.error(f"转换失败 {symbol}: {e}")
                continue
        
        return results


# 使用示例和测试函数
def example_usage():
    """使用示例"""
    # 创建转换器实例
    converter = QlibConverter()
    
    # 示例数据
    sample_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', '2020-12-31', freq='D'),
        'open': np.random.randn(366).cumsum() + 100,
        'high': np.random.randn(366).cumsum() + 105,
        'low': np.random.randn(366).cumsum() + 95,
        'close': np.random.randn(366).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 366),
        'amount': np.random.randint(100000000, 1000000000, 366)
    })
    
    # 转换数据格式
    qlib_data = converter.convert_to_qlib_format(sample_data, '000001')
    
    # 保存数据
    output_path = converter.save_to_qlib_data(qlib_data, '000001', 'binary')
    
    # 加载数据
    loaded_data = converter.load_qlib_data('000001')
    
    # 创建数据集
    dataset_info = converter.create_qlib_dataset(['000001'], '2020-01-01', '2020-12-31')
    
    print("转换完成!")
    print(f"输出路径: {output_path}")
    print(f"数据集信息: {dataset_info}")


# if __name__ == "__main__":
    # example_usage()