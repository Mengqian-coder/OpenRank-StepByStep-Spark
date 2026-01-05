# src/metrics/base_calculator.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any


class BaseMetricCalculator(ABC):
    """所有指标计算器的基类"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metric_name = self.__class__.__name__

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算指标的核心方法"""
        pass

    def normalize_score(self, series: pd.Series, method='minmax') -> pd.Series:
        """标准化分数到0-100范围"""
        if method == 'minmax':
            if series.max() == series.min():
                return pd.Series(50, index=series.index)
            return 100 * (series - series.min()) / (series.max() - series.min())
        elif method == 'log_norm':
            # 对数归一化，处理长尾分布
            log_series = np.log1p(series)
            return self.normalize_score(log_series, 'minmax')
        return series