# src/metrics/community_vitality.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .base_calculator import BaseMetricCalculator


class CommunityVitalityCalculator(BaseMetricCalculator):
    """社区活力指数计算器"""

    def __init__(self, config=None):
        super().__init__(config)
        # 默认权重配置
        self.weights = {
            'continuous_activity': 0.40,
            'contributor_health': 0.35,
            'interaction_quality': 0.25
        }

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算社区活力指数的三个子指标"""

        print(f"开始计算社区活力指数...")

        # 1. 持续贡献活跃度
        df['cca_score'] = self._calculate_continuous_activity(df)

        # 2. 贡献者生态系统健康度
        df['ceh_score'] = self._calculate_contributor_health(df)

        # 3. 社区互动质量（简化版，需要issue数据）
        df['ciq_score'] = self._calculate_interaction_quality(df)

        # 综合得分
        df['community_vitality_score'] = (
                self.weights['continuous_activity'] * df['cca_score'] +
                self.weights['contributor_health'] * df['ceh_score'] +
                self.weights['interaction_quality'] * df['ciq_score']
        )

        print(f"社区活力指数计算完成，平均分: {df['community_vitality_score'].mean():.2f}")
        return df

    def _calculate_continuous_activity(self, df: pd.DataFrame) -> pd.Series:
        """计算持续贡献活跃度"""
        # 基于现有数据近似计算
        # 实际中需要获取commit历史，这里用pushed_at近似

        # 计算距离上次更新的天数（已有时区处理）
        current_time_utc = pd.Timestamp.now(tz='UTC')
        days_since_update = (current_time_utc - df['pushed_at']).dt.days

        # 转换活跃度分数（最近更新得分高）
        # 使用指数衰减函数：score = 100 * exp(-λ * days)
        # λ=0.1表示每10天衰减63%
        activity_score = 100 * np.exp(-0.1 * days_since_update.clip(0, 30))

        # 结合star和fork数（对数归一化）
        star_score = self.normalize_score(np.log1p(df['stargazers_count']), 'minmax')
        fork_score = self.normalize_score(np.log1p(df['forks_count']), 'minmax')

        # 综合得分
        cca_score = 0.6 * activity_score + 0.25 * star_score + 0.15 * fork_score

        return self.normalize_score(cca_score)

    def _calculate_contributor_health(self, df: pd.DataFrame) -> pd.Series:
        """计算贡献者健康度（简化版）"""
        # 注意：我们当前数据缺少详细贡献者信息
        # 需要后续通过API补充

        # 临时使用fork_count作为贡献者多样性的代理指标
        # fork数越高，可能表示更多人关注和参与

        fork_proxy = np.log1p(df['forks_count'])

        # 结合open_issues_count（issues多可能表示社区参与度高）
        issues_proxy = np.log1p(df['open_issues_count'] + 1)

        # 简单加权
        health_score = 0.7 * self.normalize_score(fork_proxy) + \
                       0.3 * self.normalize_score(issues_proxy)

        return health_score

    def _calculate_interaction_quality(self, df: pd.DataFrame) -> pd.Series:
        """计算社区互动质量（占位实现）"""
        # 实际需要获取issue/PR的响应时间和解决率
        # 这里先用简化版本

        # 基于是否有description和topics推断文档质量
        has_description = df['description'].str.len() > 10
        has_topics = df['topics'].str.len() > 2

        interaction_score = (
                has_description.astype(int) * 60 +
                has_topics.astype(int) * 40
        )

        return pd.Series(interaction_score, index=df.index)