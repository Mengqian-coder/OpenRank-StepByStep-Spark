#!/usr/bin/env python3
"""
外部吸引力维度计算器
包含三个子维度：增长势头(GR)、可见性(VI)、网络效应(NET)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging


class ExternalAppealCalculator:
    """外部吸引力计算器"""

    def __init__(self, config=None):
        """
        初始化计算器

        Args:
            config: 配置字典，可自定义权重和阈值
        """
        self.config = config or self.get_default_config()
        self.logger = logging.getLogger(__name__)

        # 子维度权重配置
        self.weights = self.config.get('weights', {
            'growth_momentum': 0.40,
            'visibility': 0.35,
            'network_effect': 0.25
        })

        # 阈值配置
        self.thresholds = self.config.get('thresholds', {
            'star_growth_threshold': 0.1,  # 日均星标增长率阈值
            'fork_ratio_threshold': 0.2,  # fork比例阈值
            'contributor_threshold': 3,  # 贡献者数量阈值
            'activity_threshold': 5  # 活动数量阈值
        })

    def get_default_config(self):
        """获取默认配置"""
        return {
            'weights': {
                'growth_momentum': 0.40,
                'visibility': 0.35,
                'network_effect': 0.25
            },
            'thresholds': {
                'star_growth_threshold': 0.1,
                'fork_ratio_threshold': 0.2,
                'contributor_threshold': 3,
                'activity_threshold': 5
            },
            'enable_logging': True
        }

    def calculate(self, df):
        """计算外部吸引力分数"""
        self.logger.info("开始计算外部吸引力维度评分...")

        result_df = df.copy()

        # 1. 增长势头 (Growth Momentum)
        self.logger.info("计算增长势头...")
        result_df['gr_score'] = self._calculate_growth_momentum(result_df)

        # 2. 可见性 (Visibility)
        self.logger.info("计算可见性...")
        result_df['vi_score'] = self._calculate_visibility(result_df)

        # 3. 网络效应 (Network Effect)
        self.logger.info("计算网络效应...")
        result_df['net_score'] = self._calculate_network_effect(result_df)

        # 4. 综合分数
        self.logger.info("计算综合外部吸引力分数...")
        result_df = self._calculate_external_appeal_score(result_df)

        # 5. 添加等级标签
        result_df = self._add_appeal_level(result_df)

        self.logger.info(f"外部吸引力计算完成，共处理 {len(result_df)} 个项目")
        return result_df

    def _calculate_growth_momentum(self, df):
        """计算增长势头"""
        scores = []

        for idx, row in df.iterrows():
            score = 0.0

            # 1. 基于项目年龄的星标增长率
            if 'created_at' in df.columns and 'stargazers_count' in df.columns:
                try:
                    created_date = pd.to_datetime(row['created_at'])
                    days_old = max(1, (pd.Timestamp.now() - created_date).days)
                    stars = float(row['stargazers_count'])

                    # 日均星标增长率
                    daily_star_rate = stars / days_old

                    if daily_star_rate > 1.0:
                        score = 0.9
                    elif daily_star_rate > 0.5:
                        score = 0.7
                    elif daily_star_rate > 0.1:
                        score = 0.5
                    elif daily_star_rate > 0.01:
                        score = 0.3
                    else:
                        score = 0.1

                    # 记录增长率（用于后续分析）
                    df.at[idx, 'star_growth_rate'] = daily_star_rate
                except Exception as e:
                    self.logger.debug(f"计算星标增长率失败: {e}")
                    score = 0.3
            else:
                score = 0.3

            # 2. 基于fork增长
            if 'forks_count' in df.columns and 'stargazers_count' in df.columns:
                forks = float(row.get('forks_count', 0))
                stars = float(row.get('stargazers_count', 1))
                fork_ratio = forks / max(stars, 1)

                if fork_ratio > 0.5:
                    score = min(1.0, score + 0.2)
                elif fork_ratio > 0.2:
                    score = min(1.0, score + 0.1)

                # 记录fork比例
                df.at[idx, 'fork_ratio'] = fork_ratio

            # 3. 基于最近活动的增长趋势（如果有pushed_at）
            if 'pushed_at' in df.columns:
                try:
                    pushed_date = pd.to_datetime(row['pushed_at'])
                    days_since_push = (pd.Timestamp.now() - pushed_date).days

                    # 最近有更新，说明项目活跃
                    if days_since_push <= 7:
                        score = min(1.0, score + 0.1)
                    elif days_since_push <= 30:
                        score = min(1.0, score + 0.05)
                except:
                    pass

            scores.append(min(1.0, score))

        return scores

    def _calculate_visibility(self, df):
        """计算可见性"""
        scores = []

        for idx, row in df.iterrows():
            score = 0.0

            # 1. 基于star数量
            if 'stargazers_count' in df.columns:
                stars = float(row['stargazers_count'])
                if stars > 1000:
                    score = 0.9
                elif stars > 500:
                    score = 0.7
                elif stars > 100:
                    score = 0.5
                elif stars > 50:
                    score = 0.4
                elif stars > 10:
                    score = 0.3
                else:
                    score = 0.2
            else:
                score = 0.3

            # 2. 基于是否有描述和readme
            text_fields = ['description', 'topics', 'homepage']
            text_score = 0.0
            for field in text_fields:
                if field in df.columns and pd.notna(row.get(field)):
                    text = str(row[field])
                    if len(text) > 10:
                        text_score += 0.1

            score = min(1.0, score + min(text_score, 0.3))

            # 3. 基于项目是否有许可证（规范的项目通常有许可证）
            if 'license' in df.columns and pd.notna(row.get('license')):
                license_text = str(row['license']).lower()
                if license_text and license_text != 'none':
                    score = min(1.0, score + 0.1)

            # 4. 基于是否有releases
            if 'has_releases' in df.columns and row.get('has_releases'):
                score = min(1.0, score + 0.1)

            scores.append(score)

        return scores

    def _calculate_network_effect(self, df):
        """计算网络效应"""
        scores = []

        for idx, row in df.iterrows():
            score = 0.3  # 基础分数

            # 1. 基于贡献者数量
            if 'contributors_count' in df.columns:
                contributors = float(row.get('contributors_count', 0))
                if contributors > 10:
                    score = 0.8
                elif contributors > 5:
                    score = 0.6
                elif contributors > 1:
                    score = 0.4

                # 记录贡献者数
                df.at[idx, 'active_contributors'] = contributors

            # 2. 基于issue和PR活动
            activity_fields = ['open_issues_count', 'closed_issues_count']
            activity_score = 0.0
            total_activity = 0

            for field in activity_fields:
                if field in df.columns:
                    count = float(row.get(field, 0))
                    total_activity += count
                    if count > 10:
                        activity_score += 0.1
                    elif count > 5:
                        activity_score += 0.05

            # 记录总活动数
            if total_activity > 0:
                df.at[idx, 'total_activity'] = total_activity

            score = min(1.0, score + min(activity_score, 0.2))

            # 3. 基于项目被引用/依赖的情况（模拟）
            if 'stargazers_count' in df.columns:
                stars = float(row.get('stargazers_count', 0))
                if stars > 500:
                    # 高星标项目可能被更多引用
                    score = min(1.0, score + 0.2)
                elif stars > 100:
                    score = min(1.0, score + 0.1)

            # 4. 基于项目是否被fork（已有fork_ratio计算）
            if 'fork_ratio' in df.columns and idx in df.index:
                fork_ratio = df.at[idx, 'fork_ratio']
                if fork_ratio > 0.5:
                    score = min(1.0, score + 0.1)

            scores.append(score)

        return scores

    def _calculate_external_appeal_score(self, df):
        """计算综合外部吸引力分数"""
        # 确保所有子维度分数都存在
        required_scores = ['gr_score', 'vi_score', 'net_score']

        for score in required_scores:
            if score not in df.columns:
                self.logger.warning(f"缺少子维度分数: {score}，使用默认值0.5")
                df[score] = 0.5

        # 计算加权综合分数
        df['external_appeal_score'] = (
                df['gr_score'] * self.weights['growth_momentum'] +
                df['vi_score'] * self.weights['visibility'] +
                df['net_score'] * self.weights['network_effect']
        )

        # 确保分数在0-1范围内
        df['external_appeal_score'] = np.clip(df['external_appeal_score'], 0, 1)

        # 转换为百分制（可选）
        df['external_appeal_score_pct'] = df['external_appeal_score'] * 100

        return df

    def _add_appeal_level(self, df):
        """添加吸引力等级标签"""
        if 'external_appeal_score' not in df.columns:
            df['external_appeal_level'] = '未知'
            return df

        # 定义等级阈值
        conditions = [
            df['external_appeal_score'] >= 0.8,
            df['external_appeal_score'] >= 0.6,
            df['external_appeal_score'] >= 0.4,
            df['external_appeal_score'] >= 0.2
        ]

        choices = ['吸引力强', '吸引力中等', '吸引力一般', '吸引力弱', '吸引力很差']

        df['external_appeal_level'] = np.select(conditions, choices[:4], default=choices[4])

        return df

    def get_dimension_summary(self, df):
        """
        获取外部吸引力维度统计摘要

        Args:
            df: 包含外部吸引力分数的DataFrame

        Returns:
            统计摘要字典
        """
        if 'external_appeal_score' not in df.columns:
            return {"error": "未找到外部吸引力分数，请先运行calculate方法"}

        summary = {
            "维度名称": "外部吸引力",
            "项目总数": len(df),
            "平均分": df['external_appeal_score'].mean(),
            "中位数": df['external_appeal_score'].median(),
            "标准差": df['external_appeal_score'].std(),
            "最低分": df['external_appeal_score'].min(),
            "最高分": df['external_appeal_score'].max(),
            "子维度平均分": {
                "增长势头": df['gr_score'].mean() if 'gr_score' in df.columns else 0,
                "可见性": df['vi_score'].mean() if 'vi_score' in df.columns else 0,
                "网络效应": df['net_score'].mean() if 'net_score' in df.columns else 0
            },
            "等级分布": df['external_appeal_level'].value_counts().to_dict()
            if 'external_appeal_level' in df.columns else {}
        }

        # 添加关键指标统计
        key_metrics = {}
        if 'star_growth_rate' in df.columns:
            key_metrics['star_growth_rate'] = df['star_growth_rate'].mean()
        if 'fork_ratio' in df.columns:
            key_metrics['fork_ratio'] = df['fork_ratio'].mean()
        if 'active_contributors' in df.columns:
            key_metrics['active_contributors'] = df['active_contributors'].mean()

        if key_metrics:
            summary['关键指标'] = key_metrics

        return summary


# 测试函数
def test_external_appeal_calculator():
    """测试外部吸引力计算器"""
    # 创建测试数据
    test_data = {
        'full_name': ['test/popular-project', 'test/average-project', 'test/new-project'],
        'description': ['A popular web framework', 'A data analysis tool', 'New AI library'],
        'stargazers_count': [1500, 150, 10],
        'forks_count': [300, 30, 2],
        'contributors_count': [20, 5, 1],
        'open_issues_count': [50, 10, 0],
        'closed_issues_count': [200, 50, 2],
        'created_at': ['2023-01-01', '2024-01-01', '2025-01-01'],
        'pushed_at': ['2025-12-01', '2025-11-15', '2025-12-25'],
        'license': ['MIT', 'Apache-2.0', None],
        'has_releases': [True, True, False],
        'topics': ['web,framework,javascript', 'data,python,analysis', 'ai,machine-learning']
    }

    df = pd.DataFrame(test_data)

    # 创建计算器实例
    calculator = ExternalAppealCalculator()

    # 计算外部吸引力分数
    result_df = calculator.calculate(df)

    # 显示结果
    print("外部吸引力测试结果:")
    print("=" * 80)
    print(f"处理项目数: {len(result_df)}")
    print("\n各项目详细分数:")
    for _, row in result_df.iterrows():
        print(f"\n{row['full_name']}:")
        print(f"  描述: {row['description'][:50]}...")
        print(f"  Star数: {row['stargazers_count']}")
        print(f"  综合吸引力: {row.get('external_appeal_score', 'N/A'):.3f}")
        print(f"  增长势头: {row.get('gr_score', 'N/A'):.3f}")
        print(f"  可见性: {row.get('vi_score', 'N/A'):.3f}")
        print(f"  网络效应: {row.get('net_score', 'N/A'):.3f}")
        print(f"  吸引力等级: {row.get('external_appeal_level', 'N/A')}")

        # 显示衍生指标
        if 'star_growth_rate' in row:
            print(f"  日均Star增长率: {row['star_growth_rate']:.3f}")
        if 'fork_ratio' in row:
            print(f"  Fork比例: {row['fork_ratio']:.3f}")

    # 获取统计摘要
    summary = calculator.get_dimension_summary(result_df)
    print("\n维度统计摘要:")
    print("-" * 40)
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    # 运行测试
    test_external_appeal_calculator()