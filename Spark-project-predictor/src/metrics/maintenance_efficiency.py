"""
维护效率与协作质量评估模块
包含三个子维度：问题解决效率、代码审查质量、协作规范化程度
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from .base_calculator import BaseMetricCalculator


class MaintenanceEfficiencyCalculator(BaseMetricCalculator):
    """维护效率与协作质量计算器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # 子维度权重配置
        self.weights = {
            'issue_resolution_efficiency': 0.40,  # 问题解决效率
            'code_review_quality': 0.35,  # 代码审查质量
            'collaboration_standardization': 0.25,  # 协作规范化程度
        }

        # 协作相关关键词
        self.collaboration_keywords = {
            # 代码审查相关
            'code_review': ['review', 'code review', 'pull request', 'pr', 'merge'],
            'testing': ['test', 'testing', 'unit test', 'integration test', 'ci'],
            'documentation': ['doc', 'documentation', 'wiki', 'readme', 'guide'],

            # 流程规范化
            'templates': ['template', 'issue template', 'pr template', 'contributing'],
            'guidelines': ['guideline', 'style guide', 'code style', 'lint'],
            'standards': ['standard', 'convention', 'best practice', 'policy'],

            # 社区管理
            'community': ['community', 'contributor', 'maintainer', 'owner'],
            'communication': ['discussion', 'chat', 'forum', 'gitter', 'slack'],
            'governance': ['governance', 'decision', 'leadership', 'maintainership'],
        }

        # 问题解决相关指标
        self.issue_keywords = {
            'bug': ['bug', 'fix', 'error', 'issue', 'bugfix'],
            'feature': ['feature', 'enhancement', 'improvement', 'new'],
            'question': ['question', 'help', 'support', 'how to'],
            'documentation': ['doc', 'documentation', 'readme', 'wiki'],
        }

        # 代码审查实践评分
        self.review_practice_scores = {
            'mandatory_review': 20,  # 要求代码审查
            'automated_checks': 15,  # 自动化检查
            'review_templates': 10,  # 审查模板
            'review_guidelines': 10,  # 审查指南
            'review_metrics': 5,  # 审查指标追踪
        }

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算维护效率与协作质量的三个子指标

        Parameters:
        -----------
        df : pd.DataFrame
            包含项目数据的DataFrame

        Returns:
        --------
        pd.DataFrame
            添加了维护效率相关列的DataFrame
        """

        print("⚙️  开始计算维护效率与协作质量...")

        # 1. 问题解决效率 (ISE)
        print("  计算子维度1: 问题解决效率")
        df['ise_score'] = self._calculate_issue_resolution_efficiency(df)

        # 2. 代码审查质量 (CRQ)
        print("  计算子维度2: 代码审查质量")
        df['crq_score'] = self._calculate_code_review_quality(df)

        # 3. 协作规范化程度 (CSD)
        print("  计算子维度3: 协作规范化程度")
        df['csd_score'] = self._calculate_collaboration_standardization(df)

        # 4. 综合维护效率得分
        df['maintenance_efficiency_score'] = (
                self.weights['issue_resolution_efficiency'] * df['ise_score'] +
                self.weights['code_review_quality'] * df['crq_score'] +
                self.weights['collaboration_standardization'] * df['csd_score']
        )

        # 确保分数在0-100范围内
        df['maintenance_efficiency_score'] = df['maintenance_efficiency_score'].clip(0, 100)

        print(f"✅ 维护效率计算完成")
        print(f"  平均分: {df['maintenance_efficiency_score'].mean():.2f}")
        print(
            f"  范围: {df['maintenance_efficiency_score'].min():.2f} - {df['maintenance_efficiency_score'].max():.2f}")

        return df

    def _calculate_issue_resolution_efficiency(self, df: pd.DataFrame) -> pd.Series:
        """
        计算问题解决效率评分

        基于现有数据推断issue解决效率：
        1. open_issues_count与项目规模的关系
        2. 项目活跃度与issue数量的关系
        3. 描述中是否提及issue处理策略
        """

        scores = []

        for _, row in df.iterrows():
            score = 50  # 基础分

            # 1. 基于open_issues_count的评估 (0-40分)
            issues_score = self._evaluate_issues_count(
                row.get('open_issues_count', 0),
                row.get('stargazers_count', 0),
                row.get('forks_count', 0)
            )
            score += issues_score

            # 2. 基于项目年龄和更新频率 (0-30分)
            activity_score = self._evaluate_activity_for_issue_resolution(
                row.get('created_at'),
                row.get('pushed_at'),
                row.get('updated_at')
            )
            score += activity_score

            # 3. 基于描述的issue管理实践 (0-20分)
            practice_score = self._evaluate_issue_practices(
                row.get('description', ''),
                row.get('topics', '')
            )
            score += practice_score

            # 4. 基于语言的项目规模调整 (-10 to +10分)
            adjustment = self._adjust_by_language_and_scale(
                row.get('language', 'Unknown'),
                row.get('stargazers_count', 0)
            )
            score += adjustment

            scores.append(min(100, max(0, score)))

        return pd.Series(scores, index=df.index)

    def _evaluate_issues_count(self, issues_count: int, stars: int, forks: int) -> float:
        """
        评估issue数量与项目规模的关系

        逻辑：
        - 完全没有issue：可能项目太新或无人使用（中等分数）
        - issue数量与star/fork比例合理：高分
        - issue过多但star/fork少：可能维护跟不上
        - issue极少但star/fork多：可能issue管理严格
        """

        if issues_count == 0:
            if stars == 0 and forks == 0:
                return 0  # 项目无人使用
            elif stars < 10 and forks < 5:
                return 20  # 小项目，合理
            else:
                return 10  # 有一定规模但无issue，可能issue被关闭或转移

        # 计算issue与项目规模的相对比例
        project_size = stars + forks * 2 + 1  # 加1避免除零，fork权重更高

        # 理想比例：每100个star/fork有1-5个open issue
        ideal_ratio_min = 0.01
        ideal_ratio_max = 0.05

        actual_ratio = issues_count / project_size

        if actual_ratio < ideal_ratio_min:
            # issue太少，可能维护严格或issue被快速解决
            return 35
        elif actual_ratio <= ideal_ratio_max:
            # 在理想范围内
            return 40
        elif actual_ratio <= ideal_ratio_max * 2:
            # 稍多，但可接受
            return 25
        elif actual_ratio <= ideal_ratio_max * 5:
            # 较多，可能维护压力大
            return 15
        else:
            # 太多，维护可能跟不上
            return 5

    def _evaluate_activity_for_issue_resolution(self, created_at, pushed_at, updated_at) -> float:
        """
        基于项目活跃度评估issue解决可能性
        """

        if pd.isna(created_at) or pd.isna(pushed_at):
            return 10  # 默认分

        try:
            current_time = pd.Timestamp.now(tz='UTC')

            # 1. 项目年龄
            age_days = (current_time - created_at).days
            age_months = age_days / 30.44

            # 2. 最近更新
            days_since_update = (current_time - pushed_at).days

            # 3. 更新频率评分
            if days_since_update <= 7:
                update_score = 15  # 非常活跃
            elif days_since_update <= 30:
                update_score = 12  # 活跃
            elif days_since_update <= 90:
                update_score = 8  # 一般活跃
            elif days_since_update <= 180:
                update_score = 4  # 不太活跃
            else:
                update_score = 0  # 不活跃

            # 4. 项目成熟度评分（老项目更可能有稳定流程）
            if age_months < 3:
                maturity_score = 5  # 太新
            elif age_months < 12:
                maturity_score = 10  # 有一定历史
            else:
                maturity_score = 15  # 成熟项目

            return update_score + maturity_score

        except:
            return 10

    def _evaluate_issue_practices(self, description: str, topics: str) -> float:
        """评估issue管理实践"""

        score = 0
        text_to_check = ""

        if pd.notna(description):
            text_to_check += str(description).lower() + " "
        if pd.notna(topics):
            text_to_check += str(topics).lower()

        # 检查issue管理关键词
        issue_keywords = [
            'issue', 'bug', 'feature request', 'bug report',
            'support', 'help', 'question', 'discussion'
        ]

        found_keywords = 0
        for keyword in issue_keywords:
            if keyword in text_to_check:
                found_keywords += 1

        # 每找到一个关键词加2分，最多10分
        score += min(10, found_keywords * 2)

        # 检查issue模板或流程
        template_keywords = ['template', 'form', 'guideline', 'process']
        for keyword in template_keywords:
            if keyword in text_to_check:
                score += 5
                break

        # 检查标签系统
        label_keywords = ['label', 'tag', 'categor', 'priority']
        for keyword in label_keywords:
            if keyword in text_to_check:
                score += 5
                break

        return min(20, score)

    def _adjust_by_language_and_scale(self, language: str, stars: int) -> float:
        """根据语言和项目规模调整分数"""

        adjustment = 0

        # 不同语言社区的issue文化不同
        language_adjustments = {
            'Rust': 5,  # Rust社区以严谨著称
            'Go': 3,  # Go社区重视简洁
            'Python': 0,  # Python社区适中
            'JavaScript': -2,  # JS项目可能issue较多
            'TypeScript': 0,
            'Java': 2,
            'C++': 2,
            'Unknown': -5,
        }

        adjustment += language_adjustments.get(language, 0)

        # 根据项目规模调整
        if stars > 1000:
            adjustment -= 5  # 大项目issue管理更难
        elif stars > 100:
            adjustment += 0  # 中等项目适中
        else:
            adjustment += 3  # 小项目更容易管理

        return adjustment

    def _calculate_code_review_quality(self, df: pd.DataFrame) -> pd.Series:
        """
        计算代码审查质量评分

        基于描述和主题推断代码审查实践：
        1. 是否有代码审查相关描述
        2. 是否有自动化测试/CI/CD
        3. 是否有贡献指南
        """

        scores = []

        for _, row in df.iterrows():
            score = 40  # 基础分

            # 1. 从描述和主题中提取代码审查关键词 (0-30分)
            review_score = self._extract_review_keywords(
                row.get('description', ''),
                row.get('topics', '')
            )
            score += review_score

            # 2. 基于工程实践推断 (0-20分)
            practice_score = self._infer_engineering_practices(
                row.get('description', ''),
                row.get('topics', '')
            )
            score += practice_score

            # 3. 基于项目活跃度和规模 (0-10分)
            activity_score = self._evaluate_review_activity(
                row.get('stargazers_count', 0),
                row.get('forks_count', 0),
                row.get('open_issues_count', 0)
            )
            score += activity_score

            scores.append(min(100, max(0, score)))

        return pd.Series(scores, index=df.index)

    def _extract_review_keywords(self, description: str, topics: str) -> float:
        """从文本中提取代码审查关键词"""

        score = 0
        text_to_check = ""

        if pd.notna(description):
            text_to_check += str(description).lower() + " "
        if pd.notna(topics):
            text_to_check += str(topics).lower()

        # 代码审查直接关键词
        review_direct_keywords = [
            'code review', 'pull request review', 'pr review',
            'review process', 'merge request', 'mr'
        ]

        for keyword in review_direct_keywords:
            if keyword in text_to_check:
                score += 10
                break  # 找到直接证据就加分

        # 审查相关实践
        review_practice_keywords = [
            'approval', 'reviewer', 'maintainer review',
            'required review', 'mandatory review'
        ]

        found_practices = 0
        for keyword in review_practice_keywords:
            if keyword in text_to_check:
                found_practices += 1

        score += min(10, found_practices * 3)

        # 自动化审查工具
        tool_keywords = [
            'sonarqube', 'codeclimate', 'codacy', 'reviewable',
            'pullapprove', 'codefactor', 'houndci'
        ]

        for keyword in tool_keywords:
            if keyword in text_to_check:
                score += 5
                break

        return min(30, score)

    def _infer_engineering_practices(self, description: str, topics: str) -> float:
        """从工程实践推断代码审查质量"""

        score = 0
        text_to_check = ""

        if pd.notna(description):
            text_to_check += str(description).lower() + " "
        if pd.notna(topics):
            text_to_check += str(topics).lower()

        # CI/CD实践（通常与代码审查结合）
        ci_cd_keywords = ['ci', 'cd', 'continuous integration', 'continuous delivery']
        for keyword in ci_cd_keywords:
            if keyword in text_to_check:
                score += 5
                break

        # 测试实践
        test_keywords = ['test', 'testing', 'unit test', 'integration test', 'coverage']
        test_count = 0
        for keyword in test_keywords:
            if keyword in text_to_check:
                test_count += 1

        score += min(5, test_count)

        # 代码质量工具
        quality_keywords = ['lint', 'linter', 'static analysis', 'code quality']
        for keyword in quality_keywords:
            if keyword in text_to_check:
                score += 5
                break

        # 贡献指南（通常包含审查流程）
        contributing_keywords = ['contributing', 'contribute', 'contributor']
        for keyword in contributing_keywords:
            if keyword in text_to_check:
                score += 5
                break

        return min(20, score)

    def _evaluate_review_activity(self, stars: int, forks: int, issues: int) -> float:
        """基于项目活跃度评估代码审查可能性"""

        # 项目规模越大，越可能需要代码审查
        project_scale = stars + forks * 0.5

        if project_scale < 10:
            return 3  # 小项目可能没有正式审查
        elif project_scale < 100:
            return 6  # 中等项目可能开始有审查
        elif project_scale < 1000:
            return 8  # 较大项目应该有审查
        else:
            return 10  # 大项目必须有审查

    def _calculate_collaboration_standardization(self, df: pd.DataFrame) -> pd.Series:
        """
        计算协作规范化程度评分

        评估项目协作流程的规范化程度：
        1. 是否有明确的贡献指南
        2. 是否有模板系统
        3. 是否有行为准则
        4. 是否有版本管理规范
        """

        scores = []

        for _, row in df.iterrows():
            score = 30  # 基础分

            # 1. 贡献指南和模板 (0-30分)
            guideline_score = self._evaluate_guidelines_and_templates(
                row.get('description', ''),
                row.get('topics', '')
            )
            score += guideline_score

            # 2. 社区管理规范 (0-20分)
            community_score = self._evaluate_community_management(
                row.get('description', ''),
                row.get('topics', '')
            )
            score += community_score

            # 3. 版本和发布管理 (0-20分)
            release_score = self._evaluate_release_management(
                row.get('description', ''),
                row.get('topics', '')
            )
            score += release_score

            scores.append(min(100, max(0, score)))

        return pd.Series(scores, index=df.index)

    def _evaluate_guidelines_and_templates(self, description: str, topics: str) -> float:
        """评估贡献指南和模板"""

        score = 0
        text_to_check = ""

        if pd.notna(description):
            text_to_check += str(description).lower() + " "
        if pd.notna(topics):
            text_to_check += str(topics).lower()

        # 贡献指南
        contributing_terms = [
            'contributing', 'contribution guidelines', 'contributor guide',
            'how to contribute', 'development guide'
        ]

        for term in contributing_terms:
            if term in text_to_check:
                score += 10
                break

        # 模板
        template_terms = [
            'template', 'issue template', 'pull request template',
            'bug report template', 'feature request template'
        ]

        for term in template_terms:
            if term in text_to_check:
                score += 8
                break

        # 代码风格指南
        style_terms = ['style guide', 'coding style', 'code convention', 'lint']
        for term in style_terms:
            if term in text_to_check:
                score += 7
                break

        # 文档指南
        doc_terms = ['documentation', 'doc', 'readme', 'wiki']
        doc_count = 0
        for term in doc_terms:
            if term in text_to_check:
                doc_count += 1

        score += min(5, doc_count * 2)

        return min(30, score)

    def _evaluate_community_management(self, description: str, topics: str) -> float:
        """评估社区管理规范"""

        score = 0
        text_to_check = ""

        if pd.notna(description):
            text_to_check += str(description).lower() + " "
        if pd.notna(topics):
            text_to_check += str(topics).lower()

        # 行为准则
        coc_terms = [
            'code of conduct', 'coc', 'conduct',
            'community guidelines', 'community standards'
        ]

        for term in coc_terms:
            if term in text_to_check:
                score += 10
                break

        # 沟通渠道
        communication_terms = [
            'discussion', 'forum', 'chat', 'gitter', 'slack',
            'discord', 'matrix', 'irc', 'mailing list'
        ]

        comm_count = 0
        for term in communication_terms:
            if term in text_to_check:
                comm_count += 1

        score += min(5, comm_count * 2)

        # 决策流程
        decision_terms = ['governance', 'decision', 'rfc', 'proposal']
        for term in decision_terms:
            if term in text_to_check:
                score += 5
                break

        return min(20, score)

    def _evaluate_release_management(self, description: str, topics: str) -> float:
        """评估版本和发布管理"""

        score = 0
        text_to_check = ""

        if pd.notna(description):
            text_to_check += str(description).lower() + " "
        if pd.notna(topics):
            text_to_check += str(topics).lower()

        # 版本管理
        version_terms = [
            'version', 'release', 'semver', 'semantic versioning',
            'changelog', 'release notes'
        ]

        version_count = 0
        for term in version_terms:
            if term in text_to_check:
                version_count += 1

        score += min(10, version_count * 3)

        # 发布流程
        release_terms = ['release process', 'deploy', 'publish', 'distribution']
        for term in release_terms:
            if term in text_to_check:
                score += 5
                break

        # 稳定性保证
        stability_terms = ['stable', 'production', 'enterprise', 'reliable']
        for term in stability_terms:
            if term in text_to_check:
                score += 5
                break

        return min(20, score)

    def generate_detailed_report(self, df: pd.DataFrame, top_n: int = 10) -> Dict[str, Any]:
        """
        生成维护效率详细报告

        Parameters:
        -----------
        df : pd.DataFrame
            包含维护效率评分的DataFrame
        top_n : int
            显示前N名项目

        Returns:
        --------
        Dict[str, Any]
            包含详细统计信息的字典
        """

        report = {
            'summary': {
                'total_projects': len(df),
                'avg_efficiency_score': df['maintenance_efficiency_score'].mean(),
                'median_efficiency_score': df['maintenance_efficiency_score'].median(),
                'std_efficiency_score': df['maintenance_efficiency_score'].std(),
            },
            'subdimension_analysis': {},
            'top_performers': [],
            'recommendations': []
        }

        # 子维度分析
        subdims = ['ise_score', 'crq_score', 'csd_score']
        subdim_names = {
            'ise_score': '问题解决效率',
            'crq_score': '代码审查质量',
            'csd_score': '协作规范化'
        }

        for dim in subdims:
            if dim in df.columns:
                report['subdimension_analysis'][subdim_names[dim]] = {
                    'average': df[dim].mean(),
                    'median': df[dim].median(),
                    'std': df[dim].std(),
                    'top_5_avg': df.nlargest(5, dim)[dim].mean()
                }

        # 表现最佳的项目
        if 'maintenance_efficiency_score' in df.columns:
            top_projects = df.nlargest(top_n, 'maintenance_efficiency_score')[
                ['full_name', 'language', 'stargazers_count', 'open_issues_count',
                 'maintenance_efficiency_score', 'ise_score', 'crq_score', 'csd_score']
            ]

            for _, row in top_projects.iterrows():
                report['top_performers'].append({
                    'full_name': row['full_name'],
                    'language': row['language'],
                    'stars': int(row['stargazers_count']),
                    'open_issues': int(row['open_issues_count']),
                    'total_score': round(row['maintenance_efficiency_score'], 2),
                    'ise_score': round(row['ise_score'], 2),
                    'crq_score': round(row['crq_score'], 2),
                    'csd_score': round(row['csd_score'], 2)
                })

        # 生成改进建议
        report['recommendations'] = self._generate_efficiency_recommendations(df)

        return report

    def _generate_efficiency_recommendations(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """生成针对维护效率的改进建议"""

        recommendations = []

        # 1. 检查普遍问题
        subdim_columns = ['ise_score', 'crq_score', 'csd_score']
        subdim_names = ['问题解决效率', '代码审查质量', '协作规范化']

        low_dimensions = []
        for col, name in zip(subdim_columns, subdim_names):
            if col in df.columns and df[col].mean() < 50:
                low_dimensions.append(name)

        if low_dimensions:
            recommendations.append({
                "priority": "高",
                "issue": f"多个子维度得分偏低",
                "suggestion": f"重点关注: {', '.join(low_dimensions)}",
                "action": "分析低分项目的具体原因，提供改进模板"
            })

        # 2. 检查issue管理
        if 'open_issues_count' in df.columns:
            high_issue_projects = df[df['open_issues_count'] > 50]
            if len(high_issue_projects) > 0:
                recommendations.append({
                    "priority": "中",
                    "issue": f"{len(high_issue_projects)} 个项目open issue过多",
                    "suggestion": "建立issue分类和优先级系统",
                    "action": "引入issue模板和自动化标签"
                })

        # 3. 检查缺乏协作规范
        if 'description' in df.columns:
            no_description = df[df['description'].isna() | (df['description'].str.len() < 20)]
            if len(no_description) > 0:
                recommendations.append({
                    "priority": "中",
                    "issue": f"{len(no_description)} 个项目缺乏详细描述",
                    "suggestion": "完善项目描述，明确贡献方式",
                    "action": "提供README模板和贡献指南示例"
                })

        # 4. 基于语言的分析
        if 'language' in df.columns and 'maintenance_efficiency_score' in df.columns:
            lang_stats = []
            for lang in df['language'].unique():
                if pd.notna(lang) and lang != 'Unknown':
                    lang_df = df[df['language'] == lang]
                    if len(lang_df) >= 3:
                        avg_score = lang_df['maintenance_efficiency_score'].mean()
                        lang_stats.append((lang, avg_score, len(lang_df)))

            # 找出表现最差的语言
            if lang_stats:
                lang_stats.sort(key=lambda x: x[1])
                worst_lang, worst_score, count = lang_stats[0]
                if worst_score < 50:
                    recommendations.append({
                        "priority": "低",
                        "issue": f"{worst_lang} 项目平均维护效率较低 ({worst_score:.1f}分)",
                        "suggestion": f"分析{worst_lang}生态的协作特点",
                        "action": f"为{worst_lang}项目提供针对性的协作指南"
                    })

        # 如果没有发现明显问题
        if not recommendations:
            recommendations.append({
                "priority": "低",
                "issue": "无显著问题",
                "suggestion": "当前维护效率整体良好",
                "action": "继续保持现有协作实践"
            })

        return recommendations


# 测试函数
def test_maintenance_efficiency_calculator():
    """测试维护效率计算器"""
    print("🧪 测试 MaintenanceEfficiencyCalculator...")

    # 创建测试数据
    test_data = {
        'full_name': ['test/repo1', 'test/repo2', 'test/repo3'],
        'language': ['Python', 'Rust', 'JavaScript'],
        'description': [
            'A well-maintained project with CI/CD and code review',
            'High performance library with contributing guidelines',
            'Simple script with no documentation'
        ],
        'topics': ['python,ci-cd,testing', 'rust,performance,no-std', ''],
        'stargazers_count': [150, 80, 10],
        'forks_count': [30, 15, 2],
        'open_issues_count': [5, 2, 15],
        'created_at': [
            pd.Timestamp('2023-01-01', tz='UTC'),
            pd.Timestamp('2023-06-01', tz='UTC'),
            pd.Timestamp('2024-01-01', tz='UTC')
        ],
        'pushed_at': [
            pd.Timestamp('2024-05-01', tz='UTC'),
            pd.Timestamp('2024-05-15', tz='UTC'),
            pd.Timestamp('2024-01-15', tz='UTC')
        ]
    }

    df_test = pd.DataFrame(test_data)

    # 创建计算器实例
    calculator = MaintenanceEfficiencyCalculator()

    # 计算分数
    result = calculator.calculate(df_test)

    print(f"\n测试结果:")
    for idx, row in result.iterrows():
        print(f"  {row['full_name']} ({row['language']}): {row['maintenance_efficiency_score']:.2f}")

    # 生成报告
    report = calculator.generate_detailed_report(result, top_n=2)

    print(f"\n报告摘要:")
    print(f"  平均分: {report['summary']['avg_efficiency_score']:.2f}")

    return result


if __name__ == "__main__":
    # 运行测试
    test_result = test_maintenance_efficiency_calculator()
    print("\n✅ 维护效率计算器测试完成")