"""
代码与工程健康度评估模块
包含三个子维度：代码结构质量、工程实践成熟度、文档完整性
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from .base_calculator import BaseMetricCalculator


class CodeHealthCalculator(BaseMetricCalculator):
    """代码与工程健康度计算器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # 子维度权重配置
        self.weights = {
            'code_structure_quality': 0.45,  # 代码结构质量
            'engineering_practice_maturity': 0.35,  # 工程实践成熟度
            'documentation_completeness': 0.20,  # 文档完整性
        }

        # 语言技术栈评分（基于社区认知和技术趋势）
        self.language_quality_scores = {
            # 高工程实践语言（90-100分）
            'Rust': 98, 'Go': 95, 'TypeScript': 92, 'Kotlin': 90,
            # 主流语言（80-89分）
            'Python': 88, 'Java': 85, 'C++': 85, 'C#': 84,
            'JavaScript': 82, 'Swift': 85, 'Dart': 83,
            # 脚本语言（70-79分）
            'Ruby': 78, 'PHP': 75, 'Shell': 72, 'Perl': 70,
            # 其他语言（60-69分）
            'R': 68, 'Scala': 85, 'Haskell': 90, 'Elixir': 88,
            'Clojure': 85, 'F#': 86,
            # 标记语言和配置（50-69分）
            'HTML': 65, 'CSS': 65, 'TeX': 60, 'Vim Script': 55,
            # 默认值
            'Unknown': 70, '': 60, None: 60
        }

        # 工程实践相关关键词
        self.engineering_keywords = {
            'test': ['test', 'tests', 'testing', 'unit-test', 'pytest', 'jest', 'junit'],
            'ci_cd': ['ci', 'cd', 'github-actions', 'travis', 'jenkins', 'gitlab-ci', 'circleci'],
            'docker': ['docker', 'container', 'kubernetes', 'k8s', 'dockerfile'],
            'security': ['security', 'sast', 'dast', 'sonarqube', 'codeql'],
            'coverage': ['coverage', 'codecov', 'coveralls'],
            'linter': ['lint', 'flake8', 'eslint', 'prettier', 'black']
        }

        # 文档质量关键词
        self.documentation_keywords = {
            'api': ['api', 'rest', 'graphql', 'endpoint', 'swagger', 'openapi'],
            'tutorial': ['tutorial', 'guide', 'example', 'demo', 'quickstart'],
            'installation': ['install', 'setup', 'configuration', 'prerequisite'],
            'contributing': ['contributing', 'contribute', 'develop', 'development'],
            'license': ['license', 'licence', 'mit', 'apache', 'gpl']
        }

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算代码与工程健康度的三个子指标

        Parameters:
        -----------
        df : pd.DataFrame
            包含项目数据的DataFrame

        Returns:
        --------
        pd.DataFrame
            添加了代码健康度相关列的DataFrame
        """

        print("💻 开始计算代码与工程健康度...")

        # 1. 代码结构质量 (CSQ)
        print("  计算子维度1: 代码结构质量")
        df['csq_score'] = self._calculate_code_structure_quality(df)

        # 2. 工程实践成熟度 (EPM)
        print("  计算子维度2: 工程实践成熟度")
        df['epm_score'] = self._calculate_engineering_practice_maturity(df)

        # 3. 文档完整性 (DC)
        print("  计算子维度3: 文档完整性")
        df['dc_score'] = self._calculate_documentation_completeness(df)

        # 4. 综合代码健康度
        df['code_health_score'] = (
                self.weights['code_structure_quality'] * df['csq_score'] +
                self.weights['engineering_practice_maturity'] * df['epm_score'] +
                self.weights['documentation_completeness'] * df['dc_score']
        )

        # 确保分数在0-100范围内
        df['code_health_score'] = df['code_health_score'].clip(0, 100)

        print(f"✅ 代码健康度计算完成")
        print(f"  平均分: {df['code_health_score'].mean():.2f}")
        print(f"  范围: {df['code_health_score'].min():.2f} - {df['code_health_score'].max():.2f}")

        return df

    def _calculate_code_structure_quality(self, df: pd.DataFrame) -> pd.Series:
        """
        计算代码结构质量评分（简化版）

        实际项目中应使用代码分析工具（如SonarQube、CodeClimate）
        这里使用语言评分和代理指标
        """

        # 1. 语言质量基础分
        df['language_score'] = df['language'].map(
            lambda x: self.language_quality_scores.get(x, 70)
        )

        # 2. 基于issues的代码健康度代理
        # 问题数量适中表示活跃维护，过多可能表示代码质量问题
        df['issues_health_proxy'] = df['open_issues_count'].apply(
            lambda x: self._issues_health_score(x)
        )

        # 3. 基于star和fork的代码质量认可度
        # 高star/fork比可能表示代码质量高（被更多人认可）
        df['recognition_ratio'] = np.where(
            df['forks_count'] > 0,
            df['stargazers_count'] / (df['forks_count'] + 1),
            0
        )
        df['recognition_score'] = self.normalize_score(
            np.log1p(df['recognition_ratio']), 'minmax'
        )

        # 4. 综合代码结构质量分数
        csq_score = (
                0.50 * df['language_score'] +
                0.30 * df['issues_health_proxy'] +
                0.20 * df['recognition_score']
        )

        return self.normalize_score(csq_score)

    def _issues_health_score(self, issue_count: int) -> float:
        """
        根据issue数量评估代码健康度

        逻辑：
        - 0个issues：可能项目太新或无人使用（中等分数）
        - 1-10个issues：良好维护
        - 10-50个issues：正常范围
        - 50+个issues：可能维护跟不上
        """
        if issue_count == 0:
            return 70
        elif issue_count <= 5:
            return 90
        elif issue_count <= 20:
            return 80
        elif issue_count <= 50:
            return 70
        elif issue_count <= 100:
            return 60
        else:
            return max(30, 100 - np.log(issue_count) * 10)

    def _calculate_engineering_practice_maturity(self, df: pd.DataFrame) -> pd.Series:
        """
        计算工程实践成熟度评分

        基于项目描述、主题标签中的关键词推断工程实践采用情况
        后续可扩展：通过API检查配置文件存在性
        """

        # 1. 从description和topics中提取工程实践关键词
        df['engineering_keyword_count'] = df.apply(
            lambda row: self._count_engineering_keywords(row), axis=1
        )

        # 2. 基于语言的技术栈成熟度
        # 某些语言社区有更好的工程实践传统
        language_maturity_scores = {
            'Rust': 95, 'Go': 93, 'TypeScript': 90, 'Python': 88,
            'Java': 87, 'C++': 85, 'JavaScript': 82, 'C#': 84,
            'Ruby': 80, 'PHP': 75, 'Shell': 70, 'HTML': 60,
            'Unknown': 65, '': 60, None: 60
        }
        df['language_maturity_score'] = df['language'].map(
            lambda x: language_maturity_scores.get(x, 70)
        )

        # 3. 基于更新频率的工程活跃度
        # 最近更新的项目更可能采用现代工程实践
        current_time = pd.Timestamp.now(tz='UTC')
        df['days_since_update'] = (current_time - df['pushed_at']).dt.days

        df['update_recency_score'] = df['days_since_update'].apply(
            lambda days: 100 * np.exp(-0.05 * days) if days >= 0 else 50
        )

        # 4. 综合工程实践成熟度分数
        keyword_score = self.normalize_score(df['engineering_keyword_count'], 'minmax')

        epm_score = (
                0.40 * df['language_maturity_score'] +
                0.30 * keyword_score +
                0.30 * df['update_recency_score']
        )

        return self.normalize_score(epm_score)

    def _count_engineering_keywords(self, row) -> int:
        """统计工程实践关键词出现次数"""
        count = 0

        # 检查description
        if pd.notna(row.get('description')):
            desc = str(row['description']).lower()
            for category, keywords in self.engineering_keywords.items():
                for keyword in keywords:
                    if keyword in desc:
                        count += 1
                        break  # 每类只计一次

        # 检查topics
        if pd.notna(row.get('topics')):
            topics = str(row['topics']).lower()
            for category, keywords in self.engineering_keywords.items():
                for keyword in keywords:
                    if keyword in topics:
                        count += 1
                        break

        return count

    def _calculate_documentation_completeness(self, df: pd.DataFrame) -> pd.Series:
        """
        计算文档完整性评分

        基于描述、主题标签评估文档质量
        后续可扩展：通过API获取README内容分析
        """

        scores = []

        for _, row in df.iterrows():
            score = 50  # 基础分

            # 1. 描述完整性 (0-30分)
            desc_score = self._evaluate_description(row.get('description', ''))

            # 2. 主题标签中的文档关键词 (0-20分)
            topic_score = self._evaluate_topics_for_docs(row.get('topics', ''))

            # 3. 项目名称规范性 (0-10分)
            name_score = self._evaluate_project_name(row.get('full_name', ''))

            # 4. 是否有LICENSE文件（通过主题推断）(0-15分)
            license_score = self._evaluate_license_info(
                row.get('topics', ''),
                row.get('description', '')
            )

            # 5. 项目年龄加分（老项目更可能有文档）(0-15分)
            age_score = self._evaluate_project_age(row.get('created_at'))

            total_score = desc_score + topic_score + name_score + license_score + age_score
            scores.append(min(100, total_score))

        return pd.Series(scores, index=df.index)

    def _evaluate_description(self, description: str) -> float:
        """评估项目描述质量"""
        if pd.isna(description) or not description:
            return 0

        desc = str(description)
        score = 0

        # 长度评分
        if len(desc) >= 200:
            score += 15
        elif len(desc) >= 100:
            score += 10
        elif len(desc) >= 50:
            score += 5

        # 结构评分
        lines = desc.split('\n')
        if len(lines) >= 3:
            score += 5

        # 关键词评分
        desc_lower = desc.lower()
        doc_keywords_found = 0

        for category, keywords in self.documentation_keywords.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    doc_keywords_found += 1
                    break

        score += min(10, doc_keywords_found * 2)

        return score

    def _evaluate_topics_for_docs(self, topics: str) -> float:
        """从主题标签评估文档质量"""
        if pd.isna(topics) or not topics:
            return 0

        topics_str = str(topics).lower()
        score = 0

        # 检查文档相关主题
        doc_topics = ['documentation', 'docs', 'wiki', 'guide', 'tutorial', 'example']
        for topic in doc_topics:
            if topic in topics_str:
                score += 5

        # 检查开发相关主题
        dev_topics = ['development', 'contributing', 'hacktoberfest', 'good-first-issue']
        for topic in dev_topics:
            if topic in topics_str:
                score += 3

        return min(20, score)

    def _evaluate_project_name(self, full_name: str) -> float:
        """评估项目名称规范性"""
        if pd.isna(full_name):
            return 0

        name = str(full_name)
        score = 5  # 基础分

        # 检查命名规范
        if '/' in name and len(name.split('/')) == 2:
            score += 3  # 符合 owner/repo 格式

        # 检查特殊字符
        if re.match(r'^[a-zA-Z0-9_\-\./]+$', name):
            score += 2

        return score

    def _evaluate_license_info(self, topics: str, description: str) -> float:
        """评估许可证信息"""
        score = 0
        text_to_check = ""

        if pd.notna(topics):
            text_to_check += str(topics).lower() + " "
        if pd.notna(description):
            text_to_check += str(description).lower()

        # 检查常见许可证关键词
        license_keywords = [
            'mit license', 'apache license', 'gpl', 'bsd license',
            'license', 'licence', 'licensed under'
        ]

        for keyword in license_keywords:
            if keyword in text_to_check:
                score += 5
                break

        # 检查具体许可证类型
        specific_licenses = ['mit', 'apache-2.0', 'gpl-3.0', 'bsd-3-clause']
        for license_type in specific_licenses:
            if license_type in text_to_check:
                score += 5
                break

        return min(15, score)

    def _evaluate_project_age(self, created_at) -> float:
        """基于项目年龄评估文档完善可能性"""
        if pd.isna(created_at):
            return 0

        try:
            # 计算项目年龄（月）
            if hasattr(created_at, 'tz'):
                current_time = pd.Timestamp.now(tz='UTC')
            else:
                current_time = pd.Timestamp.now()

            age_days = (current_time - created_at).days
            age_months = age_days / 30.44

            # 年龄评分逻辑
            if age_months < 3:
                return 5  # 太新，文档可能不完善
            elif age_months < 12:
                return 10  # 有一定时间发展文档
            else:
                return 15  # 老项目更可能有完善文档
        except:
            return 0

    def generate_detailed_report(self, df: pd.DataFrame, top_n: int = 10) -> Dict[str, Any]:
        """
        生成代码健康度详细报告

        Parameters:
        -----------
        df : pd.DataFrame
            包含代码健康度评分的DataFrame
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
                'avg_code_health_score': df['code_health_score'].mean(),
                'median_code_health_score': df['code_health_score'].median(),
                'std_code_health_score': df['code_health_score'].std(),
            },
            'language_analysis': {},
            'top_performers': [],
            'bottom_performers': [],
            'recommendations': []
        }

        # 按语言分析
        if 'language' in df.columns:
            language_stats = []
            for lang in df['language'].unique():
                if pd.notna(lang):
                    lang_df = df[df['language'] == lang]
                    if len(lang_df) >= 5:  # 至少有5个项目才统计
                        language_stats.append({
                            'language': lang,
                            'count': len(lang_df),
                            'avg_score': lang_df['code_health_score'].mean(),
                            'median_score': lang_df['code_health_score'].median()
                        })

            # 按平均分排序
            language_stats.sort(key=lambda x: x['avg_score'], reverse=True)
            report['language_analysis'] = language_stats[:10]  # 前10种语言

        # 表现最佳的项目
        top_projects = df.nlargest(top_n, 'code_health_score')[
            ['full_name', 'language', 'stargazers_count',
             'code_health_score', 'csq_score', 'epm_score', 'dc_score']
        ]

        for _, row in top_projects.iterrows():
            report['top_performers'].append({
                'full_name': row['full_name'],
                'language': row['language'],
                'stars': int(row['stargazers_count']),
                'total_score': round(row['code_health_score'], 2),
                'csq_score': round(row['csq_score'], 2),
                'epm_score': round(row['epm_score'], 2),
                'dc_score': round(row['dc_score'], 2)
            })

        # 表现最差的项目（需要改进的）
        bottom_projects = df.nsmallest(min(top_n, len(df)), 'code_health_score')[
            ['full_name', 'language', 'stargazers_count', 'code_health_score']
        ]

        for _, row in bottom_projects.iterrows():
            report['bottom_performers'].append({
                'full_name': row['full_name'],
                'language': row['language'],
                'stars': int(row['stargazers_count']),
                'score': round(row['code_health_score'], 2)
            })

        # 生成改进建议
        report['recommendations'] = self._generate_recommendations(df)

        return report

    def _generate_recommendations(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """生成针对低分项目的改进建议"""
        recommendations = []

        # 找出低分项目（低于平均分1个标准差）
        threshold = df['code_health_score'].mean() - df['code_health_score'].std()
        low_score_projects = df[df['code_health_score'] < threshold]

        if len(low_score_projects) == 0:
            return [{"message": "所有项目代码健康度均在合理范围内"}]

        # 分析常见问题
        common_issues = []

        # 1. 文档问题
        if 'dc_score' in low_score_projects.columns:
            low_docs = low_score_projects[low_score_projects['dc_score'] < 50]
            if len(low_docs) > len(low_score_projects) * 0.5:
                common_issues.append("文档不完善是普遍问题")

        # 2. 工程实践问题
        if 'epm_score' in low_score_projects.columns:
            low_epm = low_score_projects[low_score_projects['epm_score'] < 50]
            if len(low_epm) > len(low_score_projects) * 0.4:
                common_issues.append("缺乏现代工程实践")

        # 3. 语言相关问题
        if 'language' in low_score_projects.columns:
            lang_counts = low_score_projects['language'].value_counts()
            for lang, count in lang_counts.head(3).items():
                if count > 5:
                    common_issues.append(f"{lang} 项目整体表现不佳")

        # 生成建议
        if common_issues:
            recommendations.append({
                "issue": "多项健康度问题",
                "suggestion": f"重点关注: {', '.join(common_issues[:3])}",
                "action": "建议为低分项目提供具体的改进指南"
            })

        # 针对没有描述的项目
        no_description = df[df['description'].isna() | (df['description'].str.len() < 10)]
        if len(no_description) > 0:
            recommendations.append({
                "issue": f"{len(no_description)} 个项目缺乏描述",
                "suggestion": "添加详细的项目描述",
                "action": "在README中明确项目目的、功能和用法"
            })

        return recommendations


# 测试函数
def test_code_health_calculator():
    """测试代码健康度计算器"""
    print("🧪 测试 CodeHealthCalculator...")

    # 创建测试数据
    test_data = {
        'full_name': ['test/repo1', 'test/repo2', 'test/repo3'],
        'language': ['Python', 'Rust', 'Unknown'],
        'description': [
            'A Python project with tests and CI/CD integration',
            'Rust library for high performance computing',
            ''
        ],
        'topics': ['python,testing,ci-cd', 'rust,performance,no-std', ''],
        'stargazers_count': [100, 50, 5],
        'forks_count': [20, 10, 1],
        'open_issues_count': [5, 2, 0],
        'created_at': [
            pd.Timestamp('2024-01-01', tz='UTC'),
            pd.Timestamp('2024-03-01', tz='UTC'),
            pd.Timestamp('2024-06-01', tz='UTC')
        ],
        'pushed_at': [
            pd.Timestamp('2024-05-01', tz='UTC'),
            pd.Timestamp('2024-05-15', tz='UTC'),
            pd.Timestamp('2024-05-30', tz='UTC')
        ]
    }

    df_test = pd.DataFrame(test_data)

    # 创建计算器实例
    calculator = CodeHealthCalculator()

    # 计算分数
    result = calculator.calculate(df_test)

    print(f"\n测试结果:")
    for idx, row in result.iterrows():
        print(f"  {row['full_name']} ({row['language']}): {row['code_health_score']:.2f}")

    # 生成报告
    report = calculator.generate_detailed_report(result, top_n=2)

    print(f"\n报告摘要:")
    print(f"  平均分: {report['summary']['avg_code_health_score']:.2f}")
    print(f"  语言分析: {len(report['language_analysis'])} 种语言")

    return result


if __name__ == "__main__":
    # 运行测试
    test_result = test_code_health_calculator()
    print("\n✅ 代码健康度计算器测试完成")