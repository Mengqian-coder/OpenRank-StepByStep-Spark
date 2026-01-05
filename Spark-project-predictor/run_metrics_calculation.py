#!/usr/bin/env python3
"""
开源项目潜力预测系统 - 指标计算主入口
包含五大维度评分：社区活力、代码健康、维护效率、主题创新、外部吸引力
"""

import pandas as pd
import yaml
import sys
import os
import glob
import numpy as np
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/metrics_calculation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入所有维度计算器
try:
    from src.metrics.community_vitality import CommunityVitalityCalculator
    from src.metrics.code_health import CodeHealthCalculator
    from src.metrics.maintenance_efficiency import MaintenanceEfficiencyCalculator
    from src.metrics.topic_innovation import TopicInnovationCalculator
    from src.metrics.external_appeal import ExternalAppealCalculator

    print("✅ 成功导入所有计算器模块")
except ImportError as e:
    print(f"❌ 导入计算器模块失败: {e}")
    print("请确保已创建以下文件:")
    print("  src/metrics/community_vitality.py")
    print("  src/metrics/code_health.py")
    print("  src/metrics/maintenance_efficiency.py")
    print("  src/metrics/topic_innovation.py")
    print("  src/metrics/external_appeal.py")
    print("  src/metrics/__init__.py")
    sys.exit(1)


def find_data_file():
    """智能查找数据文件"""

    print("🔍 搜索数据文件...")

    # 按优先级尝试的文件路径
    search_patterns = [
        "data/processed/cleaned_repositories.csv",  # 标准位置
        "cleaned_repositories.csv",  # 当前目录
        "data/processed/*clean*.csv",  # 包含clean关键词
        "data/processed/*repository*.csv",  # 包含repository关键词
        "data/*clean*.csv",  # data目录下的clean文件
        "../data/processed/cleaned_repositories.csv",  # 上级目录的标准位置
    ]

    found_files = []

    for pattern in search_patterns:
        if "*" in pattern:
            # 使用通配符搜索
            matches = glob.glob(pattern)
            if matches:
                found_files.extend(matches)
                print(f"  找到匹配模式 {pattern}: {len(matches)} 个文件")
        elif os.path.exists(pattern):
            found_files.append(pattern)
            print(f"  找到文件: {pattern}")

    # 去重并排序（按修改时间，最新的优先）
    if found_files:
        found_files = list(set(found_files))  # 去重
        found_files.sort(key=os.path.getmtime, reverse=True)

        print(f"\n📁 找到以下数据文件:")
        for i, file in enumerate(found_files[:5], 1):  # 只显示前5个
            mtime = datetime.fromtimestamp(os.path.getmtime(file))
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"  {i}. {file}")
            print(f"     大小: {size_mb:.2f} MB, 修改时间: {mtime:%Y-%m-%d %H:%M}")

        return found_files[0]

    # 如果没有找到，列出当前目录结构
    print("\n❌ 未找到数据文件")
    print("\n当前目录结构:")
    for root, dirs, files in os.walk("."):
        level = root.replace(".", "").count(os.sep)
        if level > 2:  # 限制深度
            continue
        indent = " " * 4 * level
        dir_name = os.path.basename(root) if root != "." else "."
        print(f"{indent}{dir_name}/")

        # 显示CSV文件
        subindent = " " * 4 * (level + 1)
        csv_files = [f for f in files if f.endswith(".csv")]
        for file in csv_files[:5]:  # 只显示前5个
            print(f"{subindent}{file}")

    return None


class MetricsPipeline:
    """指标计算流水线"""

    def __init__(self, config_path='config/metrics_config.yaml'):
        self.config = self._load_config(config_path)
        self.calculators = self._initialize_calculators()
        logger.info(f"指标计算流水线初始化完成，包含 {len(self.calculators)} 个维度")

    def _load_config(self, config_path):
        """加载配置文件"""
        try:
            # 使用UTF-8编码打开文件
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            return self._create_default_config()
        except yaml.YAMLError as e:
            logger.error(f"配置文件解析错误: {e}")
            print(f"❌ 配置文件解析错误: {e}")
            return self._create_default_config()
        except UnicodeDecodeError as e:
            logger.error(f"配置文件编码错误: {e}")
            print(f"❌ 配置文件编码错误，尝试使用GBK编码...")
            # 尝试使用GBK编码
            try:
                with open(config_path, 'r', encoding='gbk') as f:
                    config = yaml.safe_load(f)
                logger.info(f"配置文件使用GBK编码加载成功: {config_path}")
                return config
            except Exception as e2:
                logger.error(f"使用GBK编码也失败: {e2}")
                print(f"❌ 使用GBK编码也失败，使用默认配置")
                return self._create_default_config()

    def _create_default_config(self):
        """创建默认配置"""
        default_config = {
            'weights': {
                'community_vitality': 0.25,
                'code_health': 0.20,
                'maintenance_efficiency': 0.15,
                'topic_innovation': 0.20,
                'external_appeal': 0.20
            },
            'normalization': {
                'method': 'minmax',
                'clip_outliers': True,
                'outlier_threshold': 3.0
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/metrics_calculation.log'
            },
            'output': {
                'save_intermediate': True,
                'generate_report': True,
                'report_format': 'both'  # 'text', 'json', 'both'
            }
        }

        # 确保config目录存在
        os.makedirs('config', exist_ok=True)

        config_file = 'config/metrics_config.yaml'
        # 使用UTF-8编码写入文件
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

        print("📝 已创建默认配置文件: config/metrics_config.yaml")
        logger.info("已创建默认配置文件")
        return default_config

    def _initialize_calculators(self):
        """初始化所有计算器"""
        calculators = {
            'community_vitality': CommunityVitalityCalculator(),
            'code_health': CodeHealthCalculator(),
            'maintenance_efficiency': MaintenanceEfficiencyCalculator(),
            'topic_innovation': TopicInnovationCalculator(),
            'external_appeal': ExternalAppealCalculator()
        }

        print(f"🛠️  已初始化 {len(calculators)} 个计算器:")
        for name in calculators.keys():
            print(f"  • {name.replace('_', ' ').title()}")

        return calculators

    def _display_code_health_details(self, df):
        """显示代码健康度详细信息"""
        if 'csq_score' in df.columns and 'epm_score' in df.columns and 'dc_score' in df.columns:
            print(f"      子维度详情:")
            print(f"        代码结构质量: {df['csq_score'].mean():.2f}")
            print(f"        工程实践成熟度: {df['epm_score'].mean():.2f}")
            print(f"        文档完整性: {df['dc_score'].mean():.2f}")

            # 识别前3大语言的代码健康度
            if 'language' in df.columns:
                top_languages = df['language'].value_counts().head(3).index
                for lang in top_languages:
                    if pd.notna(lang):
                        lang_df = df[df['language'] == lang]
                        if len(lang_df) > 0:
                            print(f"        {lang}: {lang_df['code_health_score'].mean():.2f}")

    def _display_community_vitality_details(self, df):
        """显示社区活力详细信息"""
        if 'activity_level_refined' in df.columns:
            print(f"      活跃度分布:")
            for level in ['非常活跃(≤7天)', '活跃(8-30天)', '一般活跃(31-90天)']:
                if level in df['activity_level_refined'].values:
                    count = (df['activity_level_refined'] == level).sum()
                    pct = count / len(df) * 100
                    print(f"        {level}: {count}个项目 ({pct:.1f}%)")

    def _display_maintenance_efficiency_details(self, df):
        """显示维护效率详细信息"""
        if 'ise_score' in df.columns and 'crq_score' in df.columns and 'csd_score' in df.columns:
            print(f"      子维度详情:")
            print(f"        问题解决效率: {df['ise_score'].mean():.2f}")
            print(f"        代码审查质量: {df['crq_score'].mean():.2f}")
            print(f"        协作规范化: {df['csd_score'].mean():.2f}")

            # 分析问题解决时间的分布
            if 'avg_issue_close_time_days' in df.columns:
                median_close_time = df['avg_issue_close_time_days'].median()
                print(f"        问题平均解决时间: {median_close_time:.1f}天")

            # 分析PR合并率的分布
            if 'pr_acceptance_rate' in df.columns:
                avg_acceptance_rate = df['pr_acceptance_rate'].mean() * 100
                print(f"        PR平均接受率: {avg_acceptance_rate:.1f}%")

    def _display_topic_innovation_details(self, df):
        """显示主题创新度详细信息"""
        if 'tc_score' in df.columns and 'ti_score' in df.columns and 'mdm_score' in df.columns:
            print(f"      子维度详情:")
            print(f"        主题集中度: {df['tc_score'].mean():.1f}")
            print(f"        技术创新性: {df['ti_score'].mean():.3f}")
            print(f"        市场需求契合度: {df['mdm_score'].mean():.3f}")

            # 显示主题分析结果
            if 'num_topics' in df.columns:
                print(f"        平均主题数: {df['num_topics'].mean():.1f}")

            if 'main_topic' in df.columns and df['main_topic'].notna().any():
                # 统计最常见的主题
                topic_counts = df['main_topic'].value_counts().head(3)
                print(f"        最常见主题:")
                for topic, count in topic_counts.items():
                    pct = count / len(df) * 100
                    print(f"          • {topic}: {count}个项目 ({pct:.1f}%)")

    def _display_external_appeal_details(self, df):
        """显示外部吸引力详细信息"""
        if 'gr_score' in df.columns and 'vi_score' in df.columns and 'net_score' in df.columns:
            print(f"      子维度详情:")
            print(f"        增长势头: {df['gr_score'].mean():.2f}")
            print(f"        可见性: {df['vi_score'].mean():.2f}")
            print(f"        网络效应: {df['net_score'].mean():.2f}")

            # 显示增长趋势
            if 'star_growth_rate' in df.columns:
                avg_growth = df['star_growth_rate'].mean() * 100
                print(f"        Star平均增长率: {avg_growth:.2f}%")

            # 显示网络效应指标
            if 'dependents_count' in df.columns:
                median_dependents = df['dependents_count'].median()
                print(f"        被依赖数中位数: {median_dependents:.0f}")

    def run(self, input_file, output_file=None):
        """运行指标计算流水线"""

        print("=" * 60)
        print("🚀 开始计算开源项目潜力评分")
        print(f"   输入文件: {input_file}")
        print("=" * 60)

        # 1. 加载数据
        print("\n📂 加载数据...")
        try:
            # 尝试多种编码读取CSV文件
            encodings_to_try = ['utf-8', 'gbk', 'utf-8-sig', 'latin1', 'cp1252']
            df = None

            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(input_file, parse_dates=['created_at', 'updated_at', 'pushed_at'],
                                     encoding=encoding)
                    print(f"✅ 使用 {encoding} 编码成功加载数据")
                    break
                except (UnicodeDecodeError, LookupError) as e:
                    print(f"⚠️  尝试 {encoding} 编码失败: {e}")
                    continue
                except Exception as e:
                    # 可能是其他错误，继续尝试
                    continue

            if df is None:
                print(f"❌ 所有编码尝试都失败，无法读取文件")
                return None

            print(f"✅ 成功加载 {len(df)} 个项目，{len(df.columns)} 个特征")

            # 显示数据基本信息
            if 'created_at' in df.columns:
                print(f"   时间范围: {df['created_at'].min().date()} 至 {df['created_at'].max().date()}")

            if 'language' in df.columns:
                print(f"   语言种类: {df['language'].nunique()} 种")

            if 'stargazers_count' in df.columns:
                print(f"   Star中位数: {df['stargazers_count'].median()}")

            logger.info(f"数据加载成功: {len(df)} 个项目")

        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            print(f"❌ 加载数据失败: {e}")
            print("\n可能的原因:")
            print("  1. 文件路径不正确")
            print("  2. 文件格式不是有效的CSV")
            print("  3. 缺少必要的列（如 created_at, pushed_at 等）")
            print("  4. 文件编码问题")
            return None

        # 2. 按顺序计算各个维度
        print("\n" + "=" * 60)
        print("📊 开始计算各维度评分")
        print("=" * 60)

        results = {}
        dimension_order = list(self.calculators.keys())

        for i, dim_name in enumerate(dimension_order, 1):
            calculator = self.calculators[dim_name]

            print(f"\n[{i}/{len(dimension_order)}] 📈 计算维度: {dim_name.replace('_', ' ').title()}")
            print(f"   计算器: {calculator.__class__.__name__}")

            try:
                # 执行计算
                start_time = datetime.now()
                df = calculator.calculate(df)
                elapsed_time = (datetime.now() - start_time).total_seconds()

                # 获取结果列名
                score_column = f'{dim_name}_score'
                if score_column in df.columns:
                    avg_score = df[score_column].mean()
                    results[dim_name] = {
                        'average_score': avg_score,
                        'min_score': df[score_column].min(),
                        'max_score': df[score_column].max(),
                        'std_score': df[score_column].std()
                    }

                    print(f"   ✅ 计算完成 (耗时: {elapsed_time:.2f}秒)")
                    print(f"      平均分: {avg_score:.2f}")
                    print(f"      范围: {df[score_column].min():.2f} - {df[score_column].max():.2f}")

                    # 对于某些维度，显示额外信息
                    if dim_name == 'code_health':
                        self._display_code_health_details(df)
                    elif dim_name == 'community_vitality':
                        self._display_community_vitality_details(df)
                    elif dim_name == 'maintenance_efficiency':
                        self._display_maintenance_efficiency_details(df)
                    elif dim_name == 'topic_innovation':
                        self._display_topic_innovation_details(df)
                    elif dim_name == 'external_appeal':
                        self._display_external_appeal_details(df)

                    # 保存中间结果（如果配置允许）
                    if self.config.get('output', {}).get('save_intermediate', True):
                        intermediate_file = f"data/processed/intermediate_{dim_name}_scores.csv"
                        intermediate_df = df[['full_name', score_column]].copy()
                        # 使用UTF-8-sig编码保存，兼容Excel
                        intermediate_df.to_csv(intermediate_file, index=False, encoding='utf-8-sig')
                        print(f"      中间结果保存至: {intermediate_file}")

                else:
                    print(f"   ⚠️  计算完成但未找到 {score_column} 列")
                    logger.warning(f"维度 {dim_name} 计算完成但未找到 {score_column} 列")

            except Exception as e:
                logger.error(f"维度 {dim_name} 计算失败: {type(e).__name__}: {e}", exc_info=True)
                print(f"   ❌ 计算失败: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                # 继续执行其他维度

        # 3. 计算综合潜力分（当至少有两个维度完成时）
        print("\n" + "=" * 60)
        print("🧮 计算综合潜力分")
        print("=" * 60)

        completed_dimensions = [dim for dim in dimension_order if f'{dim}_score' in df.columns]

        # 在 run() 方法中修改综合潜力分计算部分：
        if len(completed_dimensions) >= 2:
            df['overall_potential_score'] = 0
            total_weight = 0

            for dim_name in completed_dimensions:
                weight = self.config['weights'].get(dim_name, 0.2)
                score_column = f'{dim_name}_score'

                if score_column in df.columns:
                    # 归一化所有维度分数到0-1范围
                    score = df[score_column].copy()

                    # 根据维度类型进行归一化
                    if dim_name in ['community_vitality', 'code_health', 'maintenance_efficiency']:
                        # 这些维度已经是0-100分制
                        normalized_score = score / 100.0
                    elif dim_name in ['topic_innovation', 'external_appeal']:
                        # 这些维度已经是0-1分制
                        normalized_score = score
                    else:
                        # 默认归一化到0-1
                        min_val = score.min()
                        max_val = score.max()
                        if max_val > min_val:
                            normalized_score = (score - min_val) / (max_val - min_val)
                        else:
                            normalized_score = 0.5

                    df['overall_potential_score'] += weight * normalized_score
                    total_weight += weight

            # 归一化到0-100范围
            if total_weight > 0:
                df['overall_potential_score'] = (df['overall_potential_score'] / total_weight) * 100

            # 确保分数在0-100范围内
            df['overall_potential_score'] = np.clip(df['overall_potential_score'], 0, 100)

            print(f"✅ 综合潜力分计算完成")
            print(f"   基于维度: {', '.join(completed_dimensions)}")
            print(f"   平均综合分: {df['overall_potential_score'].mean():.2f}")
            print(f"   范围: {df['overall_potential_score'].min():.2f} - {df['overall_potential_score'].max():.2f}")

            # 添加综合潜力等级
            df['overall_potential_level'] = pd.cut(
                df['overall_potential_score'],
                bins=[0, 20, 40, 60, 80, 100],
                labels=['潜力低', '潜力一般', '潜力中等', '潜力高', '潜力很高']
            )
        else:
            print(f"⚠️  完成维度不足（{len(completed_dimensions)}个），无法计算综合分")
            print(f"   需要至少2个维度，当前完成: {completed_dimensions}")

        # 4. 保存最终结果
        print("\n" + "=" * 60)
        print("💾 保存结果")
        print("=" * 60)

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"data/processed/scored_repositories_{timestamp}.csv"

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 选择要保存的列（避免数据过大）
        columns_to_save = ['full_name', 'language', 'description', 'stargazers_count',
                           'created_at', 'updated_at', 'pushed_at']

        # 添加所有计算得到的分数列
        score_columns = [col for col in df.columns if col.endswith('_score')]
        columns_to_save.extend(score_columns)

        # 添加主题关键词列
        if 'topic_keywords' in df.columns:
            columns_to_save.append('topic_keywords')
        if 'main_topic' in df.columns:
            columns_to_save.append('main_topic')
        if 'num_topics' in df.columns:
            columns_to_save.append('num_topics')
        if 'topic_entropy' in df.columns:
            columns_to_save.append('topic_entropy')

        # 添加其他有用的列
        other_columns = ['overall_potential_score', 'overall_potential_level']
        columns_to_save.extend([col for col in other_columns if col in df.columns])

        # 确保列存在
        columns_to_save = [col for col in columns_to_save if col in df.columns]

        # 保存数据，使用UTF-8-sig编码（兼容Excel）
        df[columns_to_save].to_csv(output_file, index=False, encoding='utf-8-sig')

        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"✅ 结果保存至: {output_file}")
        print(f"   文件大小: {file_size_mb:.2f} MB")
        print(f"   包含列数: {len(columns_to_save)}")

        # 5. 生成报告
        if self.config.get('output', {}).get('generate_report', True):
            self._generate_report(df, results, output_file, completed_dimensions)

        logger.info(f"指标计算流水线执行完成，结果保存至: {output_file}")
        return df

    def _generate_report(self, df, results, output_file, completed_dimensions):
        """生成分析报告"""
        print("\n" + "=" * 60)
        print("📈 指标计算分析报告")
        print("=" * 60)

        print(f"\n📊 数据集概况:")
        print(f"   项目总数: {len(df)}")

        if 'created_at' in df.columns:
            try:
                print(f"   时间范围: {df['created_at'].min().date()} 至 {df['created_at'].max().date()}")

                avg_age_days = (pd.Timestamp.now() - df['created_at']).dt.days.mean()
                avg_age_months = avg_age_days / 30.44
                print(f"   平均项目年龄: {avg_age_months:.1f} 个月 ({avg_age_days:.0f} 天)")
            except:
                pass

        # 各维度评分统计
        print(f"\n📋 各维度评分统计:")
        for dim_name, stats in results.items():
            dim_display = dim_name.replace('_', ' ').title()
            print(f"   {dim_display}:")
            print(f"     平均分: {stats['average_score']:.2f}")
            print(f"     范围: {stats['min_score']:.2f} - {stats['max_score']:.2f}")
            print(f"     标准差: {stats['std_score']:.2f}")

        # 综合潜力排名
        if 'overall_potential_score' in df.columns:
            print(f"\n🏆 综合潜力排名前10:")
            top_10 = df.nlargest(10, 'overall_potential_score')[
                ['full_name', 'language', 'stargazers_count', 'overall_potential_score']
            ]

            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                # 确保显示完整项目名
                name = str(row['full_name'])
                if len(name) > 40:
                    name = name[:37] + "..."

                lang = str(row['language']) if pd.notna(row['language']) else "Unknown"
                stars = int(row['stargazers_count'])
                score = row['overall_potential_score']

                print(f"   {i:2d}. {name:<40} {lang:>12} {stars:>5}⭐ {score:>6.2f}分")

            # 各语言表现
            if 'language' in df.columns:
                print(f"\n🌍 各语言表现（基于综合潜力分）:")
                lang_stats = []
                for lang in df['language'].unique():
                    if pd.notna(lang) and lang != 'Unknown':
                        lang_df = df[df['language'] == lang]
                        if len(lang_df) >= 3:  # 至少有3个项目才统计
                            avg_score = lang_df['overall_potential_score'].mean()
                            lang_stats.append((lang, avg_score, len(lang_df)))

                # 按平均分排序
                lang_stats.sort(key=lambda x: x[1], reverse=True)

                for i, (lang, avg_score, count) in enumerate(lang_stats[:5], 1):
                    print(f"   {i:2d}. {lang:<15} {avg_score:>6.2f}分 ({count:>3}个项目)")

        # 改进建议
        print(f"\n💡 改进建议:")

        # 检查数据缺失
        missing_fields = []
        for field in ['language', 'description', 'topics']:
            if field in df.columns and df[field].isnull().mean() > 0.1:
                missing_pct = df[field].isnull().mean() * 100
                missing_fields.append(f"{field} ({missing_pct:.1f}%缺失)")

        if missing_fields:
            print(f"   1. 数据质量: 需要处理缺失字段 - {', '.join(missing_fields)}")

        # 检查是否需要更多数据
        if len(completed_dimensions) < 5:
            print(f"   2. 维度覆盖: 当前完成{len(completed_dimensions)}/5个维度")

        # 检查是否需要平衡采样
        if 'language' in df.columns:
            lang_counts = df['language'].value_counts()
            if len(lang_counts) > 0:
                top_lang_pct = lang_counts.iloc[0] / len(df) * 100
                if top_lang_pct > 50:
                    print(f"   3. 样本平衡: {lang_counts.index[0]}占比过高 ({top_lang_pct:.1f}%)，建议补充其他语言项目")

        print(f"\n📁 输出文件位置:")
        print(f"   完整数据: {output_file}")

        # 列出所有中间文件
        intermediate_files = glob.glob("data/processed/intermediate_*.csv")
        if intermediate_files:
            print(f"   中间文件:")
            for file in intermediate_files[:5]:  # 只显示前5个
                try:
                    file_size_kb = os.path.getsize(file) / 1024
                    print(f"     • {os.path.basename(file)} ({file_size_kb:.1f} KB)")
                except:
                    print(f"     • {os.path.basename(file)}")


def main():
    """主程序入口"""

    print("=" * 70)
    print("📊 开源项目潜力预测系统 - 指标计算模块")
    print("=" * 70)

    # 查找数据文件
    data_file = find_data_file()

    if data_file is None:
        print("\n❌ 无法自动找到数据文件")
        print("\n请执行以下操作之一:")
        print("  1. 将数据文件放入 data/processed/ 目录")
        print("  2. 修改下面的 manual_file_path 变量指定文件路径")

        # 手动指定文件路径（如果需要）
        manual_file_path = "data/processed/cleaned_repositories.csv"

        if os.path.exists(manual_file_path):
            print(f"\n✅ 使用手动指定的文件: {manual_file_path}")
            data_file = manual_file_path
        else:
            print(f"\n❌ 手动指定的文件也不存在: {manual_file_path}")
            print("\n请确保:")
            print("  • 数据文件已正确放置")
            print("  • 文件包含以下列: created_at, updated_at, pushed_at, language, description, topics")
            sys.exit(1)

    print(f"\n✅ 使用数据文件: {data_file}")

    # 创建并运行流水线
    try:
        pipeline = MetricsPipeline()

        print("\n" + "=" * 70)
        print("🚀 开始指标计算流水线")
        print("=" * 70)

        result_df = pipeline.run(input_file=data_file)

        if result_df is not None:
            print("\n" + "=" * 70)
            print("🎉 指标计算流水线执行完成!")
            print("=" * 70)

            # 显示最终统计
            total_dimensions = len(
                [col for col in result_df.columns if col.endswith('_score') and not col.startswith('overall')])

            print(f"\n📈 计算完成统计:")
            print(f"   总项目数: {len(result_df)}")
            print(f"   完成维度: {total_dimensions}个")

            if 'overall_potential_score' in result_df.columns:
                print(
                    f"   综合潜力分范围: {result_df['overall_potential_score'].min():.2f} - {result_df['overall_potential_score'].max():.2f}")
                print(f"   综合潜力分中位数: {result_df['overall_potential_score'].median():.2f}")

                # 显示潜力分布
                if 'overall_potential_level' in result_df.columns:
                    level_counts = result_df['overall_potential_level'].value_counts()
                    print(f"\n📊 综合潜力分布:")
                    for level, count in level_counts.items():
                        pct = count / len(result_df) * 100
                        print(f"   • {level}: {count}个项目 ({pct:.1f}%)")


            # 生成配置文件说明
            config_file = 'config/metrics_config.yaml'
            if os.path.exists(config_file):
                print(f"\n⚙️  配置文件位置: {config_file}")
                print("   可以修改此文件调整各维度权重:")
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    if config and 'weights' in config:
                        print("   当前权重设置:")
                        for dim, weight in config['weights'].items():
                            dim_name = dim.replace('_', ' ').title()
                            print(f"     • {dim_name}: {weight}")
                except Exception as e:
                    print(f"   无法读取配置文件: {e}")

        else:
            print("\n❌ 流水线执行失败")

    except KeyboardInterrupt:
        print("\n⚠️  用户中断执行")
        logger.info("用户中断执行")
    except Exception as e:
        logger.error(f"执行过程中出现错误: {type(e).__name__}: {e}", exc_info=True)
        print(f"\n❌ 执行过程中出现错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 如果有命令行参数，第一个参数作为数据文件路径
        data_file_arg = sys.argv[1]
        if os.path.exists(data_file_arg):
            print(f"使用命令行参数指定的文件: {data_file_arg}")

            # 创建流水线并运行
            pipeline = MetricsPipeline()
            result_df = pipeline.run(input_file=data_file_arg)
        else:
            print(f"❌ 指定的文件不存在: {data_file_arg}")
            sys.exit(1)
    else:
        # 没有命令行参数，执行主程序
        main()