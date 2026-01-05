#!/usr/bin/env python3
"""
主题创新度维度计算器
包含三个子维度：主题集中度(TC)、技术创新性(TI)、市场需求契合度(MDM)
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import logging
from collections import Counter
import math
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

try:
    # 尝试导入BERTopic及相关依赖
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN

    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("⚠️ BERTopic及相关依赖未安装，主题集中度分析将使用简化方法")
    print("   如需完整功能，请安装: pip install bertopic sentence-transformers umap-learn hdbscan")


class TopicInnovationCalculator:
    """
    主题创新度计算器
    评估项目的技术方向专注度、创新性和市场匹配度
    """

    def __init__(self, config=None):
        """
        初始化计算器

        Args:
            config: 配置字典，可自定义权重和阈值
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)

        # 子维度权重配置
        self.weights = self.config.get('weights', {
            'topic_concentration': 0.40,  # 主题集中度
            'technical_innovation': 0.35,  # 技术创新性
            'market_demand_match': 0.25  # 市场需求契合度
        })

        # TI子维度权重
        self.ti_weights = self.config.get('ti_weights', {
            'novel_tech_stack': 0.4,
            'research_connection': 0.3,
            'problem_novelty': 0.3
        })

        # MDM子维度权重
        self.mdm_weights = self.config.get('mdm_weights', {
            'topic_trend': 0.4,
            'industry_application': 0.4,
            'search_trend': 0.2
        })

        # 技术栈新颖度配置
        self.tech_adoption_curve = self.config.get('tech_adoption_curve', {
            'emerging': ['WebAssembly', 'Rust', 'Wasm', 'Edge Computing', 'Blockchain'],
            'growth': ['Kubernetes', 'TensorFlow', 'React Native', 'GraphQL', 'Serverless'],
            'mature': ['JavaScript', 'Python', 'Java', 'MySQL', 'jQuery'],
            'declining': ['Flash', 'jQuery UI', 'Backbone.js', 'AngularJS', 'CoffeeScript']
        })

        # 学术关键词
        self.research_keywords = self.config.get('research_keywords', [
            'paper', 'research', 'arxiv', 'conference', 'journal', 'thesis',
            'preprint', 'citation', 'peer-reviewed', 'academic', 'scientific',
            'experiment', 'methodology', 'evaluation', 'benchmark', 'survey'
        ])

        # 行业应用关键词
        self.industry_keywords = self.config.get('industry_keywords', {
            'ai_ml': ['artificial intelligence', 'machine learning', 'deep learning', 'neural network', 'ai'],
            'cloud': ['cloud computing', 'aws', 'azure', 'gcp', 'kubernetes', 'docker'],
            'data_science': ['data science', 'big data', 'analytics', 'visualization'],
            'web': ['web development', 'frontend', 'backend', 'fullstack'],
            'mobile': ['mobile app', 'ios', 'android', 'flutter', 'react native'],
            'devops': ['devops', 'ci/cd', 'automation', 'infrastructure'],
            'blockchain': ['blockchain', 'crypto', 'web3', 'smart contract'],
            'iot': ['internet of things', 'iot', 'embedded', 'raspberry pi']
        })

        # BERTopic模型（延迟加载）
        self.topic_model = None
        self.sentence_model = None

    def _get_default_config(self):
        """获取默认配置"""
        return {
            'weights': {
                'topic_concentration': 0.40,
                'technical_innovation': 0.35,
                'market_demand_match': 0.25
            },
            'ti_weights': {
                'novel_tech_stack': 0.4,
                'research_connection': 0.3,
                'problem_novelty': 0.3
            },
            'mdm_weights': {
                'topic_trend': 0.4,
                'industry_application': 0.4,
                'search_trend': 0.2
            },
            'enable_logging': True,
            'use_bertopic': True,  # 是否使用BERTopic进行主题分析
            'fallback_to_simple': True  # BERTopic不可用时是否使用简化方法
        }

    def calculate(self, df):
        """
        计算主题创新度分数

        Args:
            df: 包含项目数据的DataFrame

        Returns:
            添加了主题创新度相关列的DataFrame
        """
        self.logger.info("开始计算主题创新度维度评分...")

        # 复制数据避免修改原数据
        result_df = df.copy()

        # 1. 计算主题集中度 (Topic Concentration)
        self.logger.info("计算主题集中度...")
        result_df = self._calculate_topic_concentration(result_df)

        # 2. 计算技术创新性 (Technical Innovation)
        self.logger.info("计算技术创新性...")
        result_df = self._calculate_technical_innovation(result_df)

        # 3. 计算市场需求契合度 (Market Demand Match)
        self.logger.info("计算市场需求契合度...")
        result_df = self._calculate_market_demand_match(result_df)

        # 4. 计算综合主题创新度分数
        self.logger.info("计算综合主题创新度分数...")
        result_df = self._calculate_topic_innovation_score(result_df)

        # 5. 添加创新度等级标签
        result_df = self._add_innovation_level(result_df)

        self.logger.info(f"主题创新度计算完成，共处理 {len(result_df)} 个项目")

        return result_df

    def _calculate_topic_concentration(self, df):
        """
        计算主题集中度 (Topic Concentration)

        基于信息熵衡量项目技术方向的专注程度
        TC_Score = 100 × (1 - Normalized_Entropy)
        """
        # 检查是否有文本数据
        text_columns = ['description', 'readme_content', 'topics']
        available_cols = [col for col in text_columns if col in df.columns]

        if not available_cols:
            self.logger.warning("没有找到文本数据字段，主题集中度使用默认值")
            df['tc_score'] = 50  # 默认中等分数
            df['topic_entropy'] = 0.5
            df['num_topics'] = 3
            return df

        # 合并文本数据
        df['combined_text'] = df[available_cols[0]].fillna('')
        for col in available_cols[1:]:
            df['combined_text'] += ' ' + df[col].fillna('')

        # 使用BERTopic或简化方法
        if self.config.get('use_bertopic', True) and BERTOPIC_AVAILABLE:
            self.logger.info("使用BERTopic进行主题分析...")
            df = self._analyze_topics_with_bertopic(df)
        else:
            self.logger.info("使用简化方法进行主题分析...")
            df = self._analyze_topics_simple(df)

        # 计算主题集中度分数
        # TC_Score = 100 × (1 - Normalized_Entropy)
        # 熵值越低（主题越集中），分数越高
        if 'topic_entropy' in df.columns:
            # 归一化熵值（0-1范围）
            max_entropy = df['topic_entropy'].max()
            min_entropy = df['topic_entropy'].min()

            if max_entropy > min_entropy:
                normalized_entropy = (df['topic_entropy'] - min_entropy) / (max_entropy - min_entropy)
            else:
                normalized_entropy = 0.5  # 所有项目熵值相同时使用默认值

            df['tc_score'] = 100 * (1 - normalized_entropy)
        else:
            df['tc_score'] = 50  # 默认值

        return df

    def _analyze_topics_with_bertopic(self, df):
        """使用BERTopic进行主题分析"""
        try:
            # 检查是否应该使用BERTopic
            if not self.config.get('topic_innovation', {}).get('use_bertopic', True):
                self.logger.info("配置为不使用BERTopic，使用简化方法")
                return self._analyze_topics_simple(df)

            # 设置超时和离线模式
            import os
            os.environ['HF_HUB_OFFLINE'] = '1'  # 离线模式
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # 启用更快传输

            # 设置超时时间
            import requests
            requests.adapters.DEFAULT_TIMEOUT = 10  # 10秒超时
            import urllib3
            urllib3.disable_warnings()  # 禁用SSL警告

            # 准备文本数据
            texts = df['combined_text'].fillna('').astype(str).tolist()

            # 如果文本太短，添加一些占位符
            texts = [text if len(text) > 10 else f"project about {text}" for text in texts]

            # 使用更简单的主题模型配置
            self.logger.info("使用轻量级BERTopic配置...")

            # 使用更小、更快的模型
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.decomposition import LatentDirichletAllocation

            # 使用LDA作为后备方案，避免下载大型模型
            try:
                # 尝试使用BERTopic，如果失败则使用LDA
                if self.topic_model is None:
                    self.logger.info("尝试初始化BERTopic模型...")

                    # 使用轻量级配置
                    from umap import UMAP
                    from hdbscan import HDBSCAN

                    umap_model = UMAP(
                        n_neighbors=5,  # 减少邻居数
                        n_components=3,  # 减少维度
                        min_dist=0.0,
                        metric='cosine',
                        random_state=42
                    )

                    hdbscan_model = HDBSCAN(
                        min_cluster_size=5,  # 增加最小簇大小
                        metric='euclidean',
                        cluster_selection_method='eom',
                        prediction_data=True
                    )

                    # 使用CountVectorizer而不是sentence-transformers，避免下载
                    vectorizer_model = CountVectorizer(
                        stop_words="english",
                        max_features=1000  # 限制特征数量
                    )

                    self.topic_model = BERTopic(
                        umap_model=umap_model,
                        hdbscan_model=hdbscan_model,
                        vectorizer_model=vectorizer_model,
                        language="english",
                        calculate_probabilities=True,
                        verbose=False
                    )

                # 训练模型
                self.logger.info("训练主题模型...")
                topics, probabilities = self.topic_model.fit_transform(texts)

                # 计算每个项目的主题分布熵
                entropy_scores = []
                num_topics_list = []

                for prob in probabilities:
                    if prob is not None and len(prob) > 0:
                        # 过滤掉接近零的概率
                        prob_filtered = prob[prob > 0.01]
                        if len(prob_filtered) > 0:
                            # 归一化概率
                            prob_filtered = prob_filtered / prob_filtered.sum()
                            # 计算香农熵
                            entropy = -np.sum(prob_filtered * np.log2(prob_filtered + 1e-10))
                            entropy_scores.append(entropy)
                            num_topics_list.append(len(prob_filtered))
                        else:
                            entropy_scores.append(0)
                            num_topics_list.append(1)
                    else:
                        entropy_scores.append(0)
                        num_topics_list.append(1)

                df['topic_entropy'] = entropy_scores
                df['num_topics'] = num_topics_list

                # 获取主题关键词
                topic_info = self.topic_model.get_topic_info()
                if not topic_info.empty:
                    # 为每个项目添加主要主题标签
                    main_topics = []
                    for i, topic_id in enumerate(topics):
                        if topic_id != -1:  # -1表示异常值
                            topic_row = topic_info[topic_info['Topic'] == topic_id]
                            if not topic_row.empty:
                                keywords = eval(topic_row.iloc[0]['Representation'])
                                main_topics.append(', '.join([kw[0] for kw in keywords[:3]]))
                            else:
                                main_topics.append('General')
                        else:
                            main_topics.append('Diverse')

                    df['main_topic'] = main_topics

                return df

            except Exception as bertopic_error:
                self.logger.warning(f"BERTopic失败，使用LDA后备方案: {bertopic_error}")
                return self._fallback_to_lda(df, texts)

        except Exception as e:
            self.logger.error(f"主题分析完全失败: {e}")
            self.logger.info("回退到简化方法...")
            return self._analyze_topics_simple(df)

    def _fallback_to_lda(self, df, texts):
        """使用LDA作为后备方案"""
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.decomposition import LatentDirichletAllocation

            # 创建文档-词矩阵
            vectorizer = CountVectorizer(
                max_df=0.95,
                min_df=2,
                stop_words='english'
            )
            doc_term_matrix = vectorizer.fit_transform(texts)

            # 应用LDA
            lda = LatentDirichletAllocation(
                n_components=10,  # 主题数
                random_state=42,
                max_iter=10
            )
            lda.fit(doc_term_matrix)

            # 获取主题分布
            topic_distribution = lda.transform(doc_term_matrix)

            # 计算熵
            entropy_scores = []
            num_topics_list = []

            for dist in topic_distribution:
                # 过滤小概率
                dist_filtered = dist[dist > 0.01]
                if len(dist_filtered) > 0:
                    dist_filtered = dist_filtered / dist_filtered.sum()
                    entropy = -np.sum(dist_filtered * np.log2(dist_filtered + 1e-10))
                    entropy_scores.append(entropy)
                    num_topics_list.append(len(dist_filtered))
                else:
                    entropy_scores.append(0)
                    num_topics_list.append(1)

            df['topic_entropy'] = entropy_scores
            df['num_topics'] = num_topics_list

            # 获取主题关键词
            feature_names = vectorizer.get_feature_names_out()
            topics_keywords = []

            for idx, topic in enumerate(lda.components_):
                top_keywords_idx = topic.argsort()[-5:][::-1]
                top_keywords = [feature_names[i] for i in top_keywords_idx]
                topics_keywords.append(top_keywords)

            # 为每个文档分配主要主题
            main_topics = []
            for i, dist in enumerate(topic_distribution):
                main_topic_idx = np.argmax(dist)
                if dist[main_topic_idx] > 0.3:  # 阈值
                    main_topics.append(', '.join(topics_keywords[main_topic_idx][:3]))
                else:
                    main_topics.append('Diverse')

            df['main_topic'] = main_topics

            return df

        except Exception as e:
            self.logger.error(f"LDA后备方案也失败: {e}")
            # 返回简化方法
            return self._analyze_topics_simple(df)

    def _analyze_topics_simple(self, df):
        """使用简化方法进行主题分析"""
        try:
            # 提取文本中的关键词作为"主题"
            df['topic_entropy'] = 0.5  # 默认熵值
            df['num_topics'] = 3  # 默认主题数

            # 简单计算：基于描述长度和技术关键词多样性
            for idx, row in df.iterrows():
                text = str(row.get('combined_text', '')).lower()

                if len(text) < 20:
                    df.at[idx, 'topic_entropy'] = 0.8  # 文本太短，假设主题分散
                    df.at[idx, 'num_topics'] = 1
                    continue

                # 提取技术关键词
                tech_keywords = []

                # 检查常见技术术语
                common_tech_terms = [
                    'web', 'mobile', 'data', 'cloud', 'ai', 'machine learning',
                    'blockchain', 'iot', 'security', 'database', 'api', 'framework',
                    'library', 'tool', 'application', 'platform', 'system'
                ]

                for term in common_tech_terms:
                    if term in text:
                        tech_keywords.append(term)

                # 计算"熵"的简化版本
                num_keywords = len(tech_keywords)
                if num_keywords <= 1:
                    entropy = 0.2  # 主题集中
                elif num_keywords <= 3:
                    entropy = 0.5  # 中等集中
                elif num_keywords <= 5:
                    entropy = 0.7  # 主题分散
                else:
                    entropy = 0.9  # 非常分散

                df.at[idx, 'topic_entropy'] = entropy
                df.at[idx, 'num_topics'] = max(1, num_keywords)

            return df

        except Exception as e:
            self.logger.error(f"简化主题分析失败: {e}")
            # 设置默认值
            df['topic_entropy'] = 0.5
            df['num_topics'] = 3
            return df

    def _calculate_technical_innovation(self, df):
        """
        计算技术创新性 (Technical Innovation)
        TI = Novel_Tech_Stack_Score × 0.4 +
             Research_Connection_Score × 0.3 +
             Problem_Novelty_Score × 0.3
        """
        # 1. 技术栈新颖度评分
        self.logger.info("计算技术栈新颖度...")
        df['novel_tech_stack_score'] = self._calculate_novel_tech_stack_score(df)

        # 2. 学术研究关联度评分
        self.logger.info("计算学术研究关联度...")
        df['research_connection_score'] = self._calculate_research_connection_score(df)

        # 3. 问题新颖性评分
        self.logger.info("计算问题新颖性...")
        df['problem_novelty_score'] = self._calculate_problem_novelty_score(df)

        # 计算技术创新性综合分数
        df['ti_score'] = (
                df['novel_tech_stack_score'] * self.ti_weights['novel_tech_stack'] +
                df['research_connection_score'] * self.ti_weights['research_connection'] +
                df['problem_novelty_score'] * self.ti_weights['problem_novelty']
        )

        # 确保分数在0-1范围内
        df['ti_score'] = np.clip(df['ti_score'], 0, 1)

        return df

    def _calculate_novel_tech_stack_score(self, df):
        """计算技术栈新颖度"""
        scores = []

        for _, row in df.iterrows():
            score = 0.5  # 基础分数

            # 检查语言字段
            language = str(row.get('language', '')).lower()

            # 基于技术采用曲线评分
            if language:
                # 新兴技术
                if any(tech.lower() in language for tech in self.tech_adoption_curve['emerging']):
                    score = 0.9
                # 成长期技术
                elif any(tech.lower() in language for tech in self.tech_adoption_curve['growth']):
                    score = 0.7
                # 成熟技术
                elif any(tech.lower() in language for tech in self.tech_adoption_curve['mature']):
                    score = 0.5
                # 衰退技术
                elif any(tech.lower() in language for tech in self.tech_adoption_curve['declining']):
                    score = 0.3
                # 其他技术
                else:
                    score = 0.6

            # 检查项目描述中的技术关键词
            description = str(row.get('description', '')).lower()
            topics = str(row.get('topics', '')).lower()
            combined_text = description + ' ' + topics

            # 统计新兴技术关键词出现次数
            emerging_count = 0
            for tech in self.tech_adoption_curve['emerging']:
                if tech.lower() in combined_text:
                    emerging_count += 1

            # 根据新兴技术数量调整分数
            if emerging_count >= 3:
                score = min(1.0, score + 0.3)
            elif emerging_count >= 2:
                score = min(1.0, score + 0.2)
            elif emerging_count >= 1:
                score = min(1.0, score + 0.1)

            scores.append(score)

        return scores

    def _calculate_research_connection_score(self, df):
        """计算学术研究关联度"""
        scores = []

        for _, row in df.iterrows():
            score = 0.3  # 基础分数（大多数项目与学术研究关联不强）

            # 检查文本中的学术关键词
            text_fields = ['description', 'readme_content', 'name', 'full_name']
            combined_text = ' '.join([str(row.get(field, '')).lower() for field in text_fields if field in row])

            # 统计学术关键词出现次数
            research_count = 0
            for keyword in self.research_keywords:
                if keyword.lower() in combined_text:
                    research_count += 1

            # 根据学术关键词数量调整分数
            if research_count >= 5:
                score = 0.9
            elif research_count >= 3:
                score = 0.7
            elif research_count >= 2:
                score = 0.5
            elif research_count >= 1:
                score = 0.4

            # 检查是否有明显的学术引用模式
            if 'arxiv' in combined_text or 'citation' in combined_text or 'doi:' in combined_text:
                score = max(score, 0.8)

            scores.append(score)

        return scores

    def _calculate_problem_novelty_score(self, df):
        """计算问题新颖性"""
        scores = []

        # 常见问题模式（越常见，新颖性越低）
        common_problem_patterns = [
            'web framework', 'todo app', 'calculator', 'weather app',
            'blog system', 'e-commerce', 'chat application', 'file manager',
            'task manager', 'note taking', 'url shortener'
        ]

        for _, row in df.iterrows():
            score = 0.5  # 基础分数

            description = str(row.get('description', '')).lower()
            name = str(row.get('name', '')).lower()

            # 检查是否为常见问题模式
            is_common = False
            for pattern in common_problem_patterns:
                if pattern in description or pattern in name:
                    is_common = True
                    break

            if is_common:
                score = 0.3  # 常见问题，新颖性较低
            else:
                # 检查创新性词汇
                innovative_words = [
                    'revolutionary', 'novel', 'innovative', 'breakthrough',
                    'new approach', 'unique', 'first of its kind', 'pioneering',
                    'groundbreaking', 'cutting-edge'
                ]

                innovative_count = sum(1 for word in innovative_words if word in description)
                if innovative_count >= 2:
                    score = 0.8
                elif innovative_count >= 1:
                    score = 0.6
                else:
                    score = 0.5  # 中等新颖性

            scores.append(score)

        return scores

    def _calculate_market_demand_match(self, df):
        """
        计算市场需求契合度 (Market Demand Match)
        MDM = w₁ × Topic_Trend_Score +
              w₂ × Industry_Application_Score +
              w₃ × Search_Trend_Correlation
        """
        # 1. 主题趋势评分
        self.logger.info("计算主题趋势评分...")
        df['topic_trend_score'] = self._calculate_topic_trend_score(df)

        # 2. 行业应用潜力评分
        self.logger.info("计算行业应用潜力评分...")
        df['industry_application_score'] = self._calculate_industry_application_score(df)

        # 3. 搜索趋势相关性（简化版）
        self.logger.info("计算搜索趋势相关性...")
        df['search_trend_correlation'] = self._calculate_search_trend_correlation(df)

        # 计算市场需求契合度综合分数
        df['mdm_score'] = (
                df['topic_trend_score'] * self.mdm_weights['topic_trend'] +
                df['industry_application_score'] * self.mdm_weights['industry_application'] +
                df['search_trend_correlation'] * self.mdm_weights['search_trend']
        )

        # 确保分数在0-1范围内
        df['mdm_score'] = np.clip(df['mdm_score'], 0, 1)

        return df

    def _calculate_topic_trend_score(self, df):
        """计算主题趋势评分"""
        # 当前热门技术趋势（可以定期更新）
        current_hot_topics = [
            'artificial intelligence', 'machine learning', 'deep learning',
            'large language models', 'llm', 'generative ai',
            'web3', 'blockchain', 'cryptocurrency',
            'edge computing', 'iot', 'internet of things',
            'devops', 'kubernetes', 'docker', 'cloud native',
            'data science', 'big data', 'analytics',
            'cybersecurity', 'privacy', 'security'
        ]

        scores = []

        for _, row in df.iterrows():
            score = 0.5  # 基础分数

            # 合并所有文本字段
            text_fields = ['description', 'topics', 'readme_content', 'name']
            combined_text = ' '.join([str(row.get(field, '')).lower() for field in text_fields if field in row])

            # 检查与热门主题的匹配度
            matched_topics = []
            for topic in current_hot_topics:
                if topic in combined_text:
                    matched_topics.append(topic)

            # 根据匹配的热门主题数量评分
            num_matched = len(matched_topics)
            if num_matched >= 3:
                score = 0.9
            elif num_matched == 2:
                score = 0.7
            elif num_matched == 1:
                score = 0.6

            # 额外检查GitHub Topics
            if 'topics' in row:
                topics_str = str(row['topics']).lower()
                if 'ai' in topics_str or 'machine-learning' in topics_str or 'llm' in topics_str:
                    score = max(score, 0.8)

            scores.append(score)

        return scores

    def _calculate_industry_application_score(self, df):
        """计算行业应用潜力评分"""
        scores = []

        # 行业权重（可以根据市场需求调整）
        industry_weights = {
            'ai_ml': 0.9,  # AI/ML 当前需求最高
            'cloud': 0.8,  # 云计算
            'data_science': 0.7,  # 数据科学
            'web': 0.6,  # Web开发
            'devops': 0.7,  # DevOps
            'blockchain': 0.6,  # 区块链
            'mobile': 0.6,  # 移动开发
            'iot': 0.5  # IoT
        }

        for _, row in df.iterrows():
            score = 0.4  # 基础分数

            # 合并所有文本字段
            text_fields = ['description', 'topics', 'readme_content']
            combined_text = ' '.join([str(row.get(field, '')).lower() for field in text_fields if field in row])

            # 检查与各行业的匹配度
            matched_industries = []
            for industry, keywords in self.industry_keywords.items():
                for keyword in keywords:
                    if keyword in combined_text:
                        matched_industries.append(industry)
                        break  # 匹配到一个关键词即可

            # 计算加权分数
            if matched_industries:
                industry_scores = [industry_weights.get(ind, 0.5) for ind in matched_industries]
                score = max(industry_scores)  # 取最高分（项目可能属于多个行业）

            scores.append(score)

        return scores

    def _calculate_search_trend_correlation(self, df):
        """计算搜索趋势相关性（简化版）"""
        # 在实际应用中，这里应该调用Google Trends API或类似服务
        # 这里使用基于项目创建时间和流行度的简化估计

        scores = []

        for idx, row in df.iterrows():
            score = 0.5  # 基础分数

            # 基于项目年龄：较新的项目可能更符合当前趋势
            if 'created_at' in row:
                try:
                    created_date = pd.to_datetime(row['created_at'])
                    days_old = (pd.Timestamp.now() - created_date).days

                    if days_old < 180:  # 半年内创建
                        score = 0.7
                    elif days_old < 365:  # 一年内创建
                        score = 0.6
                    else:
                        score = 0.4  # 较老的项目可能趋势相关性较低
                except:
                    pass

            # 基于星标增长：快速增长可能表明符合趋势
            if 'stargazers_count' in row and 'created_at' in row:
                try:
                    stars = float(row['stargazers_count'])
                    created_date = pd.to_datetime(row['created_at'])
                    days_old = max(1, (pd.Timestamp.now() - created_date).days)

                    # 计算日均星标数
                    stars_per_day = stars / days_old

                    if stars_per_day > 1.0:
                        score = min(1.0, score + 0.3)
                    elif stars_per_day > 0.5:
                        score = min(1.0, score + 0.2)
                    elif stars_per_day > 0.1:
                        score = min(1.0, score + 0.1)
                except:
                    pass

            scores.append(score)

        return scores

    def _calculate_topic_innovation_score(self, df):
        """
        计算综合主题创新度分数
        主题创新度 = 0.40 × TC_Score + 0.35 × TI + 0.25 × MDM
        """
        # 确保所有子维度分数都存在
        required_scores = ['tc_score', 'ti_score', 'mdm_score']

        for score in required_scores:
            if score not in df.columns:
                self.logger.warning(f"缺少子维度分数: {score}，使用默认值")
                if score == 'tc_score':
                    df[score] = 50  # TC是百分制
                else:
                    df[score] = 0.5

        # 归一化TC分数到0-1范围
        df['tc_normalized'] = df['tc_score'] / 100.0

        # 计算加权综合分数
        df['topic_innovation_score'] = (
                df['tc_normalized'] * self.weights['topic_concentration'] +
                df['ti_score'] * self.weights['technical_innovation'] +
                df['mdm_score'] * self.weights['market_demand_match']
        )

        # 转换为百分制（可选）
        df['topic_innovation_score_pct'] = df['topic_innovation_score'] * 100

        return df

    def _add_innovation_level(self, df):
        """添加创新度等级标签"""
        if 'topic_innovation_score' not in df.columns:
            df['topic_innovation_level'] = '未知'
            return df

        # 定义等级阈值
        conditions = [
            df['topic_innovation_score'] >= 0.8,
            df['topic_innovation_score'] >= 0.6,
            df['topic_innovation_score'] >= 0.4,
            df['topic_innovation_score'] >= 0.2
        ]

        choices = ['高度创新', '中度创新', '一般创新', '创新不足', '缺乏创新']

        df['topic_innovation_level'] = np.select(conditions, choices[:4], default=choices[4])

        return df

    def get_dimension_summary(self, df):
        """
        获取主题创新度维度统计摘要

        Args:
            df: 包含主题创新度分数的DataFrame

        Returns:
            统计摘要字典
        """
        if 'topic_innovation_score' not in df.columns:
            return {"error": "未找到主题创新度分数，请先运行calculate方法"}

        summary = {
            "维度名称": "主题创新度",
            "项目总数": len(df),
            "平均分": df['topic_innovation_score'].mean(),
            "中位数": df['topic_innovation_score'].median(),
            "标准差": df['topic_innovation_score'].std(),
            "最低分": df['topic_innovation_score'].min(),
            "最高分": df['topic_innovation_score'].max(),
            "子维度平均分": {
                "主题集中度": df['tc_score'].mean() if 'tc_score' in df.columns else 0,
                "技术创新性": df['ti_score'].mean() if 'ti_score' in df.columns else 0,
                "市场需求契合度": df['mdm_score'].mean() if 'mdm_score' in df.columns else 0
            },
            "等级分布": df['topic_innovation_level'].value_counts().to_dict()
            if 'topic_innovation_level' in df.columns else {}
        }

        return summary


# 测试函数
def test_topic_innovation_calculator():
    """测试主题创新度计算器"""
    # 创建测试数据
    test_data = {
        'full_name': ['test/ai-project', 'test/web-framework', 'test/data-tool'],
        'description': [
            'A revolutionary AI model using novel neural architecture for language understanding with research paper citations.',
            'A simple web framework for building REST APIs with JavaScript and Node.js.',
            'Data visualization tool for big data analytics with machine learning integration.'
        ],
        'language': ['Python', 'JavaScript', 'Python'],
        'topics': ['ai, machine-learning, deep-learning', 'web, javascript, api',
                   'data-science, visualization, python'],
        'stargazers_count': [1500, 200, 500],
        'created_at': ['2023-01-01', '2022-06-01', '2023-03-15']
    }

    df = pd.DataFrame(test_data)

    # 创建计算器实例
    calculator = TopicInnovationCalculator()

    # 计算主题创新度分数
    result_df = calculator.calculate(df)

    # 显示结果
    print("主题创新度测试结果:")
    print("=" * 80)
    print(f"处理项目数: {len(result_df)}")
    print("\n各项目详细分数:")
    for _, row in result_df.iterrows():
        print(f"\n{row['full_name']}:")
        print(f"  描述: {row['description'][:50]}...")
        print(f"  综合创新度: {row.get('topic_innovation_score', 'N/A'):.3f}")
        print(f"  主题集中度: {row.get('tc_score', 'N/A'):.1f}")
        print(f"  技术创新性: {row.get('ti_score', 'N/A'):.3f}")
        print(f"  市场需求契合: {row.get('mdm_score', 'N/A'):.3f}")
        print(f"  创新等级: {row.get('topic_innovation_level', 'N/A')}")

        # 显示子维度详情
        if 'novel_tech_stack_score' in row:
            print(f"  技术栈新颖度: {row['novel_tech_stack_score']:.3f}")
        if 'research_connection_score' in row:
            print(f"  学术关联度: {row['research_connection_score']:.3f}")

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
    test_topic_innovation_calculator()