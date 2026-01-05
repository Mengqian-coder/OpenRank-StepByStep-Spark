import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import json

# === 1. 配置部分（保持不变） ===
GITHUB_TOKEN = '待替换'  # 替换为你的token
HEADERS = {'Authorization': f'token {GITHUB_TOKEN}'}
BASE_URL = 'https://api.github.com'


# === 2. 基础搜索函数（需保留或替换） ===
def search_github_with_query(query, max_repos, headers):
    """根据单个查询语句搜索仓库"""
    repos = []
    page = 1
    per_page = 100

    while len(repos) < max_repos:
        url = f'{BASE_URL}/search/repositories'
        params = {
            'q': query,
            'sort': 'updated',
            'order': 'desc',
            'page': page,
            'per_page': per_page
        }

        print(f'  正在获取第 {page} 页...', end='')
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f' 失败! 状态码：{response.status_code}')
            # 检查是否达到API限制
            if response.status_code == 403:
                reset_time = response.headers.get('X-RateLimit-Reset')
                if reset_time:
                    wait_seconds = int(reset_time) - int(time.time())
                    print(f'API限制，等待 {wait_seconds} 秒...')
                    time.sleep(max(wait_seconds, 1))
                    continue
            break

        data = response.json()
        page_repos = data.get('items', [])

        if not page_repos:
            print(' 没有更多结果')
            break

        repos.extend(page_repos)
        print(f' 成功，本页获取 {len(page_repos)} 个，累计 {len(repos)} 个')

        # 检查是否达到API返回的总数限制
        if len(repos) >= min(data.get('total_count', 0), max_repos):
            break

        page += 1
        time.sleep(1)  # 避免触发次级限流

    return repos[:max_repos]


# === 3. 新的平衡抓取策略（替换原有的简单搜索） ===
def fetch_balanced_repository_sample():
    """
    获取平衡的仓库样本
    返回：仓库字典列表
    """

    # 计算动态日期范围（始终相对当前时间）
    def get_date_ranges():
        today = datetime.now()
        eighteen_months_ago = (today - timedelta(days=30 * 18)).strftime('%Y-%m-%d')
        twelve_months_ago = (today - timedelta(days=30 * 12)).strftime('%Y-%m-%d')
        six_months_ago = (today - timedelta(days=30 * 6)).strftime('%Y-%m-%d')
        three_months_ago = (today - timedelta(days=30 * 3)).strftime('%Y-%m-%d')
        return eighteen_months_ago, twelve_months_ago, six_months_ago, three_months_ago

    start_18m, start_12m, start_6m, start_3m = get_date_ranges()

    # 定义四个互补的搜索策略
    search_strategies = [
        {
            'name': 'python_diverse_stars',
            'query': f'created:{start_12m}..{start_6m} stars:5..300 pushed:>{start_3m} language:python',
            'target_count': 250,
            'description': 'Python项目，中等Star范围，确保活跃度'
        },
        {
            'name': 'js_ts_newer_projects',
            'query': f'created:{start_6m}..now stars:1..100 pushed:>{start_3m} language:javascript,typescript',
            'target_count': 200,
            'description': 'JS/TS新项目，较低Star数'
        },
        {
            'name': 'emerging_languages',
            'query': f'created:{start_12m}..now stars:3..200 pushed:>{start_3m} language:go,rust,kotlin,swift',
            'target_count': 200,
            'description': '新兴语言项目，技术栈多样'
        },
        {
            'name': 'mixed_lang_moderate',
            'query': f'created:{start_12m}..{start_6m} stars:10..150 pushed:>{start_3m}',
            'target_count': 150,
            'description': '混合语言，避免语言过滤偏差'
        }
    ]

    all_repos = []

    print("🚀 开始执行平衡数据抓取策略")
    print("=" * 60)

    for i, strategy in enumerate(search_strategies, 1):
        print(f"\n策略 {i}/{len(search_strategies)}: {strategy['name']}")
        print(f"描述: {strategy['description']}")
        print(f"查询: {strategy['query']}")
        print(f"目标数量: {strategy['target_count']}")

        try:
            repos = search_github_with_query(
                strategy['query'],
                strategy['target_count'],
                HEADERS
            )
            all_repos.extend(repos)
            print(f"✅ 成功获取: {len(repos)} 个仓库")

            # 显示本批次的简单统计
            if repos:
                stars = [r.get('stargazers_count', 0) for r in repos]
                langs = [r.get('language', 'Unknown') for r in repos]
                print(f"   平均Star数: {sum(stars) / len(stars):.1f}")
                print(f"   语言分布: {pd.Series(langs).value_counts().head(3).to_dict()}")

        except Exception as e:
            print(f"❌ 策略执行失败: {strategy['name']}")
            print(f"   错误: {e}")

        if i < len(search_strategies):
            print(f"等待2秒，准备下一个策略...")
            time.sleep(2)

    # === 4. 去重和整理 ===
    print(f"\n{'=' * 60}")
    print("📦 数据整理阶段")

    # 基于ID去重
    seen_ids = set()
    unique_repos = []

    for repo in all_repos:
        repo_id = repo['id']
        if repo_id not in seen_ids:
            seen_ids.add(repo_id)
            unique_repos.append(repo)

    print(f"去重前: {len(all_repos)} 个仓库")
    print(f"去重后: {len(unique_repos)} 个唯一仓库")

    # === 5. 导出基本信息供验证 ===
    if unique_repos:
        # 提取关键信息
        repo_data = []
        for repo in unique_repos:
            repo_info = {
                'id': repo['id'],
                'full_name': repo['full_name'],
                'html_url': repo['html_url'],
                'created_at': repo['created_at'],
                'updated_at': repo['updated_at'],
                'pushed_at': repo['pushed_at'],
                'stargazers_count': repo['stargazers_count'],
                'forks_count': repo['forks_count'],
                'open_issues_count': repo['open_issues_count'],
                'language': repo['language'],
                'topics': ', '.join(repo.get('topics', [])),
                'description': repo['description']
            }
            repo_data.append(repo_info)

        # 保存到CSV
        df = pd.DataFrame(repo_data)
        output_file = '../../../Tree/readme/balanced_github_repositories.csv'
        df.to_csv(output_file, index=False, encoding='utf-8-sig')

        # 快速统计
        print(f"\n📊 数据集快速统计:")
        print(f"   总项目数: {len(df)}")
        print(f"   Star数范围: {df['stargazers_count'].min()} - {df['stargazers_count'].max()}")
        print(f"   平均Star数: {df['stargazers_count'].mean():.1f}")
        print(f"   中位数Star数: {df['stargazers_count'].median():.1f}")
        print(f"   语言种类: {df['language'].nunique()}")
        print(
            f"   零Star项目: {(df['stargazers_count'] == 0).sum()} ({(df['stargazers_count'] == 0).mean() * 100:.1f}%)")

        # 语言分布
        print(f"\n   前5大语言:")
        lang_dist = df['language'].value_counts().head(5)
        for lang, count in lang_dist.items():
            pct = count / len(df) * 100
            lang_display = lang if pd.notna(lang) else 'Unknown'
            print(f"     {lang_display}: {count} ({pct:.1f}%)")

        print(f"\n💾 数据已保存至: {output_file}")

    return unique_repos


# === 6. 执行抓取（这是你需要调用的部分） ===
if __name__ == "__main__":
    print("开始执行改进的数据抓取...")
    print("=" * 60)

    try:
        # 调用新的平衡抓取函数
        repositories = fetch_balanced_repository_sample()

        print("\n" + "=" * 60)
        print("✅ 数据抓取完成!")

        # 验证数据质量
        if repositories:
            # 转换为DataFrame用于验证
            df_test = pd.DataFrame([{
                'id': r['id'],
                'name': r['full_name'],
                'stars': r['stargazers_count'],
                'language': r['language'],
                'created_at': r['created_at']
            } for r in repositories])

            # 运行我们之前的数据质量验证
            from datetime import datetime

            df_test['created_at'] = pd.to_datetime(df_test['created_at']).dt.tz_localize(None)
            df_test['project_age_days'] = (datetime.now() - df_test['created_at']).dt.days

            print("\n📈 最终数据质量检查:")
            print(f"   样本大小: {len(df_test)} 个项目")
            print(f"   Star中位数: {df_test['stars'].median()}")
            print(f"   零Star项目比例: {(df_test['stars'] == 0).mean() * 100:.1f}%")
            print(f"   语言多样性: {df_test['language'].nunique()} 种不同语言")

            # 保存详细数据
            detailed_data = []
            for repo in repositories:
                detailed_data.append({
                    'full_name': repo['full_name'],
                    'stars': repo['stargazers_count'],
                    'forks': repo['forks_count'],
                    'language': repo['language'],
                    'created_at': repo['created_at'],
                    'pushed_at': repo['pushed_at'],
                    'topics': ', '.join(repo.get('topics', [])),
                    'description': repo.get('description', ''),
                    'html_url': repo['html_url']
                })

            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_csv('detailed_balanced_repos.csv', index=False, encoding='utf-8-sig')
            print(f"💾 详细数据已保存至: detailed_balanced_repos.csv")

    except KeyboardInterrupt:
        print("\n⚠️  用户中断抓取过程")
    except Exception as e:
        print(f"\n❌ 抓取过程中出现错误: {e}")
        import traceback

        traceback.print_exc()