import requests
import pandas as pd
import time
import json
from datetime import datetime, timedelta
import sys


# ========== 1. 安全的基础搜索函数 ==========
def safe_github_search(query, max_items=100, token=None):
    """
    安全的GitHub搜索函数，避免编码问题
    使用直接API调用，简化处理逻辑
    """

    # 清理查询字符串中的非ASCII字符
    if isinstance(query, str):
        query = ''.join(char for char in query if ord(char) < 128)

    url = "https://api.github.com/search/repositories"

    # 准备请求参数
    params = {
        "q": query,
        "sort": "updated",
        "order": "desc",
        "per_page": min(100, max_items),
        "page": 1
    }

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "GitHub-Data-Collector"
    }

    # 添加token（如果有）
    if token and token.strip():
        safe_token = ''.join(char for char in token.strip() if ord(char) < 128)
        if safe_token:
            headers["Authorization"] = f"token {safe_token}"

    all_items = []
    total_fetched = 0

    while total_fetched < max_items:
        try:
            # 发送请求
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=30
            )

            # 检查响应
            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])

                if not items:
                    break  # 没有更多数据

                all_items.extend(items)
                total_fetched += len(items)

                print(f"    已获取 {total_fetched}/{max_items} 个项目")

                # 检查是否还有更多页面
                if len(items) < params["per_page"]:
                    break

                # 准备下一页
                params["page"] += 1
                time.sleep(1.5)  # 礼貌间隔，避免API限制

            elif response.status_code == 403:
                # 处理API限制
                reset_time = response.headers.get("X-RateLimit-Reset")
                if reset_time:
                    wait_seconds = max(10, int(reset_time) - int(time.time()))
                    print(f"    API限制，等待 {wait_seconds} 秒...")
                    time.sleep(wait_seconds)
                    continue
                else:
                    print("    API限制，等待60秒...")
                    time.sleep(60)
                    continue

            else:
                print(f"    请求失败: 状态码 {response.status_code}")
                break

        except requests.exceptions.RequestException as e:
            print(f"    网络错误: {e}")
            time.sleep(10)
            continue

        except Exception as e:
            print(f"    意外错误: {e}")
            break

    print(f"    本次搜索完成，获取 {len(all_items)} 个项目")
    return all_items[:max_items]


# ========== 2. 主扩充函数 ==========
def expand_with_relaxed_queries(existing_file='balanced_github_repositories.csv',
                                target_size=550,
                                github_token=None):
    """
    使用放宽的搜索条件扩充数据集
    """

    print("=" * 60)
    print("🚀 开始执行放宽条件的数据扩充")
    print("=" * 60)

    # 1. 加载现有数据
    print("\n📂 加载现有数据...")
    try:
        existing_df = pd.read_csv(existing_file)
        existing_count = len(existing_df)
        print(f"✅ 成功加载 {existing_count} 个项目")

        # 显示现有数据的基本统计
        if existing_count > 0:
            if 'stargazers_count' in existing_df.columns:
                zero_star = (existing_df['stargazers_count'] == 0).sum()
                zero_pct = zero_star / existing_count * 100
                print(f"   零Star项目: {zero_star} ({zero_pct:.1f}%)")

            if 'language' in existing_df.columns:
                lang_count = existing_df['language'].nunique()
                print(f"   语言种类: {lang_count}")

    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return None

    # 2. 检查是否需要扩充
    if existing_count >= target_size:
        print(f"✅ 已达到目标数量 {target_size}")
        return existing_df

    needed = target_size - existing_count
    print(f"📈 需要补充: {needed} 个项目")

    # 3. 准备放宽的搜索条件
    print("\n🔍 准备放宽的搜索条件...")

    # 计算日期范围（放宽条件）
    today = datetime.now()

    # 条件1：更宽的时间范围（3-24个月，而不是6-18个月）
    three_months_ago = (today - timedelta(days=90)).strftime('%Y-%m-%d')
    twentyfour_months_ago = (today - timedelta(days=720)).strftime('%Y-%m-%d')

    # 条件2：更宽的Star范围（1-1000，而不是1-500）
    # 条件3：放宽pushed条件（最近6个月有更新，而不是3个月）
    six_months_ago = (today - timedelta(days=180)).strftime('%Y-%m-%d')

    # 定义多个放宽的搜索条件
    relaxed_queries = [
        # 查询1：宽时间范围，宽Star范围，宽松活跃度
        {
            "name": "宽范围基础查询",
            "query": f"created:{twentyfour_months_ago}..{three_months_ago} stars:1..1000",
            "target": min(200, needed + 50)
        },

        # 查询2：聚焦近期项目，但Star范围更宽
        {
            "name": "近期项目宽Star范围",
            "query": f"created:{three_months_ago}..now stars:1..800 pushed:>{six_months_ago}",
            "target": min(150, max(50, needed // 2))
        },

        # 查询3：按语言分组查询（放宽条件）
        {
            "name": "Python放宽条件",
            "query": f"language:python created:{twentyfour_months_ago}..now stars:1..700",
            "target": min(100, max(30, needed // 3))
        },

        {
            "name": "JavaScript放宽条件",
            "query": f"language:javascript created:{twentyfour_months_ago}..now stars:1..600",
            "target": min(100, max(30, needed // 3))
        },

        # 查询4：完全不限制语言，只限制时间和Star
        {
            "name": "全语言放宽条件",
            "query": f"created:{twentyfour_months_ago}..{three_months_ago} stars:10..500",
            "target": min(150, needed)
        }
    ]

    # 4. 执行搜索
    print("\n⚡ 开始执行放宽条件的搜索...")

    all_new_repos = []
    total_fetched = 0

    for query_info in relaxed_queries:
        name = query_info["name"]
        query = query_info["query"]
        target = query_info["target"]

        if total_fetched >= needed:
            print("✅ 已获取足够的新项目")
            break

        print(f"\n📋 执行查询: {name}")
        print(f"   查询条件: {query}")
        print(f"   目标数量: {target}")

        # 执行搜索
        try:
            new_repos = safe_github_search(
                query=query,
                max_items=target,
                token=github_token
            )

            if new_repos:
                all_new_repos.extend(new_repos)
                total_fetched += len(new_repos)
                print(f"   ✅ 成功获取 {len(new_repos)} 个项目")
                print(f"   累计获取: {total_fetched}/{needed}")
            else:
                print("   ⚠️  未获取到项目")

            # 查询间休息
            time.sleep(3)

        except Exception as e:
            print(f"   ❌ 查询执行失败: {e}")
            continue

    print(f"\n📦 搜索阶段完成")
    print(f"   新获取项目总数: {len(all_new_repos)}")

    # 5. 处理新数据
    if not all_new_repos:
        print("⚠️  未获取到新数据，使用现有数据")
        return existing_df

    # 转换为DataFrame
    new_data = []
    for repo in all_new_repos:
        try:
            # 提取关键信息
            repo_info = {
                'id': str(repo.get('id', '')),
                'full_name': str(repo.get('full_name', '')),
                'html_url': str(repo.get('html_url', '')),
                'created_at': str(repo.get('created_at', '')),
                'updated_at': str(repo.get('updated_at', '')),
                'pushed_at': str(repo.get('pushed_at', '')),
                'stargazers_count': int(repo.get('stargazers_count', 0)),
                'forks_count': int(repo.get('forks_count', 0)),
                'open_issues_count': int(repo.get('open_issues_count', 0)),
                'language': str(repo.get('language', '')) if repo.get('language') else '',
                'topics': ', '.join([str(t) for t in repo.get('topics', [])][:5]),
                'description': str(repo.get('description', ''))[:200] if repo.get('description') else '',
                'source': 'github_relaxed'
            }
            new_data.append(repo_info)
        except Exception as e:
            print(f"   处理仓库数据时出错: {e}")
            continue

    new_df = pd.DataFrame(new_data)

    # 去重（基于id）
    if existing_count > 0 and 'id' in existing_df.columns and 'id' in new_df.columns:
        # 转换为字符串确保类型一致
        existing_ids = set(existing_df['id'].astype(str).tolist())
        new_df['id_str'] = new_df['id'].astype(str)

        # 过滤掉已存在的项目
        before_dedup = len(new_df)
        new_df = new_df[~new_df['id_str'].isin(existing_ids)]
        after_dedup = len(new_df)

        if 'id_str' in new_df.columns:
            new_df = new_df.drop(columns=['id_str'])

        print(f"   去重: {before_dedup} → {after_dedup} 个项目")

    # 6. 合并数据
    print("\n🔄 合并数据...")

    if len(new_df) > 0:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        final_count = len(combined_df)

        # 保存数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'expanded_repositories_{final_count}_{timestamp}.csv'
        combined_df.to_csv(output_file, index=False, encoding='utf-8')

        print(f"✅ 合并完成!")
        print(f"   最终项目数: {final_count}")
        print(f"   保存至: {output_file}")

        # 显示统计信息
        print(f"\n📊 扩充后统计:")
        print(f"   Star中位数: {combined_df['stargazers_count'].median():.1f}")

        zero_star = (combined_df['stargazers_count'] == 0).sum()
        zero_pct = zero_star / final_count * 100
        print(f"   零Star项目: {zero_star} ({zero_pct:.1f}%)")

        lang_count = combined_df['language'].nunique()
        print(f"   语言种类: {lang_count}")

        # 显示前5大语言
        if 'language' in combined_df.columns:
            top_langs = combined_df['language'].value_counts().head(5)
            print(f"\n   前5大语言分布:")
            for lang, count in top_langs.items():
                pct = count / final_count * 100
                lang_display = str(lang)[:20] if pd.notna(lang) else 'Unknown'
                print(f"     {lang_display:<20} {count:>4} ({pct:>5.1f}%)")

        return combined_df
    else:
        print("⚠️  没有新增的唯一项目")
        return existing_df


# ========== 3. 辅助函数：测试API连接 ==========
def test_github_connection(token=None):
    """测试GitHub API连接"""

    print("🧪 测试GitHub API连接...")

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "GitHub-Connection-Test"
    }

    if token and token.strip():
        safe_token = ''.join(char for char in token.strip() if ord(char) < 128)
        if safe_token:
            headers["Authorization"] = f"token {safe_token}"

    try:
        # 简单的API调用测试
        response = requests.get(
            "https://api.github.com/zen",
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:
            print(f"✅ GitHub API连接正常")
            print(f"   状态: {response.text}")
            return True
        else:
            print(f"❌ 连接失败，状态码: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ 连接异常: {e}")
        return False


# ========== 4. 主程序入口 ==========
def main():
    """主程序"""

    print("=" * 60)
    print("📊 GitHub新生代项目数据扩充工具 (放宽条件版)")
    print("=" * 60)

    # 配置参数
    EXISTING_FILE = "../../../Tree/readme/balanced_github_repositories.csv"  # 你的现有数据文件
    TARGET_SIZE = 1000  # 目标项目数量
    GITHUB_TOKEN = "代替换"  # 替换为你的token

    # 测试连接
    print("\n1. 测试API连接...")
    if not test_github_connection(GITHUB_TOKEN):
        print("⚠️  API连接测试失败，将继续使用公开API（速率限制较低）")
        # 可以不使用token，但速率限制会更严格
        use_token = None
    else:
        use_token = GITHUB_TOKEN
        print("✅ 使用Token进行API调用")

    print(f"\n2. 数据扩充配置:")
    print(f"   现有文件: {EXISTING_FILE}")
    print(f"   目标数量: {TARGET_SIZE}")

    # 执行数据扩充
    print("\n3. 开始数据扩充...")

    try:
        result_df = expand_with_relaxed_queries(
            existing_file=EXISTING_FILE,
            target_size=TARGET_SIZE,
            github_token=use_token
        )

        # 显示最终结果
        print("\n" + "=" * 60)
        if result_df is not None:
            final_count = len(result_df)

            if final_count >= TARGET_SIZE:
                print(f"🎉 成功! 达到目标规模: {final_count} 个项目")
            elif final_count > 400:
                print(f"✅ 部分成功: {final_count} 个项目 (接近目标)")
            else:
                print(f"⚠️  未达预期: 仅 {final_count} 个项目")

            # 提供后续建议
            print(f"\n💡 后续建议:")
            if final_count < 500:
                print(f"   1. 再次运行此脚本，可能会获取更多项目")
                print(f"   2. 检查现有数据的Star分布，调整搜索条件")
                print(f"   3. 考虑使用多个GitHub Token轮换")
            else:
                print(f"   1. 数据量已足够，可以进入下一阶段分析")
                print(f"   2. 使用EDA脚本验证数据质量")
                print(f"   3. 开始设计多维度评分指标")

        else:
            print("❌ 数据扩充失败")

    except KeyboardInterrupt:
        print("\n⚠️  用户中断执行")
    except Exception as e:
        print(f"\n❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


# ========== 5. 快速使用说明 ==========
if __name__ == "__main__":
    # 使用说明
    print("=" * 60)
    print("使用说明:")
    print("1. 确保 balanced_github_repositories.csv 文件存在")
    print("2. 将代码中的 GITHUB_TOKEN 替换为你的Personal Access Token")
    print("3. 运行此脚本")
    print("=" * 60)

    # 确认执行
    confirm = input("\n是否开始执行数据扩充? (y/n): ")

    if confirm.lower() == 'y':
        main()
    else:
        print("取消执行")