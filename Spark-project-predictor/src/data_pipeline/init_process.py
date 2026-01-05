import pandas as pd
import numpy as np
from datetime import datetime


def refine_activity_classification(df):
    """调整活跃度分类 - 修复时区问题版本"""

    print("🕒 处理时区问题并计算活跃度...")

    # 方法1：统一时区（推荐）
    # 将当前时间转换为UTC时区
    current_time_utc = pd.Timestamp.now(tz='UTC')

    # 确保pushed_at列是datetime类型且有时区信息
    if not pd.api.types.is_datetime64_any_dtype(df['pushed_at']):
        df['pushed_at'] = pd.to_datetime(df['pushed_at'], utc=True)
    elif df['pushed_at'].dt.tz is None:
        # 如果datetime列没有时区，添加UTC时区
        df['pushed_at'] = df['pushed_at'].dt.tz_localize('UTC')

    # 计算天数差
    df['days_since_last_update'] = (current_time_utc - df['pushed_at']).dt.days

    print(f"  最近更新统计:")
    print(f"    平均 {df['days_since_last_update'].mean():.1f} 天前更新")
    print(f"    中位数 {df['days_since_last_update'].median():.1f} 天前更新")

    # 更精细的活跃度分类
    def classify_activity_refined(days):
        if days <= 7:
            return '非常活跃(≤7天)'
        elif days <= 30:
            return '活跃(8-30天)'
        elif days <= 90:
            return '一般活跃(31-90天)'
        elif days <= 180:
            return '低活跃(91-180天)'
        else:
            return '可能停滞(>180天)'

    df['activity_level_refined'] = df['days_since_last_update'].apply(classify_activity_refined)

    # 统计新分类
    activity_dist = df['activity_level_refined'].value_counts()
    print("\n📊 调整后的活跃度分布:")
    for level, count in activity_dist.items():
        pct = count / len(df) * 100
        print(f"  {level}: {count}个项目 ({pct:.1f}%)")

    return df

def load_and_clean_data(filepath):
    """加载并清理数据，处理时区问题"""

    print(f"📂 加载数据: {filepath}")

    # 读取CSV文件，不自动解析日期（我们自己处理）
    df = pd.read_csv(filepath)

    print(f"✅ 数据加载完成: {df.shape[0]}行, {df.shape[1]}列")

    # 处理时间列 - 方法1：统一转换为带时区的datetime
    time_columns = ['created_at', 'updated_at', 'pushed_at']

    for col in time_columns:
        if col in df.columns:
            print(f"  处理 {col} 列...")

            # 方法A：转换为带UTC时区的datetime
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')

            # 检查转换结果
            null_count = df[col].isnull().sum()
            if null_count > 0:
                print(f"    ⚠️  {col}列有{null_count}个无法解析的时间值")

    return df


def enhance_data_quality(df):
    """提升数据质量 - 修复版本"""

    print("\n🔧 开始数据质量提升...")

    # 1. 处理language缺失
    df['language'] = df['language'].fillna('Unknown')

    # 2. 处理description缺失
    df['description'] = df['description'].fillna('')

    # 3. 处理topics缺失
    df['topics'] = df['topics'].fillna('')

    # 4. 处理source缺失
    df['source'] = df.get('source', 'github_relaxed')

    print(f"✅ 数据质量提升完成")
    print(f"   缺失值统计:")
    for col in ['language', 'description', 'topics', 'source']:
        if col in df.columns:
            missing = df[col].isnull().sum()
            pct = missing / len(df) * 100
            print(f"     {col}: {missing}个缺失 ({pct:.1f}%)")

    return df


# ========== 主执行流程 ==========
if __name__ == "__main__":
    # 1. 加载数据
    df = load_and_clean_data('../../data/raw/expanded_repositories_702_20251231_105833.csv')  # 替换为你的文件名

    # 2. 数据质量提升
    df = enhance_data_quality(df)

    # 3. 调整活跃度分类（使用修复后的函数）
    df = refine_activity_classification(df)

    # 4. 计算项目年龄（同样需要处理时区）
    print("\n📅 计算项目年龄...")
    current_time_utc = pd.Timestamp.now(tz='UTC')
    df['project_age_days'] = (current_time_utc - df['created_at']).dt.days
    df['project_age_months'] = df['project_age_days'] / 30.44

    print(f"  项目年龄统计:")
    print(f"    平均 {df['project_age_months'].mean():.1f} 个月")
    print(f"    范围 {df['project_age_months'].min():.1f} - {df['project_age_months'].max():.1f} 个月")

    # 5. 保存清理后的数据
    output_file = '../../data/processed/cleaned_repositories.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 清理后的数据已保存至: {output_file}")

    # 6. 显示最终统计
    print("\n" + "=" * 60)
    print("🎯 最终数据质量总结")
    print("=" * 60)
    print(f"总项目数: {len(df)}")
    print(f"语言种类: {df['language'].nunique()}")
    print(f"Star中位数: {df['stargazers_count'].median()}")
    print(f"零Star项目: {(df['stargazers_count'] == 0).sum()} ({(df['stargazers_count'] == 0).mean() * 100:.1f}%)")

    # 活跃度详细分布
    if 'activity_level_refined' in df.columns:
        print(f"\n活跃度分布:")
        for level in ['非常活跃(≤7天)', '活跃(8-30天)', '一般活跃(31-90天)', '低活跃(91-180天)', '可能停滞(>180天)']:
            if level in df['activity_level_refined'].values:
                count = (df['activity_level_refined'] == level).sum()
                pct = count / len(df) * 100
                print(f"  {level}: {count} ({pct:.1f}%)")