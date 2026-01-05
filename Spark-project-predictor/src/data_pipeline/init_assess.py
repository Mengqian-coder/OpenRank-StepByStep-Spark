import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# 1. Load Data
file_path = '../../data/raw/expanded_repositories_702_20251231_105833.csv'
try:
    df = pd.read_csv(file_path, parse_dates=['created_at', 'updated_at', 'pushed_at'])
    print("✅ Data loaded successfully!")
    print(f"📊 Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print(f"❌ File not found: {file_path}")
    exit()
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit()

# 2. Data Preview
print("\n=== Data Preview ===")
print(df.head())
print("\n=== Data Information ===")
print(df.info())
print("\n=== Descriptive Statistics ===")
print(df[['stargazers_count', 'forks_count', 'open_issues_count']].describe())

# 3. Data Quality Check
print("\n=== Data Quality Check ===")
print(f"Missing values:")
print(df.isnull().sum())
print(f"\nKey fields missing percentage:")
for col in ['language', 'description', 'pushed_at']:
    missing_pct = df[col].isnull().mean() * 100
    print(f"  {col}: {missing_pct:.1f}% missing")

# Fix timezone issues for datetime calculations
print("\nFixing timezone issues...")
df['created_at'] = df['created_at'].dt.tz_localize(None)
df['pushed_at'] = df['pushed_at'].dt.tz_localize(None)

# 4. Project Age Analysis
print("\n=== Project Age Analysis ===")
df['project_age_days'] = (pd.Timestamp.now() - df['created_at']).dt.days
df['project_age_months'] = df['project_age_days'] / 30.44

print(f"📅 Project age statistics:")
print(f"  Average age: {df['project_age_months'].mean():.1f} months")
print(f"  Minimum age: {df['project_age_months'].min():.1f} months")
print(f"  Maximum age: {df['project_age_months'].max():.1f} months")
print(f"  Median age: {df['project_age_months'].median():.1f} months")

# Create visualization figures
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Subplot 1: Project Age Histogram
ax1 = axes[0, 0]
n, bins, patches = ax1.hist(df['project_age_months'], bins=30, edgecolor='black', alpha=0.7)
ax1.axvline(x=6, color='r', linestyle='--', label='6 months lower bound')
ax1.axvline(x=18, color='r', linestyle='--', label='18 months upper bound')
ax1.set_xlabel('Project Age (Months)')
ax1.set_ylabel('Number of Projects')
ax1.set_title('Project Age Distribution Histogram')
ax1.legend()

# Add distribution curve
from scipy.stats import gaussian_kde
kde = gaussian_kde(df['project_age_months'].dropna())
x_range = np.linspace(df['project_age_months'].min(), df['project_age_months'].max(), 1000)
ax1.plot(x_range, kde(x_range) * len(df['project_age_months'].dropna()) * (bins[1]-bins[0]),
         'r-', linewidth=2)

# Subplot 2: Monthly Creation Trend
ax2 = axes[0, 1]
df['created_month'] = df['created_at'].dt.to_period('M').astype(str)
monthly_counts = df.groupby('created_month').size()
ax2.plot(monthly_counts.index, monthly_counts.values, marker='o', linewidth=2)
ax2.set_xlabel('Creation Month')
ax2.set_ylabel('Number of Projects')
ax2.set_title('Monthly New Projects Trend')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

# 5. Star Count Analysis
print("\n=== Star Count Analysis ===")
print(f"🌟 Star count statistics:")
print(f"  Mean: {df['stargazers_count'].mean():.1f}")
print(f"  Median: {df['stargazers_count'].median():.1f}")
print(f"  Max: {df['stargazers_count'].max()}")
print(f"  Min: {df['stargazers_count'].min()}")

# Count projects with less than 500 stars
below_500 = df[df['stargazers_count'] < 500].shape[0]
below_500_pct = below_500 / df.shape[0] * 100
print(f"\n🎯 Core check: Projects with Star Count < 500")
print(f"  {below_500} / {df.shape[0]} = {below_500_pct:.1f}%")
if below_500_pct > 90:
    print("  ✅ Excellent! Most projects meet 'cold-start' criteria")
elif below_500_pct > 70:
    print("  ⚠️ Good, but some popular projects mixed in")
else:
    print("  ❌ May need to adjust filtering criteria")

# Subplot 3: Star Distribution Histogram (Log scale)
ax3 = axes[0, 2]
star_data = df['stargazers_count']
ax3.hist(star_data, bins=50, edgecolor='black', alpha=0.7, log=True)
ax3.axvline(x=500, color='r', linestyle='--', label='500 Stars threshold')
ax3.set_xlabel('Star Count (Log Scale)')
ax3.set_ylabel('Number of Projects (Log Scale)')
ax3.set_title('Star Distribution (Log Scale)')
ax3.legend()

# Subplot 4: Star Count Boxplot
ax4 = axes[1, 0]
ax4.boxplot(star_data, vert=False, patch_artist=True)
ax4.axvline(x=500, color='r', linestyle='--')
ax4.set_xlabel('Star Count')
ax4.set_title('Star Count Boxplot')

# Subplot 5: Stars vs Project Age
ax5 = axes[1, 1]
ax5.scatter(df['project_age_months'], df['stargazers_count'], alpha=0.5, s=10)
ax5.axhline(y=500, color='r', linestyle='--', label='500 Stars')
ax5.set_xlabel('Project Age (Months)')
ax5.set_ylabel('Star Count')
ax5.set_title('Star Count vs Project Age')
ax5.legend()
ax5.set_yscale('log')

# 6. Programming Language Analysis
print("\n=== Programming Language Analysis ===")
lang_counts = df['language'].value_counts()
print(f"📚 Total languages: {len(lang_counts)}")
print(f"🏆 Top 10 language distribution:")
for i, (lang, count) in enumerate(lang_counts.head(10).items(), 1):
    pct = count / df.shape[0] * 100
    lang_display = lang if pd.notna(lang) else 'No language'
    print(f"  {i:2d}. {lang_display}: {count:3d} projects ({pct:.1f}%)")

# Subplot 6: Language Distribution Pie Chart
ax6 = axes[1, 2]
top_n = 10
top_langs = lang_counts.head(top_n)
other_count = lang_counts[top_n:].sum()
if other_count > 0:
    top_langs = top_langs._append(pd.Series([other_count], index=['Other']))

ax6.pie(top_langs.values, labels=top_langs.index, autopct='%1.1f%%', startangle=90)
ax6.axis('equal')
ax6.set_title(f'Top {top_n} Programming Languages Distribution')

plt.suptitle('Exploratory Analysis of Newborn Open Source Projects Dataset',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# 7. Project Activity Analysis
print("\n=== Project Activity Analysis ===")
df['days_since_last_update'] = (pd.Timestamp.now() - df['pushed_at']).dt.days

print(f"🔄 Recent update status:")
print(f"  Average updated {df['days_since_last_update'].mean():.1f} days ago")
print(f"  Median updated {df['days_since_last_update'].median():.1f} days ago")

def classify_activity(days):
    if days <= 30:
        return 'Highly Active (≤30 days)'
    elif days <= 90:
        return 'Moderately Active (31-90 days)'
    elif days <= 180:
        return 'Low Activity (91-180 days)'
    else:
        return 'Possibly Stalled (>180 days)'

df['activity_level'] = df['days_since_last_update'].apply(classify_activity)
activity_dist = df['activity_level'].value_counts()

print(f"\n📈 Activity level distribution:")
for level, count in activity_dist.items():
    pct = count / df.shape[0] * 100
    print(f"  {level}: {count} projects ({pct:.1f}%)")

# 8. Correlation Analysis
print("\n=== Correlation Analysis ===")
correlation_cols = ['stargazers_count', 'forks_count', 'open_issues_count', 'project_age_days']
corr_matrix = df[correlation_cols].corr()

print("Correlation Matrix (Pearson correlation coefficient):")
print(corr_matrix)

# Visualization of correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Key Metrics Correlation Heatmap')
plt.tight_layout()
plt.show()

# 9. Analysis Summary
print("\n" + "="*60)
print("🎯 Data Quality & Filtering Effectiveness Summary")
print("="*60)
print(f"1. Data Scale: {df.shape[0]} projects, {df.shape[1]} features")
print(f"2. Project Age: {df['project_age_months'].mean():.1f} ± {df['project_age_months'].std():.1f} months")
print(f"3. Star Distribution: Median={df['stargazers_count'].median()}, Mean={df['stargazers_count'].mean():.1f}")
print(f"4. Cold-start Qualification Rate: {below_500_pct:.1f}% projects with Star Count < 500")
print(f"5. Language Diversity: {len(lang_counts)} languages, Top 3: {', '.join(lang_counts.head(3).index.fillna('None').tolist())}")
print(f"6. Data Completeness: Key fields missing rate <10%: {all(df[col].isnull().mean() < 0.1 for col in ['language', 'pushed_at'])}")
print("="*60)

# 10. Save processed data for next phase
df.to_csv('github_newborn_repos_processed.csv', index=False, encoding='utf-8')
print("\n💾 Processed data saved to 'github_newborn_repos_processed.csv'")
print("✅ EDA completed successfully!")