
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Config
DATA_PATH = Path("outputs/ewri_scores.csv")
OUTPUT_DIR = Path("/Users/huypham/.gemini/antigravity/brain/0eb4a4a9-6111-4624-bb8f-7a8926846c9e")

def generate_charts():
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records.")

    # Style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.2)
    
    # 1. RISK DISTRIBUTION (Pie Chart)
    plt.figure(figsize=(10, 6))
    counts = df['risk_level'].value_counts()
    colors = {'Low': '#2ecc71', 'Medium': '#f1c40f', 'High': '#e67e22', 'Very High': '#e74c3c'}
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, 
            colors=[colors.get(x, '#95a5a6') for x in counts.index],
            wedgeprops={'edgecolor': 'white'})
    plt.title('Distribution of ESG-Washing Risk Levels (2020-2024)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'risk_distribution.png', dpi=300)
    print(f"Saved: {OUTPUT_DIR / 'risk_distribution.png'}")
    plt.close()

    # 2. TREND OVER TIME (Line Plot)
    plt.figure(figsize=(10, 6))
    yearly_avg = df.groupby('year')['ewri'].mean().reset_index()
    sns.lineplot(data=yearly_avg, x='year', y='ewri', marker='o', linewidth=3, color='#3498db')
    plt.title('Average EWRI Trend (2020-2024)')
    plt.xlabel('Year')
    plt.ylabel('Average Risk Score')
    plt.xticks(yearly_avg['year'])
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ewri_trend.png', dpi=300)
    print(f"Saved: {OUTPUT_DIR / 'ewri_trend.png'}")
    plt.close()

    # 3. HEATMAP (Bank x Year)
    plt.figure(figsize=(12, 8))
    pivot_df = df.pivot(index='bank', columns='year', values='ewri')
    # Sort by avg risk
    pivot_df['avg'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('avg', ascending=False).drop('avg', axis=1)
    
    sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdYlGn_r', linewidths=.5)
    plt.title('ESG-Washing Risk Heatmap (Red = Higher Risk)')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ewri_heatmap.png', dpi=300)
    print(f"Saved: {OUTPUT_DIR / 'ewri_heatmap.png'}")
    plt.close()

    # 4. RANKINGS (Bar Chart - Top 10 High vs Low)
    avg_scores = df.groupby('bank')['ewri'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    top_risk = avg_scores.head(5)
    bottom_risk = avg_scores.tail(5)
    combined = pd.concat([top_risk, bottom_risk])
    
    colors_bar = ['#e74c3c']*5 + ['#2ecc71']*5
    
    sns.barplot(x=combined.index, y=combined.values, palette=colors_bar)
    plt.title('Highest vs. Lowest Risk Banks (Average EWRI)')
    plt.ylabel('EWRI Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ewri_rankings.png', dpi=300)
    print(f"Saved: {OUTPUT_DIR / 'ewri_rankings.png'}")
    plt.close()

if __name__ == "__main__":
    generate_charts()
