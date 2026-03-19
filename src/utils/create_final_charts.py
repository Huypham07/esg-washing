
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.bank_anonymization import anonymize_bank_name

# Config
DATA_PATH = Path("outputs/ewri_scores.csv")
OUTPUT_DIR = Path("outputs/figures")  # Generate directly into repo outputs first
ARTIFACT_DIR = Path("/Users/huypham/.gemini/antigravity/brain/0eb4a4a9-6111-4624-bb8f-7a8926846c9e")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_plot(filename):
    # Save to both local repo output and artifact dir
    # Save as PNG
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.savefig(ARTIFACT_DIR / filename, dpi=300, bbox_inches='tight')
    
    # Save as EPS
    eps_filename = filename.replace('.png', '.eps')
    plt.savefig(OUTPUT_DIR / eps_filename, format='eps', bbox_inches='tight')
    plt.savefig(ARTIFACT_DIR / eps_filename, format='eps', bbox_inches='tight')
    
    print(f"Saved: {filename} and {eps_filename}")
    plt.close()

def generate_ewri_charts():
    print("Generating EWRI Charts...")
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found. Skipping EWRI charts.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Anonymize bank names
    print("Anonymizing bank names...")
    df['bank'] = df['bank'].apply(anonymize_bank_name)
    
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
    # plt.title('Distribution of ESG-Washing Risk Levels (2020-2024)')
    save_plot('risk_distribution.png')

    # 2. TREND OVER TIME (Line Plot)
    plt.figure(figsize=(10, 6))
    yearly_avg = df.groupby('year')['ewri'].mean().reset_index()
    sns.lineplot(data=yearly_avg, x='year', y='ewri', marker='o', linewidth=3, color='#3498db')
    # plt.title('Average EWRI Trend (2020-2024)')
    plt.xlabel('Year')
    plt.ylabel('Average Risk Score')
    plt.xticks(yearly_avg['year']) # Ensure integers
    plt.grid(True, linestyle='--')
    save_plot('ewri_trend.png')

    # 3. HEATMAP (Bank x Year)
    plt.figure(figsize=(12, 8))
    pivot_df = df.pivot(index='bank', columns='year', values='ewri')
    # Sort by avg risk
    pivot_df['avg'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('avg', ascending=False).drop('avg', axis=1)
    
    sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdYlGn_r', linewidths=.5)
    # plt.title('ESG-Washing Risk Heatmap (EWRI) (Red = Higher Risk)')
    plt.ylabel('')
    save_plot('ewri_heatmap.png')

    # 4. RANKINGS (Bar Chart - Top 5 High vs Low)
    avg_scores = df.groupby('bank')['ewri'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    top_risk = avg_scores.head(5)
    bottom_risk = avg_scores.tail(5)
    # Reverse bottom risk so best is at the end or handle display? 
    # Just concating top 5 worst and top 5 best
    combined = pd.concat([top_risk, bottom_risk])
    
    colors_bar = ['#e74c3c']*5 + ['#2ecc71']*5
    
    sns.barplot(x=combined.index, y=combined.values, palette=colors_bar)
    # plt.title('ESG Risk Ranking (EWRI): Highest Risk vs. Lowest Risk')
    plt.ylabel('Average EWRI Score')
    plt.xticks(rotation=45, ha='right')
    save_plot('ewri_rankings.png')

def generate_model_performance_charts():
    print("Generating Model Performance Charts...")
    plt.style.use('seaborn-v0_8')
    
    # --- Actionability Model ---
    # Data from Notebook (Gold Set)
    act_data = {
        'Class': ['Implemented', 'Planning', 'Indeterminate'],
        'F1-Score': [0.78, 0.88, 0.79],
        'Support': [166, 166, 166] 
    }
    df_act = pd.DataFrame(act_data)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_act, x='Class', y='F1-Score', palette="Blues_d")
    # plt.title('Actionability Model Performance (Gold Set)', fontsize=14, pad=15)
    plt.ylim(0, 1.1)
    plt.ylabel('F1-Score', fontsize=12)
    
    # Add labels
    for i, v in enumerate(df_act['F1-Score']):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=11, fontweight='bold')
        
    save_plot('actionability_performance.png')
    
    # --- Topic Classifier ---
    # Data from Notebook (Gold Set)
    topic_data = {
        'Topic': ['E (Env)', 'S (Labor)', 'S (Comm)', 'S (Prod)', 'G (Gov)', 'Non-ESG'],
        'F1-Score': [0.965, 0.970, 0.965, 0.942, 0.945, 0.946]
    }
    df_topic = pd.DataFrame(topic_data)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_topic, x='Topic', y='F1-Score', palette="viridis")
    # plt.title('Topic Classifier Performance (Gold Set)', fontsize=14, pad=15)
    plt.ylim(0.8, 1.05) # Zoom in to show high performance differences
    plt.ylabel('F1-Score', fontsize=12)
    
    # Add labels
    for i, v in enumerate(df_topic['F1-Score']):
        ax.text(i, v + 0.005, f"{v:.3f}", ha='center', fontsize=10)
        
    save_plot('topic_performance.png')

    # --- Overall Accuracy Comparison ---
    overall_data = {
        'Model': ['Actionability', 'Topic Classifier'],
        'Accuracy': [0.819, 0.950]
    }
    df_overall = pd.DataFrame(overall_data)
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=df_overall, x='Model', y='Accuracy', palette="magma")
    # plt.title('Overall Model Accuracy on Gold Set', fontsize=14)
    plt.ylim(0.8, 1.0)
    
    for i, v in enumerate(df_overall['Accuracy']):
        ax.text(i, v + 0.01, f"{v:.1%}", ha='center', fontsize=12, fontweight='bold')
        
    save_plot('overall_accuracy.png')

if __name__ == "__main__":
    generate_ewri_charts()
    generate_model_performance_charts()
