import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def save_plot(name, output_dir='outputs/figures'):
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, name), dpi=300, bbox_inches='tight')
    if name.endswith('.png'):
        plt.savefig(os.path.join(output_dir, name.replace('.png', '.eps')), bbox_inches='tight')
    print(f"Saved: {name}")

def generate_topic_action_viz():
    # Data from latest topic_ewri_analysis.py run
    data = {
        'Topic': ['G', 'S_labor', 'E', 'S_community', 'S_product'],
        'Implemented': [29.4, 32.9, 41.2, 42.9, 43.6],
        'Planning': [2.8, 2.3, 6.1, 2.4, 5.2],
        'Indeterminate': [67.8, 64.8, 52.8, 54.7, 51.2]
    }
    
    df = pd.DataFrame(data)
    
    # Sort by Indeterminate (Washing Risk) descending
    df = df.sort_values('Indeterminate', ascending=False)
    
    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "DejaVu Sans"
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    topics = df['Topic']
    bottom = np.zeros(len(topics))
    
    colors = ['#2ecc71', '#f1c40f', '#e74c3c'] # Implemented, Planning, Indeterminate
    labels = ['Implemented', 'Planning', 'Indeterminate']
    
    for i, col in enumerate(['Implemented', 'Planning', 'Indeterminate']):
        ax.bar(topics, df[col], bottom=bottom, label=labels[i], color=colors[i], alpha=0.85)
        
        # Add percentage labels in the middle of bars
        for idx, val in enumerate(df[col]):
            if val > 5: # Only show if segment is large enough
                ax.text(idx, bottom[idx] + val/2, f"{val}%", 
                        ha='center', va='center', fontweight='bold', color='white' if i != 1 else 'black')
        
        bottom += df[col]

    ax.set_ylabel('Percentage of Statements (%)', fontsize=12)
    ax.set_xlabel('ESG Topic (GRI Pillar)', fontsize=12)
    # ax.set_title('ESG Actionability Breakdown by Topic', fontsize=14, pad=20)
    ax.legend(title='Actionability Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    save_plot('topic_actionability_breakdown.png')

    # Also create a simple Washing Rate comparison
    df['Washing Rate'] = df['Indeterminate'] + df['Planning']
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Topic', y='Washing Rate', palette='Reds_r')
    plt.ylabel('Washing Risk Rate (%)', fontsize=12)
    plt.xlabel('ESG Topic', fontsize=12)
    plt.ylim(0, 100)
    
    for i, val in enumerate(df['Washing Rate']):
        plt.text(i, val + 1, f"{val:.1f}%", ha='center', fontweight='bold')
        
    save_plot('topic_washing_rate.png')

if __name__ == "__main__":
    generate_topic_action_viz()
