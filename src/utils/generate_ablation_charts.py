
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.bank_anonymization import anonymize_bank_name

# Config
EWRI_PATH = Path("outputs/ewri_scores.csv")
CORPUS_PATH = Path("data/corpus/esg_sentences_enhanced_v2.parquet")
OUTPUT_DIR = Path("outputs/figures")
ARTIFACT_DIR = Path("/Users/huypham/.gemini/antigravity/brain/0eb4a4a9-6111-4624-bb8f-7a8926846c9e")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_plot(filename):
    # Save as PNG
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.savefig(ARTIFACT_DIR / filename, dpi=300, bbox_inches='tight')
    
    # Save as EPS
    eps_filename = filename.replace('.png', '.eps')
    plt.savefig(OUTPUT_DIR / eps_filename, format='eps', bbox_inches='tight')
    plt.savefig(ARTIFACT_DIR / eps_filename, format='eps', bbox_inches='tight')
    
    print(f"Saved: {filename} and {eps_filename}")
    plt.close()

def generate_component_correlation():
    """Generate heatmap showing correlation between EWRI components"""
    print("Generating Component Correlation Heatmap...")
    
    if not CORPUS_PATH.exists() or not EWRI_PATH.exists():
        print(f"Warning: Required files not found at {CORPUS_PATH} or {EWRI_PATH}")
        return
    
    # Load data
    df = pd.read_parquet(CORPUS_PATH)
    ewri_df = pd.read_csv(EWRI_PATH)
    
    # Anonymize
    df['bank'] = df['bank'].apply(anonymize_bank_name)
    ewri_df['bank'] = ewri_df['bank'].apply(anonymize_bank_name)
    
    # Aggregate corpus to bank-year
    # We want to see how these document-level properties relate to EWRI
    bank_year_stats = df.groupby(['bank', 'year']).agg({
        'action_pred': [
            ('Indeterminate_Ratio', lambda x: (x == 'Indeterminate').mean()),
            ('Planning_Ratio', lambda x: (x == 'Planning').mean())
        ],
        'nli_label': [
            ('NLI_Entailment_Rate', lambda x: (x == 'entailment').mean())
        ],
        'evidence_strength': [('Avg_Evidence_Strength', 'mean')],
        'similarity_score': [('Avg_Similarity', 'mean')]
    })
    
    # Flatten multi-index columns
    bank_year_stats.columns = [col[1] for col in bank_year_stats.columns]
    bank_year_stats = bank_year_stats.reset_index()
    
    # Merge with EWRI
    merged = bank_year_stats.merge(ewri_df[['bank', 'year', 'ewri']], on=['bank', 'year'])
    merged = merged.rename(columns={'ewri': 'EWRI'})
    
    # Compute correlation
    cols_to_corr = ['Indeterminate_Ratio', 'Planning_Ratio', 'NLI_Entailment_Rate', 'Avg_Evidence_Strength', 'Avg_Similarity', 'EWRI']
    corr_data = merged[cols_to_corr]
    corr_matrix = corr_data.corr()
    
    # Rename for cleaner plot
    corr_matrix.columns = ['Indeterminate', 'Planning', 'NLI Entailment', 'Evidence Strength', 'Similarity', 'EWRI Score']
    corr_matrix.index = ['Indeterminate', 'Planning', 'NLI Entailment', 'Evidence Strength', 'Similarity', 'EWRI Score']
    
    print("\n" + "="*40)
    print("CORRELATION MATRIX")
    print("="*40)
    print(corr_matrix.round(3))
    print("="*40 + "\n")
    
    # Plot heatmap
    plt.figure(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_plot('ablation_correlation.png')
    
    return merged

def generate_evidence_scatter(merged_data):
    """Generate scatter showing Evidence Strength vs EWRI"""
    print("Generating Evidence Impact Scatter...")
    
    if merged_data is None:
        print("Warning: No data for scatter plot")
        return
    
    plt.figure(figsize=(10, 7))
    
    # Scatter plot with color by Indeterminate Ratio
    scatter = plt.scatter(merged_data['Avg_Evidence_Strength'], merged_data['EWRI'], 
                         c=merged_data['Indeterminate_Ratio'], cmap='RdYlGn_r', 
                         s=120, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    plt.colorbar(scatter, label='Indeterminate Ratio (Symbolic Content)')
    plt.xlabel('Average Evidence Strength', fontsize=12)
    plt.ylabel('EWRI Score', fontsize=12)
    # plt.title('Evidence Strength vs EWRI', fontsize=14, pad=15)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add trend line
    z = np.polyfit(merged_data['Avg_Evidence_Strength'], merged_data['EWRI'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged_data['Avg_Evidence_Strength'].min(), 
                        merged_data['Avg_Evidence_Strength'].max(), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.6, linewidth=2.5, 
             label=f'Trend: EWRI = {z[0]:.1f} × ES + {z[1]:.1f}')
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    save_plot('ablation_scatter.png')

def generate_evidence_distribution():
    """Generate distribution comparison of evidence strength"""
    print("Generating Evidence Strength Distribution...")
    
    if not CORPUS_PATH.exists():
        return
    
    df = pd.read_parquet(CORPUS_PATH)
    df['bank'] = df['bank'].apply(anonymize_bank_name)
    
    plt.figure(figsize=(10, 6))
    
    # Only plot sentences with evidence
    evidence_df = df[df['has_evidence']]
    
    # KDE plots
    sns.kdeplot(data=evidence_df, x='rule_based_strength', 
                label='Rule-based Only', fill=True, alpha=0.4, linewidth=2)
    sns.kdeplot(data=evidence_df, x='evidence_strength', 
                label='Neuro-Symbolic (Enhanced)', fill=True, alpha=0.4, linewidth=2)
    
    # plt.title('Evidence Strength Distribution (Rule-based vs Neuro-Symbolic)', fontsize=14, pad=15)
    plt.xlabel('Evidence Strength (0-1)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_plot('evidence_strength_dist.png')

def generate_linking_viz():
    """Generate claim-evidence linking visualization"""
    print("Generating Linking Visualization...")
    sample_path = Path("outputs/evidence_analysis/linking_samples.csv")
    
    if not sample_path.exists():
        print(f"Warning: {sample_path} not found")
        return
    
    samples = pd.read_csv(sample_path)
    # samples['bank'] = samples['bank'].apply(anonymize_bank_name) # Bank column missing in samples
    samples = samples.head(5)  # Top 5 samples
    
    # Similar to before - create table visualization
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    import textwrap
    
    # Prepare table data with wrapped text
    table_data = []
    for _, row in samples.iterrows():
        claim_wrapped = textwrap.fill(str(row['sentence']), width=60)
        evidence_wrapped = textwrap.fill(str(row['best_evidence']) if pd.notna(row['best_evidence']) else 'N/A', width=60)
        
        table_data.append([
            claim_wrapped,
            row['action_pred'],
            evidence_wrapped,
            f"{row['similarity_score']:.2f}"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['ESG Claim', 'Actionability', 'Semantic Evidence', 'Sim'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.4, 0.15, 0.4, 0.05])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 3)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    # plt.title('Claim-Evidence Linking Examples (Top 5)', fontsize=14, pad=20)
    plt.tight_layout()
    save_plot('claim_evidence_linking.png')

if __name__ == "__main__":
    print("="*60)
    print("Generating Ablation Study Charts")
    print("="*60)
    
    # 1. Component Correlation
    merged_data = generate_component_correlation()
    
    # 2. Evidence Impact Scatter
    generate_evidence_scatter(merged_data)
    
    # 3. Evidence Distribution
    generate_evidence_distribution()
    
    # 4. Linking Visualization
    generate_linking_viz()
    
    print("\\nAll charts generated successfully!")
