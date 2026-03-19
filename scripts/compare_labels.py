import pandas as pd
from collections import Counter

weak = pd.read_parquet('data/labels/weak_labels.parquet')
llm = pd.read_parquet('data/labels/llm_prelabels.parquet')

print('='*70)
print('COMPARISON: WEAK LABELS (Option B) vs LLM LABELS (Option C)')
print('='*70)

merged = llm.merge(weak[['sent_id', 'weak_topic', 'weak_conf']], on='sent_id', how='left')
merged = merged.dropna(subset=['weak_topic'])

print(f'Overlapping samples: {len(merged)}')

agree = (merged['llm_topic'] == merged['weak_topic']).sum()
print(f'Agreement: {agree}/{len(merged)} ({100*agree/len(merged):.1f}%)')

print('\n--- DISTRIBUTION ---')
print(f'{"Topic":<15} {"LLM":>8} {"Weak":>8} {"LLM%":>7} {"Weak%":>7}')
print('-'*50)
for topic in ['E', 'S_labor', 'S_community', 'S_product', 'G', 'Non_ESG']:
    l = (merged['llm_topic'] == topic).sum()
    w = (merged['weak_topic'] == topic).sum()
    print(f'{topic:<15} {l:>8} {w:>8} {100*l/len(merged):>6.1f}% {100*w/len(merged):>6.1f}%')

print('\n--- TOP DISAGREEMENTS (Weak -> LLM) ---')
dis = merged[merged['llm_topic'] != merged['weak_topic']]
print(f'Total disagreements: {len(dis)} ({100*len(dis)/len(merged):.1f}%)')
for (w, l), c in Counter(zip(dis['weak_topic'], dis['llm_topic'])).most_common(10):
    print(f'  {w:<12} -> {l:<12}: {c}')

print('\n--- SAMPLE DISAGREEMENTS ---')
for topic in ['G', 'S_labor', 'E']:
    sample = dis[(dis['weak_topic'] == topic) & (dis['llm_topic'] == 'Non_ESG')].head(2)
    if len(sample) > 0:
        print(f'\n{topic} (weak) -> Non_ESG (llm):')
        for _, row in sample.iterrows():
            print(f'  "{row["sentence"][:80]}..."')
            print(f'  Reason: {row["llm_reason"]}')
