import pandas as pd
pd.set_option('display.max_colwidth', 120)

blocks = pd.read_parquet('data/corpus/blocks.parquet')
sents = pd.read_parquet('data/corpus/sentences.parquet')

print('='*80)
print('SAMPLE BLOCKS (các loại block_type)')
print('='*80)

for btype in ['paragraph', 'kpi_like', 'heading_like', 'table_like', 'bullet_like', 'meta_heading']:
    subset = blocks[blocks['block_type']==btype]
    if len(subset) == 0:
        continue
    sample = subset.sample(1, random_state=42).iloc[0]
    print(f'\n--- {btype.upper()} ({len(subset):,} blocks) ---')
    print(f'doc_id: {sample.doc_id} | section: {sample.section_title[:50]}')
    print(f'block_text:\n{sample.block_text[:350]}')
    print()

print('='*80)
print('SAMPLE SENTENCES (với context)')
print('='*80)

good_sents = sents[(sents['ctx_prev'].str.len() > 20) & (sents['ctx_next'].str.len() > 20)]
samples = good_sents.sample(3, random_state=123)

for i, (_, s) in enumerate(samples.iterrows(), 1):
    print(f'\n--- Sample {i} ({s.doc_id}, {s.block_type}) ---')
    print(f'section: {s.section_title[:60]}')
    print(f'[prev] {s.ctx_prev[:100]}...')
    print(f'[SENT] {s.sentence}')
    print(f'[next] {s.ctx_next[:100]}...')

print('\n' + '='*80)
print('CHECK: section_title còn ## không?')
print('='*80)
has_md = blocks[blocks['section_title'].str.contains(r'^##', regex=True, na=False)]
print(f'Blocks với section_title có "##": {len(has_md)}')
if len(has_md) > 0:
    print(has_md['section_title'].head(5).tolist())
