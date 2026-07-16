"""Build Kaggle inputs cho P3c ablation (CPU, local). Chay: python experiments/kaggle/build_kaggle_inputs.py

Xuat experiments/kaggle/:
  - c2c3_input.parquet     : 390 committed (chunk_id, content_text, c1_gold_level) — c1_gold_level de check agreement voi C1-live
  - inv_vague_seeds.json    : cau vague (gold L0) cho INV
  - ablation_c3_fewshot.json: copy 6 exemplar
Assert: committed set == 390, chunk_id unique (finding review: chan sai chunk-set truoc khi burn paid run).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/kaggle"


def main() -> None:
    gc = pd.read_parquet(ROOT / "experiments/eval/gold_classified.parquet")[
        ["chunk_id", "content_text", "spec_level"]].rename(columns={"spec_level": "c1_gold_level"})
    A = pd.read_excel(ROOT / "data/gold_annot_1_relabeled.xlsx")[
        ["chunk_id", "g_spec_level", "g_is_commit"]]
    m = gc.merge(A, on="chunk_id", how="inner")
    assert len(m) == len(gc) == 400, f"merge loss: {len(m)}"

    com = m[m["g_is_commit"] == 1].copy()
    # GUARD: committed set == 390, chunk_id unique (khong duoc lech sample voi C1_full/majority downstream)
    assert len(com) == 390, f"committed != 390: {len(com)}"
    assert com["chunk_id"].is_unique, "chunk_id NOT unique"

    com[["chunk_id", "content_text", "c1_gold_level"]].to_parquet(OUT / "c2c3_input.parquet", index=False)
    print(f"c2c3_input.parquet: {len(com)} committed chunks (+ c1_gold_level for agreement check)")

    vague = com.loc[com["g_spec_level"] == 0, ["chunk_id", "content_text"]]
    vague.to_json(OUT / "inv_vague_seeds.json", orient="records", force_ascii=False, indent=2)
    print(f"inv_vague_seeds.json: {len(vague)} gold-L0 vague seeds")

    shutil.copy(ROOT / "experiments/prompts/ablation_c3_fewshot.json", OUT / "ablation_c3_fewshot.json")
    # committed chunk_id list -> notebook assert (finding 9)
    (OUT / "committed_chunk_ids.json").write_text(
        json.dumps(sorted(com["chunk_id"].astype(str).tolist()), ensure_ascii=False), encoding="utf-8")
    print(f"committed_chunk_ids.json: {len(com)} ids")
    print("kaggle inputs rebuilt.")


if __name__ == "__main__":
    main()
