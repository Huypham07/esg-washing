"""Embed commitment chunks, compute BRI per (bank,year), test RQ4 vs CTI/NAR.

Embeddings are mean-centered corpus-wide before cosine (anisotropy fix). SBS was
dropped (raw topical cosine saturated; null vs every index) — see Phase 1 plan."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from scipy.stats import spearmanr

from esgwash.indices.alignment import signals_per_panel

PILLARS = ["is_env", "is_soc", "is_gov"]


def _esg_commit(clf: pd.DataFrame) -> pd.DataFrame:
    esg = clf[PILLARS].max(axis=1).astype(bool)
    return clf[(clf["is_commitment"] == 1) & esg].reset_index(drop=True)


def compute(classified: pd.DataFrame, embedder) -> pd.DataFrame:
    df = _esg_commit(classified)
    emb = embedder.embed(df["content_text"].astype(str).tolist())
    return signals_per_panel(df, emb)


def _load_classified() -> pd.DataFrame:
    files = sorted(glob.glob("outputs/cti/*/*/classified.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)


def main(out_dir: str = "experiments/panel") -> pd.DataFrame:
    from esgwash.corpus.sentence_embedder import SentenceEmbedder
    clf = _load_classified()
    sig = compute(clf, SentenceEmbedder())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sig.to_csv(out / "embedding_signals.csv", index=False)

    panel = pd.read_csv(out / "panel.csv").merge(sig, on=["bank", "year"], how="left")
    panel.to_csv(out / "panel_with_signals.csv", index=False)
    vb = panel.dropna(subset=["bri", "cti"])
    rho_c, p_c = spearmanr(vb["cti"], vb["bri"])
    rho_n, p_n = spearmanr(vb["nar"], vb["bri"])
    print(f"RQ4 (n={len(vb)}): boilerplate vs rubric")
    print(f"  Spearman(CTI, BRI) = {rho_c:.3f} (p={p_c:.2e})")
    print(f"  Spearman(NAR, BRI) = {rho_n:.3f} (p={p_n:.2e})")
    print(f"-> {out/'embedding_signals.csv'}")
    return panel


if __name__ == "__main__":
    main()
