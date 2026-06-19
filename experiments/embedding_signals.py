"""Embed commitment chunks, compute SBS/BRI per (bank,year), test RQ4 vs CTI."""
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
    valid = panel.dropna(subset=["sbs", "cti"])
    rho_s, p_s = spearmanr(valid["cti"], 1 - valid["sbs"])
    vb = panel.dropna(subset=["bri", "cti"])
    rho_b, p_b = spearmanr(vb["cti"], vb["bri"])
    print(f"RQ4 convergent validity (n={len(valid)}):")
    print(f"  Spearman(CTI, 1-SBS) = {rho_s:.3f} (p={p_s:.2e})")
    print(f"  Spearman(CTI, BRI)   = {rho_b:.3f} (p={p_b:.2e})")
    print(f"-> {out/'embedding_signals.csv'}")
    return panel


if __name__ == "__main__":
    main()
