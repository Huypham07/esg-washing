"""CLI demo P05: corpus -> classify 5 model -> CTI + selective disclosure -> parquet+json+print.

Chạy:
  python src/pipeline/run_cti.py                       # full corpus, GPU, track theo config
  python src/pipeline/run_cti.py --limit 300 --device cpu   # smoke toàn chuỗi
  python src/pipeline/run_cti.py --track gold          # swap sang gold (sau Phase 04)
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipeline.classify_corpus import classify_corpus, load_cti_config
from src.pipeline.cti import (compute_cti, cti_sensitivity, print_cti_summary,
                               selective_disclosure, summarize_to_dict)


def main(args=None) -> None:
    p = argparse.ArgumentParser(description="P05 demo: chạy CTI trên corpus")
    p.add_argument("--config", default="config/cti.yml")
    p.add_argument("--track", default=None, help="silver | gold")
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=0, help="0=full; >0=smoke")
    p.add_argument("--no-cache", action="store_true")
    a = p.parse_args(args)

    config = load_cti_config(a.config)
    track = a.track or config["track"]

    enriched = classify_corpus(config, track=track, limit=a.limit,
                               device=a.device, use_cache=not a.no_cache)

    cti_cfg = config.get("cti", {})
    min_commit = cti_cfg.get("min_commit", 30)
    B = cti_cfg.get("bootstrap_B", 1000)
    seed = config.get("seed", 42)
    spec_thr = tuple(cti_cfg.get("spec_thresholds", [0.4, 0.5, 0.6]))
    exclude = set(cti_cfg.get("exclude_banks", []))

    # Loại bank artifact (vd bsc OCR mất dấu) khỏi ranking CHÍNH (pooled) + sensitivity
    enriched_main = enriched[~enriched["bank"].isin(exclude)] if exclude else enriched

    df_pooled = compute_cti(enriched_main, min_commit=min_commit, bootstrap_B=B, seed=seed, group_cols=("bank",))
    df_year = compute_cti(enriched, min_commit=min_commit, bootstrap_B=B, seed=seed, group_cols=("bank", "year"))
    df_sd = selective_disclosure(enriched)
    sens = cti_sensitivity(enriched_main, spec_thresholds=spec_thr, min_commit=min_commit)

    out_dir = Path(config["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = f"_smoke_{track}" if a.limit else f"_{track}"
    df_pooled.to_parquet(out_dir / f"cti_pooled{sfx}.parquet", index=False)   # ranking CHÍNH
    df_year.to_parquet(out_dir / f"cti{sfx}.parquet", index=False)            # per-year (phụ lục)
    df_sd.to_parquet(out_dir / f"selective_disclosure{sfx}.parquet", index=False)
    summary = summarize_to_dict(df_pooled, df_sd, track)
    summary["excluded_banks"] = sorted(exclude)
    summary["sensitivity"] = sens
    (out_dir / f"cti_summary{sfx}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print_cti_summary(df_pooled, df_sd,
                      title=f"CTI POOLED (bank×trụ, gộp năm) — track={track} · ranking TƯƠNG ĐỐI · loại {sorted(exclude)}")
    print("\nSensitivity (Kendall τ ranking vs ngưỡng 0.5 specificity — gần 1 = ổn định):")
    for pillar, d in sens["kendall_tau"].items():
        print(f"  {pillar}: n_banks={d['n_banks']}  τ={d['tau_vs_base']}")
    print(f"\nSaved -> {out_dir} (cti_pooled{sfx} · cti{sfx} · selective_disclosure{sfx} · cti_summary{sfx})")


if __name__ == "__main__":
    main()
