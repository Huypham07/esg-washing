"""CTI (commitment-to-implementation / cheap-talk index) + bootstrap CI + selective disclosure.

CTI(bank,year,pillar) = #{commit=1 & spec=0} / #{commit=1}  trong các câu thuộc pillar.
  -> tỉ lệ cam kết MƠ HỒ (không cụ thể). Cao = washing nặng.
  -> TỈ LỆ THUẦN 2 classifier, KHÔNG trọng số tay (khác EWRI) = luận điểm phòng thủ.

Input: enriched parquet từ classify_corpus (cột is_E/is_S/is_G, is_esg, commitment, specificity).
Chấm chéo: 1 câu thuộc nhiều pillar -> đếm cho MỌI pillar nó thuộc.
"""
import math

import numpy as np
import pandas as pd

PILLARS = ["E", "S", "G"]


def _norm_entropy(counts: list[int]) -> float:
    """Shannon entropy chuẩn hoá [0,1]: 1=phân bố đều E/S/G, 0=dồn 1 pillar."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    if len(probs) <= 1:
        return 0.0
    h = -sum(p * math.log2(p) for p in probs)
    return h / math.log2(len(probs))


def compute_cti(enriched: pd.DataFrame, min_commit: int = 30,
                bootstrap_B: int = 1000, seed: int = 42,
                group_cols: tuple = ("bank", "year")) -> pd.DataFrame:
    """CTI per (group_cols × pillar) + bootstrap CI 95%. Ô n_commit<min_commit -> low_n.
    group_cols=("bank",) -> POOLED gộp năm (ranking CHÍNH Phase 05); ("bank","year") -> per-year (phụ lục).
    LƯU Ý: CTI tuyệt đối lệch CAO (cascade + specificity recall) -> dùng SO SÁNH TƯƠNG ĐỐI."""
    rng = np.random.default_rng(seed)
    gc = list(group_cols)
    rows = []
    for pillar in PILLARS:
        sub = enriched[enriched[f"is_{pillar}"] == 1]
        for keys, g in sub.groupby(gc):
            keys = keys if isinstance(keys, tuple) else (keys,)
            rec = {c: (int(v) if c == "year" else v) for c, v in zip(gc, keys)}
            rec["pillar"] = pillar
            commits = g[g["commitment"] == 1]
            n_commit = len(commits)
            rec.update(n_pillar=int(len(g)), n_commit=int(n_commit))
            if n_commit == 0:
                rec.update(n_vague=0, cti=np.nan, ci_low=np.nan, ci_high=np.nan, low_n=True)
                rows.append(rec)
                continue
            vague = (commits["specificity"] == 0).to_numpy().astype(float)  # 1 = cam kết mơ hồ
            cti = float(vague.mean())
            if n_commit >= 2:
                boot = rng.choice(vague, size=(bootstrap_B, n_commit), replace=True).mean(axis=1)
                lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
            else:
                lo = hi = cti
            rec.update(n_vague=int(vague.sum()), cti=round(cti, 4),
                       ci_low=round(lo, 4), ci_high=round(hi, 4), low_n=bool(n_commit < min_commit))
            rows.append(rec)
    return pd.DataFrame(rows).sort_values(["pillar", "cti"], ascending=[True, False]).reset_index(drop=True)


def cti_sensitivity(enriched: pd.DataFrame, spec_thresholds=(0.4, 0.5, 0.6),
                    min_commit: int = 30) -> dict:
    """Sensitivity: quét ngưỡng SPECIFICITY (commit cố định 0.5) -> CTI pooled per (bank,pillar)
    -> Kendall τ ranking vs ngưỡng 0.5. Đo ranking có ổn định với ngưỡng specificity không
    (specificity recall thấp = bias chính của CTI). τ≈1 = ranking ổn -> so sánh tương đối đáng tin."""
    from scipy.stats import kendalltau
    rankings = {}
    for thr in spec_thresholds:
        recs = []
        for pillar in PILLARS:
            sub = enriched[(enriched[f"is_{pillar}"] == 1) & (enriched["commitment"] == 1)]
            for bank, g in sub.groupby("bank"):
                if len(g) < min_commit:
                    continue
                recs.append({"bank": bank, "pillar": pillar,
                             "cti": float((g["p_spec"] < thr).mean())})  # p_spec<thr => mơ hồ
        rankings[thr] = pd.DataFrame(recs, columns=["bank", "pillar", "cti"])  # cột cố định -> rỗng vẫn an toàn

    base = 0.5 if 0.5 in spec_thresholds else spec_thresholds[len(spec_thresholds) // 2]
    out = {"spec_thresholds": list(spec_thresholds), "base": base, "kendall_tau": {}}
    for pillar in PILLARS:
        rb = rankings[base][rankings[base].pillar == pillar].set_index("bank")["cti"]
        taus = {}
        for thr in spec_thresholds:
            if thr == base:
                continue
            rt = rankings[thr][rankings[thr].pillar == pillar].set_index("bank")["cti"]
            common = rb.index.intersection(rt.index)
            if len(common) >= 3:
                tau, _ = kendalltau(rb[common], rt[common])
                taus[str(thr)] = round(float(tau), 3)
        out["kendall_tau"][pillar] = {"n_banks": int(len(rb)), "tau_vs_base": taus}
    return out


def selective_disclosure(enriched: pd.DataFrame) -> pd.DataFrame:
    """Phân bố câu/commit theo E/S/G mỗi (bank,year) + entropy + cờ né-pillar."""
    esg = enriched[enriched.is_esg == 1]
    rows = []
    for (bank, year), g in esg.groupby(["bank", "year"]):
        sent = {p: int((g[f"is_{p}"] == 1).sum()) for p in PILLARS}
        com = {p: int(((g[f"is_{p}"] == 1) & (g["commitment"] == 1)).sum()) for p in PILLARS}
        total = sum(sent.values())
        shares = {p: (sent[p] / total if total else 0.0) for p in PILLARS}
        rec = {"bank": bank, "year": int(year), "n_esg": int(len(g))}
        rec.update({f"n_{p}": sent[p] for p in PILLARS})
        rec.update({f"commit_{p}": com[p] for p in PILLARS})
        rec.update({f"share_{p}": round(shares[p], 3) for p in PILLARS})
        rec["entropy"] = round(_norm_entropy(list(sent.values())), 3)
        rec["skew_flag"] = bool(total and min(shares.values()) < 0.10)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["bank", "year"]).reset_index(drop=True)


def print_cti_summary(df_cti: pd.DataFrame, df_sd: pd.DataFrame,
                      title: str = "CTI SUMMARY (gold, CTI THUẦN — ranking TƯƠNG ĐỐI, SƠ BỘ chờ validation)") -> None:
    has_year = "year" in df_cti.columns

    def _r(r):
        yr = f" {int(r.year)}" if has_year else ""
        return (f"  {r.bank:12s}{yr} [{r.pillar}]  CTI={r.cti:.2f}  "
                f"CI=[{r.ci_low:.2f},{r.ci_high:.2f}]  n_commit={r.n_commit}")

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    valid = df_cti[(~df_cti.low_n) & df_cti.cti.notna()]
    print(f"\nÔ: {len(df_cti)} | hợp lệ (n_commit>=min): {len(valid)} | low_n: {int(df_cti.low_n.sum())}")
    print("\nCTI trung bình theo pillar (ô hợp lệ) — cao = nhiều cam kết mơ hồ:")
    for p in PILLARS:
        v = valid[valid.pillar == p].cti
        if len(v):
            print(f"  {p}: mean={v.mean():.3f}  median={v.median():.3f}  range=[{v.min():.2f},{v.max():.2f}]  n_ô={len(v)}")
        else:
            print(f"  {p}: (không đủ ô hợp lệ)")
    print("\nTop CTI CAO (washing nặng nhất — TƯƠNG ĐỐI):")
    for _, r in valid.sort_values("cti", ascending=False).head(5).iterrows():
        print(_r(r))
    print("\nTop CTI THẤP:")
    for _, r in valid.sort_values("cti").head(5).iterrows():
        print(_r(r))
    skew = df_sd[df_sd.skew_flag]
    print(f"\nSelective disclosure — {len(skew)}/{len(df_sd)} ô lệch mạnh (1 pillar <10% câu ESG):")
    for _, r in skew.sort_values("entropy").head(8).iterrows():
        print(f"  {r.bank:12s} {int(r.year)}  câu E/S/G={r.n_E}/{r.n_S}/{r.n_G}  entropy={r.entropy:.2f}")


def summarize_to_dict(df_cti: pd.DataFrame, df_sd: pd.DataFrame, track: str) -> dict:
    valid = df_cti[(~df_cti.low_n) & df_cti.cti.notna()]
    out = {
        "track": track,
        "note": "CTI THUẦN gold (translate-train). Ranking TƯƠNG ĐỐI (CTI tuyệt đối lệch cao do cascade+specificity). "
                "SƠ BỘ — chờ Phase 04 (verify classifier trên VN thật) + Phase 06 (validation chỉ số).",
        "n_cells": int(len(df_cti)), "n_valid": int(len(valid)), "n_low_n": int(df_cti.low_n.sum()),
        "selective_disclosure_skew_cells": int(df_sd.skew_flag.sum()),
        "cti_by_pillar": {},
    }
    for p in PILLARS:
        v = valid[valid.pillar == p].cti
        out["cti_by_pillar"][p] = (
            {"mean": round(float(v.mean()), 4), "median": round(float(v.median()), 4), "n": int(len(v))}
            if len(v) else None
        )
    return out
