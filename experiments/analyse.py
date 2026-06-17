"""Phan tich tong hop ket qua demo (bidv 2023/2024) -> experiments/analyse_bidv/.

Xuat: report.md (toan dien) + summary.json + cac bieu do PNG + bang vi du dung/nham.
  python experiments/analyse_demo.py                 # mac dinh bidv 2023,2024
  python experiments/analyse_demo.py --bank bidv --years 2023 2024
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from esgwash.run import to_long  # noqa: E402

PILLARS = ["env", "soc", "gov"]
PLABEL = {"env": "E (Môi trường)", "soc": "S (Xã hội)", "gov": "G (Quản trị)"}
LVL = {0: "Mức 0 — mơ hồ", 1: "Mức 1 — cụ thể", 2: "Mức 2 — định lượng"}
OUT = Path("experiments/analyse_bidv")

# Heuristic phat hien NGHI NHAM (khong phai ground-truth, chi de soi)
RE_NONACTOR = re.compile(r"(?:NHNN|Ngân hàng Nhà nước|Chính phủ|toàn ngành|toàn nền kinh tế|"
                         r"quốc gia|GDP|lạm phát|Quốc hội)", re.I)
RE_FIN = re.compile(r"(?:lợi nhuận|LNTT|ROA|ROE|vốn điều lệ|vốn hóa|cổ phiếu|EPS|CASA|"
                    r"tổng tài sản|nợ xấu)", re.I)
RE_NAMED = re.compile(r"(?:Basel|ISO\s?\d|L/C|Open API|SmartBanking|BIDV Home|B\.One|"
                      r"Core Banking|Payment Hub|Edmond|VIETGAP|SA-CCR|iBank)")


def load(bank: str, year: int) -> dict:
    d = Path(f"outputs/cti/{bank}/{year}")
    clf = pd.read_parquet(d / "classified.parquet")
    gr = pd.read_parquet(d / "claims_grounded.parquet") if (d / "claims_grounded.parquet").exists() else None
    return {"year": year, "clf": clf, "gr": gr}


def pillar_table(clf: pd.DataFrame) -> pd.DataFrame:
    """Per pillar: n_commit, dem theo spec_level, CTI_loose (Muc0), CTI_strict (Muc0+1), share."""
    cm = clf[clf["is_commitment"] == 1]
    long = to_long(cm)
    esg_all = to_long(clf)  # tat ca chunk ESG (ke ca khong commit) -> selective disclosure
    rows = []
    for p in PILLARS:
        g = long[long["pillar"] == p]
        n = len(g)
        l0 = int((g["spec_level"] == 0).sum())
        l1 = int((g["spec_level"] == 1).sum())
        l2 = int((g["spec_level"] == 2).sum())
        rows.append({
            "pillar": p, "n_commit": n, "lvl0": l0, "lvl1": l1, "lvl2": l2,
            "cti_loose": round(l0 / n, 4) if n else np.nan,
            "cti_strict": round((l0 + l1) / n, 4) if n else np.nan,
            "n_esg_chunks": int((esg_all["pillar"] == p).sum()),
        })
    t = pd.DataFrame(rows)
    t["disclosure_share"] = (t["n_esg_chunks"] / t["n_esg_chunks"].sum()).round(4)
    return t


def overall_esg(clf: pd.DataFrame) -> dict:
    """CTI tong tren ESG: chunk commitment co >=1 tru ESG (dem DUY NHAT, khong cong trung tru)."""
    cm = clf[clf["is_commitment"] == 1]
    esg = cm[cm[[f"is_{p}" for p in PILLARS]].sum(axis=1) > 0]
    n = len(esg)
    l0 = int((esg["spec_level"] == 0).sum())
    l1 = int((esg["spec_level"] == 1).sum())
    l2 = int((esg["spec_level"] == 2).sum())
    return {"n_commit_esg": n, "lvl0": l0, "lvl1": l1, "lvl2": l2,
            "cti_loose": round(l0 / n, 4) if n else None,
            "cti_strict": round((l0 + l1) / n, 4) if n else None}


# gCTI-L1: grounding-aware, nam GIUA dai [loose, strict]. Band theo l1_sim_floor cua cac run L1.
L1_ROOTS = {"floor0.50": "outputs/cti_b2", "floor0.25": "outputs/cti_b2_floor"}


def load_l1_support(root: str, bank: str, year: int):
    """Series support theo chunk_index tu run L1 (claims_grounded). None neu chua co."""
    p = Path(root) / bank / str(year) / "claims_grounded.parquet"
    return pd.read_parquet(p).set_index("chunk_index")["support"] if p.exists() else None


def gcti_l1(clf: pd.DataFrame, sup, theta_l1: float) -> dict | None:
    """gCTI-L1: refine RIÊNG Mức 1 bằng grounding (Mức 2 giữ substantive như loose/strict).
    cheap = Muc0 | (Muc1 & sup<θL1) -> NẰM GIỮA [CTI_loose, CTI_strict]. Aggregate ESG + per-tru.
    (Full gCTI ground cả Mức 2 -> xem cti.parquet/compare_arms.)"""
    if sup is None:
        return None
    cm = clf[clf["is_commitment"] == 1].copy()
    cm["_sup"] = cm["chunk_index"].map(sup).fillna(0.0)

    def _rate(sub):
        if not len(sub):
            return None
        sl, s = sub["spec_level"].to_numpy(), sub["_sup"].to_numpy()
        cheap = (sl == 0) | ((sl == 1) & (s < theta_l1))
        return round(float(cheap.mean()), 4)

    esg = cm[cm[[f"is_{p}" for p in PILLARS]].sum(axis=1) > 0]
    long = to_long(cm)
    return {"esg": _rate(esg), "per_pillar": {p: _rate(long[long["pillar"] == p]) for p in PILLARS}}


def basic_stats(clf: pd.DataFrame) -> dict:
    n = len(clf)
    n_esg = int(clf[[f"is_{p}" for p in PILLARS]].sum(axis=1).gt(0).sum())
    cm = clf[clf["is_commitment"] == 1]
    return {"n_chunks": n, "n_esg_chunks": n_esg, "esg_share": round(n_esg / n, 3),
            "n_commitment": int(len(cm)), "commit_share": round(len(cm) / n, 3),
            "n_commit_esg": int((cm[[f"is_{p}" for p in PILLARS]].sum(axis=1) > 0).sum())}


def examples(clf: pd.DataFrame, year: int) -> dict:
    """Lay vi du DUNG (minh hoa) va NGHI NHAM (heuristic) theo nhom."""
    cm = clf[clf["is_commitment"] == 1].copy()

    def items_of(r, key):
        try:
            j = json.loads(r["spec_rubric"])
        except (TypeError, ValueError):
            return []
        return [it.get("action_or_event") for it in j.get("items", []) if it.get(key)]

    def pack(sub, note_fn, k=6):
        rows = []
        for _, r in sub.head(k).iterrows():
            rows.append({"chunk_index": int(r["chunk_index"]), "spec_level": int(r["spec_level"]),
                         "pillar(e/s/g)": f"{r['is_env']}/{r['is_soc']}/{r['is_gov']}",
                         "text": str(r["content_text"])[:240].replace("\n", " "),
                         "note": note_fn(r)})
        return rows

    out = {}
    # ---- DUNG (minh hoa) ----
    m2 = cm[(cm.spec_level == 2) & (cm[["is_env", "is_soc", "is_gov"]].sum(axis=1) > 0)]
    out["correct_lvl2"] = pack(m2, lambda r: "quant items: " + str(items_of(r, "is_quantified")[:3]))
    m1 = cm[(cm.spec_level == 1) & cm.content_text.astype(str).str.contains(RE_NAMED)]
    out["correct_lvl1"] = pack(m1, lambda r: "concrete items: " + str(items_of(r, "is_concrete_action")[:3]))
    m0 = cm[(cm.spec_level == 0) & ~cm.content_text.astype(str).str.contains(RE_NAMED)]
    out["correct_lvl0"] = pack(m0, lambda r: "đúng mơ hồ (không tên riêng/số)")
    # ---- NGHI NHAM (heuristic) ----
    attr = cm[(cm.spec_level == 2) & cm.content_text.astype(str).str.contains(RE_NONACTOR)]
    out["suspect_attribution"] = pack(attr, lambda r: "Mức 2 nhưng có NHNN/quốc gia -> số có thể KHÔNG của BIDV")
    ofire = cm[(cm[["is_env", "is_soc", "is_gov"]].sum(axis=1) == 0)
               & cm.content_text.astype(str).str.contains(RE_FIN)]
    out["suspect_commit_overfire"] = pack(ofire, lambda r: "commitment nhưng là KẾT QUẢ tài chính (không trụ ESG)")
    fn = cm[(cm.spec_level == 0) & cm.content_text.astype(str).str.contains(RE_NAMED)]
    out["suspect_lvl0_missed"] = pack(fn, lambda r: "Mức 0 nhưng CÓ tên riêng -> đáng lẽ Mức 1")
    return out


def fig_cti_band(tables: dict, gl1: dict | None = None):
    fig, axes = plt.subplots(1, len(tables), figsize=(6 * len(tables), 4.2), squeeze=False)
    for ax, (yr, t) in zip(axes[0], tables.items()):
        x = np.arange(len(PILLARS))
        ax.bar(x - 0.2, t["cti_loose"], 0.4, label="CTI_loose (Mức 0)", color="tab:green")
        ax.bar(x + 0.2, t["cti_strict"], 0.4, label="CTI_strict (Mức 0+1)", color="tab:red", alpha=.8)
        for i, r in t.iterrows():
            ax.text(i - 0.2, r["cti_loose"] + .02, f"{r['cti_loose']:.2f}", ha="center", fontsize=8)
            ax.text(i + 0.2, r["cti_strict"] + .02, f"{r['cti_strict']:.2f}", ha="center", fontsize=8)
        if gl1 and yr in gl1:  # gCTI-L1 (P3-L1): điểm data-driven + band l1_floor, NẰM GIỮA loose/strict
            for i, p in enumerate(PILLARS):
                vals = [v for v in gl1[yr].get(p, []) if v is not None]
                if not vals:
                    continue
                lo, hi = min(vals), max(vals)
                ax.errorbar(i, (lo + hi) / 2, yerr=[[(hi - lo) / 2], [(hi - lo) / 2]], fmt="D",
                            color="black", capsize=5, markersize=7, zorder=5,
                            label="gCTI-L1 (band l1_floor)" if i == 0 else None)
                ax.text(i, hi + .03, f"{lo:.2f}–{hi:.2f}", ha="center", fontsize=7, color="black")
        ax.set_xticks(x); ax.set_xticklabels([p.upper() for p in PILLARS])
        ax.set_ylim(0, 1); ax.set_title(f"CTI band + gCTI-L1 — {yr}"); ax.set_ylabel("tỉ lệ cheap talk")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "cti_band.png", dpi=120); plt.close(fig)


def fig_spec_level(tables: dict):
    fig, axes = plt.subplots(1, len(tables), figsize=(6 * len(tables), 4.2), squeeze=False)
    for ax, (yr, t) in zip(axes[0], tables.items()):
        x = np.arange(len(PILLARS))
        b0 = t["lvl0"].to_numpy(); b1 = t["lvl1"].to_numpy(); b2 = t["lvl2"].to_numpy()
        ax.bar(x, b0, label="Mức 0 mơ hồ", color="tab:red")
        ax.bar(x, b1, bottom=b0, label="Mức 1 cụ thể", color="gold")
        ax.bar(x, b2, bottom=b0 + b1, label="Mức 2 định lượng", color="tab:green")
        ax.set_xticks(x); ax.set_xticklabels([p.upper() for p in PILLARS])
        ax.set_title(f"Phân bố specificity 3 mức — {yr}"); ax.set_ylabel("số chunk commitment")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "spec_level.png", dpi=120); plt.close(fig)


def fig_disclosure(tables: dict):
    fig, axes = plt.subplots(1, len(tables), figsize=(5 * len(tables), 4.2), squeeze=False)
    for ax, (yr, t) in zip(axes[0], tables.items()):
        ax.bar([p.upper() for p in PILLARS], t["n_esg_chunks"], color=["tab:blue", "tab:orange", "tab:purple"])
        for i, v in enumerate(t["n_esg_chunks"]):
            ax.text(i, v + 1, f"{v}\n({t['disclosure_share'].iloc[i]*100:.0f}%)", ha="center", fontsize=8)
        ax.set_title(f"Selective disclosure — số chunk ESG/trụ — {yr}"); ax.set_ylabel("số chunk")
    fig.tight_layout(); fig.savefig(OUT / "selective_disclosure.png", dpi=120); plt.close(fig)


def md_table(t: pd.DataFrame) -> str:
    cols = ["pillar", "n_commit", "lvl0", "lvl1", "lvl2", "cti_loose", "cti_strict",
            "n_esg_chunks", "disclosure_share"]
    head = "| " + " | ".join(cols) + " |\n| " + " | ".join(["---"] * len(cols)) + " |\n"
    body = "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n" for _, r in t.iterrows())
    return head + body


def md_examples(title: str, rows: list) -> str:
    if not rows:
        return f"\n**{title}** — (không có)\n"
    s = f"\n**{title}**\n\n| idx | lvl | E/S/G | text | ghi chú |\n| --- | --- | --- | --- | --- |\n"
    for r in rows:
        txt = r["text"].replace("|", "/")
        s += f"| {r['chunk_index']} | {r['spec_level']} | {r['pillar(e/s/g)']} | {txt} | {r['note']} |\n"
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="bidv")
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    ap.add_argument("--theta-l1", type=float, default=0.4)
    ap.add_argument("--theta-l2", type=float, default=0.7)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    data = [load(args.bank, y) for y in args.years]
    tables = {d["year"]: pillar_table(d["clf"]) for d in data}
    summary = {"bank": args.bank, "years": {}}

    for d in data:
        yr = d["year"]
        gl1 = {lbl: gcti_l1(d["clf"], load_l1_support(root, args.bank, yr), args.theta_l1)
               for lbl, root in L1_ROOTS.items()}
        summary["years"][yr] = {
            "basic": basic_stats(d["clf"]),
            "overall_esg": overall_esg(d["clf"]),
            "gcti_l1": {"theta_l1": args.theta_l1, "theta_l2": args.theta_l2, "by_floor": gl1},
            "per_pillar": tables[yr].to_dict(orient="records"),
        }

    gl1_tables = {}
    for d in data:
        yr = d["year"]
        bf = summary["years"][yr].get("gcti_l1", {}).get("by_floor", {})
        gl1_tables[yr] = {p: [bf[lbl]["per_pillar"][p] for lbl in bf if bf[lbl]] for p in PILLARS}
    fig_cti_band(tables, gl1_tables); fig_spec_level(tables); fig_disclosure(tables)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- report.md ----
    R = [f"# Phân tích ESG-washing — {args.bank.upper()} ({', '.join(map(str, args.years))})", ""]
    R += ["> Đơn vị = **chunk** (≤256 token). CTI = tỉ lệ cam kết \"cheap talk\". Specificity 3 mức: "
          "Mức 0 mơ hồ / Mức 1 cụ thể (hành động có tên) / Mức 2 định lượng. "
          "**CTI_loose** = chỉ Mức 0; **CTI_strict** = Mức 0+1 (chỉ định lượng mới tính thực chất). "
          "Sự thật nằm trong dải [loose, strict].", ""]

    for d in data:
        yr = d["year"]; b = summary["years"][yr]["basic"]; o = summary["years"][yr]["overall_esg"]
        R += [f"## {yr}", "",
              f"- Chunk: **{b['n_chunks']}** | ESG: **{b['n_esg_chunks']}** ({b['esg_share']*100:.0f}%) "
              f"| commitment: **{b['n_commitment']}** ({b['commit_share']*100:.0f}%) | commitment-ESG: **{b['n_commit_esg']}**",
              "",
              f"### CTI tổng trên ESG (gộp trụ, đếm chunk duy nhất, n={o['n_commit_esg']})",
              f"- Phân bố: Mức0={o['lvl0']} · Mức1={o['lvl1']} · Mức2={o['lvl2']}",
              f"- **CTI_loose = {o['cti_loose']}** · **CTI_strict = {o['cti_strict']}**  → dải cheap-talk ESG",
              "",
              "### CTI & selective disclosure theo trụ", "",
              md_table(tables[yr]), ""]
        gl = summary["years"][yr].get("gcti_l1", {})
        R += [f"### gCTI-L1 (refine Mức 1 bằng grounding, θ_L1={gl.get('theta_l1')})",
              "> L1 grounding chấm TỪNG hành động Mức 1 (có được corroborate trong báo cáo không) → thay vì "
              "đoán \"tất cả Mức 1 thực chất\" (loose) hay \"tất cả cheap\" (strict), gCTI-L1 là **điểm NẰM "
              "GIỮA dải [CTI_loose, CTI_strict]**. Band theo `l1_sim_floor`. (Mức 2 giữ substantive ở đây; "
              "full gCTI ground cả Mức 2 ở cti.parquet/compare_arms.) Proxy nhiễu — xem report P3-L1.", ""]
        for lbl, v in (gl.get("by_floor") or {}).items():
            if v:
                pp = v["per_pillar"]
                R += [f"- **{lbl}** → gCTI-L1 (ESG) = **{v['esg']}**  ·  env={pp['env']} soc={pp['soc']} gov={pp['gov']}"]
            else:
                R += [f"- **{lbl}** → (chưa có run L1 ở `{L1_ROOTS[lbl]}`)"]
        R += [""]
        ex = examples(d["clf"], yr)
        R += ["### Ví dụ phân loại ĐÚNG (minh hoạ)"]
        R += [md_examples("Mức 2 — định lượng quy về BIDV", ex["correct_lvl2"])]
        R += [md_examples("Mức 1 — hành động/công cụ có tên", ex["correct_lvl1"])]
        R += [md_examples("Mức 0 — mơ hồ thật sự", ex["correct_lvl0"])]
        R += ["### Ví dụ NGHI NHẦM (heuristic — cần kiểm tay)"]
        R += [md_examples("Nghi sai attribution (số của NHNN/quốc gia)", ex["suspect_attribution"])]
        R += [md_examples("Commitment bắt nhầm kết quả tài chính", ex["suspect_commit_overfire"])]
        R += [md_examples("Mức 0 bỏ sót tên riêng (đáng lẽ Mức 1)", ex["suspect_lvl0_missed"])]
        R += [""]

    R += ["## Hình", "",
          "- `cti_band.png` — CTI_loose vs CTI_strict theo trụ",
          "- `spec_level.png` — phân bố 3 mức specificity theo trụ",
          "- `selective_disclosure.png` — số chunk ESG mỗi trụ (né chủ đề khó?)", ""]
    (OUT / "report.md").write_text("\n".join(R), encoding="utf-8")

    print(f"-> {OUT}/ : report.md, summary.json, cti_band.png, spec_level.png, selective_disclosure.png")
    for yr in args.years:
        o = summary["years"][yr]["overall_esg"]
        print(f"  {yr}: CTI_ESG dải [{o['cti_loose']}, {o['cti_strict']}] (n={o['n_commit_esg']})")


if __name__ == "__main__":
    main()
