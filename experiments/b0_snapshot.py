"""Phase 00 (P3-L1) — chốt baseline B0 từ output ĐÃ CÓ (không chạy lại pipeline, không GPU).

Đọc outputs/cti/<bank>/<year>/{cti.parquet, info_check.json} + git-SHA + model id (từ configs)
-> plans/20260616-p3-level1-grounding/b0_snapshot.json. Làm mốc so sánh cho B1/B2 (xem phase-00).

  python experiments/b0_snapshot.py
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

CTI_ROOT = Path("outputs/cti")
CONFIG_DIR = Path("configs")
OUT = Path("plans/20260616-p3-level1-grounding/b0_snapshot.json")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def model_ids() -> dict:
    """Model thực dùng ở P2-P3 (topic từ run.HF_REPOS; còn lại từ configs)."""
    def cfg(name: str) -> dict:
        p = CONFIG_DIR / f"{name}.yml"
        return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    g = cfg("grounding")
    return {
        "topic": "huypham71/esg-topic",
        "commitment": cfg("commitment").get("model"),
        "specificity": cfg("specificity").get("model"),
        "retriever": g.get("retriever"),
        "nli": g.get("nli_model"),
    }


def collect_cells() -> list[dict]:
    """Mỗi (bank, year) đã chạy: bảng cti + info_check (chỉ đọc, không tính lại)."""
    rows = []
    for cti_path in sorted(CTI_ROOT.glob("*/*/cti.parquet")):
        cell_dir = cti_path.parent
        bank, year = cell_dir.parent.name, cell_dir.name
        cti = pd.read_parquet(cti_path)
        info_path = cell_dir / "info_check.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        rows.append({
            "bank": bank, "year": int(year),
            "cti_table": json.loads(cti.round(4).to_json(orient="records")),
            "info_check": info,
        })
    return rows


def main() -> None:
    cells = collect_cells()
    snap = {
        "label": "B0 (baseline — P3 Mức2-only, nguyên trạng)",
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "model_ids": model_ids(),
        "theta_main": 0.7,
        "n_cells": len(cells),
        "cells": cells,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"B0 snapshot -> {OUT} ({len(cells)} cell, git {snap['git_sha'][:8]})")
    if not cells:
        print("  CẢNH BÁO: chưa có outputs/cti/*/*/cti.parquet — cần chạy esgwash.run trước.")
    for c in cells:
        ic = c["info_check"]
        print(f"  {c['bank']} {c['year']}: n_commit={ic.get('n_commitment')} "
              f"spec_level={ic.get('spec_level_counts')} "
              f"no_evid_share={ic.get('commit_no_evidence_share')}")
        for r in c["cti_table"]:
            print(f"    {r.get('pillar')}: cti={r.get('cti')} "
                  f"gcti@0.7={r.get('gcti@0.7')} n_commit={r.get('n_commit')}")


if __name__ == "__main__":
    main()
