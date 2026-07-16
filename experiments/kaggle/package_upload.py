"""Dong goi 1 zip upload len Kaggle: code (esgwash + parser) + 4 input file.
Chay: python experiments/kaggle/package_upload.py  (sau build_kaggle_inputs.py)
Xuat: experiments/kaggle/upload/esgwash-ablation.zip  -> tao Kaggle Dataset tu file nay.
Cau truc: src/esgwash/** · experiments/ablation_direct_parser.py · 4 input o root
-> tren Kaggle: REPO_DIR = INPUT_DIR = /kaggle/input/<slug>.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KG = ROOT / "experiments/kaggle"
OUT = KG / "upload"
OUT.mkdir(parents=True, exist_ok=True)
ZPATH = OUT / "esgwash-ablation.zip"
INPUTS = ["c2c3_input.parquet", "inv_vague_seeds.json", "ablation_c3_fewshot.json", "committed_chunk_ids.json"]


def main() -> None:
    for f in INPUTS:
        assert (KG / f).exists(), f"THIEU input {f} — chay build_kaggle_inputs.py truoc."
    n_code = 0
    with zipfile.ZipFile(ZPATH, "w", zipfile.ZIP_DEFLATED) as z:
        for p in (ROOT / "src/esgwash").rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                z.write(p, p.relative_to(ROOT))   # arcname: src/esgwash/...
                n_code += 1
        z.write(ROOT / "experiments/ablation_direct_parser.py", "experiments/ablation_direct_parser.py")
        n_code += 1
        for f in INPUTS:
            z.write(KG / f, f)                    # input o root zip
    size_mb = ZPATH.stat().st_size / 1e6
    print(f"WROTE {ZPATH} ({size_mb:.2f} MB): {n_code} code files + {len(INPUTS)} inputs")
    with zipfile.ZipFile(ZPATH) as z:
        names = z.namelist()
    must = ["src/esgwash/models/specificity_llm.py", "src/esgwash/models/trainer.py",
            "src/esgwash/nlp/segmentation.py", "src/esgwash/validation/digit_shortcut.py",
            "experiments/ablation_direct_parser.py", *INPUTS]
    for m in must:
        assert m in names, f"THIEU {m} trong zip!"
    print("verify: tat ca file bat buoc co trong zip. OK.")


if __name__ == "__main__":
    main()
