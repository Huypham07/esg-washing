"""Convert model -> HF format va upload len Hugging Face Hub.

Model esgwash la PhoBERT multi-head sigmoid (kien truc tuy bien, KHONG phai
AutoModelForSequenceClassification) -> luu model.safetensors + tokenizer +
config.json (chua heads/thresholds). Repo tai ve can code esgwash de load.

  export HF_TOKEN=hf_xxx                       # hoac: huggingface-cli login
  python experiments/upload_to_hf.py outputs/models/topic/vi --repo <user>/esgwash-topic-vi
  python experiments/upload_to_hf.py outputs/models/commitment/vi --repo <user>/esgwash-commitment-vi --private

Cung dung de CHI convert model.pt cu -> safetensors (khong upload):
  python experiments/upload_to_hf.py outputs/models/commitment/vi --convert-only
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from esgwash.models.trainer import MultiHeadTrainer


def ensure_hf_format(model_dir: Path) -> dict:
    """Doc model (model.pt cu HOAC model.safetensors) roi ghi lai dang HF
    (safetensors + tokenizer). Xoa model.pt cu de khong upload trung 540MB."""
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    trainer = MultiHeadTrainer(cfg)
    trainer.load(model_dir)        # load duoc ca .pt lan .safetensors
    trainer.save(model_dir)        # ghi lai: safetensors + tokenizer + config.json
    legacy = model_dir / "model.pt"
    if legacy.exists() and (model_dir / "model.safetensors").exists():
        legacy.unlink()
    return cfg


def write_readme(model_dir: Path, repo_id: str, cfg: dict) -> None:
    """Sinh model card neu chua co (khong ghi de neu da chinh tay)."""
    p = model_dir / "README.md"
    if p.exists():
        return
    heads = cfg.get("heads")
    thresholds = cfg.get("thresholds")
    backbone = cfg.get("backbone")
    p.write_text(f"""---
language: vi
license: cc-by-nc-sa-4.0
library_name: pytorch
tags:
- esg
- phobert
- multi-label
- vietnamese
- text-classification
---

# {repo_id}

PhoBERT multi-head (sigmoid) cho ESG — backbone `{backbone}`, các đầu: `{heads}`.
Đây là kiến trúc **tùy biến** (một encoder + nhiều đầu nhị phân độc lập, masked BCE),
KHÔNG phải `AutoModelForSequenceClassification`, nên cần dùng code `esgwash` để tải.

- **Thresholds theo từng đầu** (tune trên tập val, tối đa hóa F1): `{thresholds}`
- File: `model.safetensors` (backbone + các đầu), tokenizer PhoBERT, `config.json`.

## Cách tải
```python
import json
from esgwash.models.trainer import MultiHeadTrainer
from huggingface_hub import snapshot_download

d = snapshot_download("{repo_id}")
cfg = json.load(open(f"{{d}}/config.json"))
model = MultiHeadTrainer(cfg).load(d)
probs = model.predict_proba(["câu tiếng Việt ..."])  # tự word-segment bên trong
```
""", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="vd outputs/models/topic/vi")
    ap.add_argument("--repo", default=None, help="<user>/<name> tren HF Hub")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--token", default=None, help="mac dinh lay tu env HF_TOKEN")
    ap.add_argument("--convert-only", action="store_true",
                    help="chi convert sang HF format, khong upload")
    args = ap.parse_args(argv)

    model_dir = Path(args.model_dir)
    cfg = ensure_hf_format(model_dir)
    print(f"da convert -> HF format tai {model_dir}: "
          f"{[p.name for p in sorted(model_dir.iterdir())]}")

    if args.convert_only:
        return
    if not args.repo:
        ap.error("can --repo khi upload (hoac dung --convert-only)")

    write_readme(model_dir, args.repo, cfg)
    from huggingface_hub import HfApi

    api = HfApi(token=args.token or os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(model_dir), repo_id=args.repo, repo_type="model")
    print(f"done -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
