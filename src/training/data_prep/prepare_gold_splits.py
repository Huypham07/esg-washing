"""Nhánh B — chuẩn hoá gold dịch về schema trainer (sentence + label 0/1).

- Topic env/soc/gov: bỏ Unnamed, NFC, dedup, split 80/10/10 stratify.
- Commitment/Specificity: đã align train/test -> bỏ leak, NFC, carve val GIỮ
  CÙNG index (giữ align), test giữ nguyên. Bỏ env_claims/netzero.
- Commitment train += action_500 (augment, action->commitment, giữ cả 500 để
  KHÔNG lệch balance ~0.42). Train-only; specificity bất biến.

Chạy:  python -m src.training.data_prep.prepare_gold_splits
"""
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

TRANSLATE = Path("data/en_gold/translate")
OUT = Path("data/vi_gold")
SEED = 42

TOPIC_SPECS = [
    ("environmental_2k.csv", "env"),
    ("social_2k.csv", "soc"),
    ("governance_2k.csv", "gov"),
]


def nfc(s) -> str:
    s = unicodedata.normalize("NFC", str(s))
    # sửa ký tự lỗi hay gặp khi dịch/OCR
    s = s.replace("−", "-").replace("�", "").replace(" ", " ")
    return s.strip()


def _std(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """-> schema trainer: sentence (NFC) + label (0/1). BỎ câu rỗng sau NFC.

    Dịch/NFC đôi khi tạo câu rỗng (vd 6 câu ở commit/spec test, có cả nhãn=1 cho
    câu rỗng = sai) -> lọc tại đây. commit & spec chia sẻ cùng text -> bỏ cùng dòng
    -> giữ alignment. Topic không có câu rỗng nên không đổi.
    """
    out = pd.DataFrame({
        "sentence": df["text"].map(nfc),
        "label": df[label_col].astype(int),
    })
    return out[out["sentence"].str.len() > 0].reset_index(drop=True)


def _save(df: pd.DataFrame, task: str, split: str) -> None:
    d = OUT / task
    d.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(d / f"{split}.parquet", index=False)


def prepare_topic() -> None:
    for fname, task in TOPIC_SPECS:
        df = pd.read_csv(TRANSLATE / fname)
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
        df = _std(df, task)
        # cảnh báo nếu 1 câu có >1 nhãn khác nhau (annotation conflict) — dedup giữ first
        n_conf = int((df.groupby("sentence")["label"].nunique() > 1).sum())
        if n_conf:
            print(f"[topic:{task}] WARN {n_conf} câu nhãn mâu thuẫn -> giữ first khi dedup")
        df = df.drop_duplicates(subset=["sentence"]).reset_index(drop=True)

        # 80/10/10 stratify theo nhãn nhị phân
        train, test = train_test_split(df, test_size=0.10, stratify=df["label"], random_state=SEED)
        train, val = train_test_split(train, test_size=1 / 9, stratify=train["label"], random_state=SEED)
        for split, part in [("train", train), ("val", val), ("test", test)]:
            _save(part, task, split)
        print(f"[topic:{task}] train {len(train)} / val {len(val)} / test {len(test)} "
              f"| pos-rate {df['label'].mean():.2f}")


def prepare_commit_spec() -> None:
    ca_tr = pd.read_parquet(TRANSLATE / "commitments_actions.train.parquet").reset_index(drop=True)
    ca_te = pd.read_parquet(TRANSLATE / "commitments_actions.test.parquet").reset_index(drop=True)
    sp_tr = pd.read_parquet(TRANSLATE / "specificity.train.parquet").reset_index(drop=True)
    sp_te = pd.read_parquet(TRANSLATE / "specificity.test.parquet").reset_index(drop=True)

    # Bất biến nguồn: commit & spec cùng câu theo vị trí (train + test)
    assert ca_tr["text"].equals(sp_tr["text"]), "commit/spec TRAIN không align"
    assert ca_te["text"].equals(sp_te["text"]), "commit/spec TEST không align"

    for d in (ca_tr, ca_te, sp_tr, sp_te):
        d["text"] = d["text"].map(nfc)

    # bỏ leak (câu train có trong test) + dedup train — cùng mask cho cả 2 (aligned)
    keep = ~ca_tr["text"].isin(set(ca_te["text"])) & ~ca_tr["text"].duplicated()
    ca_tr, sp_tr = ca_tr[keep].reset_index(drop=True), sp_tr[keep].reset_index(drop=True)

    # carve val: split index 1 lần, áp cho CẢ 2 -> giữ align; stratify trên (commit,spec)
    joint = ca_tr["label"].astype(str) + sp_tr["label"].astype(str)
    idx = list(range(len(ca_tr)))
    tr_idx, val_idx = train_test_split(idx, test_size=0.15, stratify=joint, random_state=SEED)

    out = {
        ("commitment", "train"): _std(ca_tr.iloc[tr_idx], "label"),
        ("commitment", "val"): _std(ca_tr.iloc[val_idx], "label"),
        ("commitment", "test"): _std(ca_te, "label"),
        ("specificity", "train"): _std(sp_tr.iloc[tr_idx], "label"),
        ("specificity", "val"): _std(sp_tr.iloc[val_idx], "label"),
        ("specificity", "test"): _std(sp_te, "label"),
    }
    # Bất biến sau split: commit & spec vẫn cùng câu ở mọi split
    for sp_name in ("train", "val", "test"):
        a = out[("commitment", sp_name)]["sentence"].reset_index(drop=True)
        b = out[("specificity", sp_name)]["sentence"].reset_index(drop=True)
        assert a.equals(b), f"misaligned after split: {sp_name}"

    # Augment commitment train bằng action_500 (CHỈ commitment, train-only).
    # action_500 KHÔNG có nhãn specificity -> sau bước này commit↔spec hết align;
    # CHẤP NHẬN vì train 5 model NHỊ PHÂN RỜI (không multi-task). Thêm CẢ 500
    # (action->commitment) để GIỮ balance (~0.42) — không lệch positive như khi chỉ thêm action=1.
    act = pd.read_csv(TRANSLATE / "action_500.csv")
    act = act.drop(columns=[c for c in act.columns if c.startswith("Unnamed")], errors="ignore")
    act_std = _std(act, "action")
    # leak-guard: bỏ câu trùng commitment test (an toàn dù khác nguồn dataset)
    act_std = act_std[~act_std["sentence"].isin(set(out[("commitment", "test")]["sentence"]))]
    n0 = len(out[("commitment", "train")])
    out[("commitment", "train")] = pd.concat(
        [out[("commitment", "train")], act_std], ignore_index=True)
    print(f"  +action_500: commitment train {n0} -> {len(out[('commitment','train')])} "
          f"(+{len(act_std)}, pos-rate {out[('commitment','train')]['label'].mean():.2f})")

    for (task, split), df in out.items():
        _save(df, task, split)

    leak = len(set(out[("commitment", "train")]["sentence"]) & set(out[("commitment", "test")]["sentence"]))
    # commit train đã augment (+action_500) -> KHÔNG còn align với spec train; báo cáo pos-rate riêng.
    print(f"[commit] train {len(out[('commitment','train')])} (pos {out[('commitment','train')]['label'].mean():.2f}) "
          f"/ val {len(out[('commitment','val')])} / test {len(out[('commitment','test')])}")
    print(f"[spec]   train {len(out[('specificity','train')])} (pos {out[('specificity','train')]['label'].mean():.2f}) "
          f"/ val {len(out[('specificity','val')])} / test {len(out[('specificity','test')])}")
    print(f"  commit train↔test leak: {leak} (kỳ vọng 0)")


def main() -> None:
    prepare_topic()
    prepare_commit_spec()
    print("Done -> data/vi_gold/")


if __name__ == "__main__":
    main()
