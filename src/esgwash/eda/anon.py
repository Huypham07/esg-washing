"""Mã hoá tên ngân hàng cho paper + figure (anonymisation).

Mã = B_A .. B_I, gán theo thứ tự CTI tăng dần (B_A = thấp nhất) để bảng/hình
đọc xuôi. Mapping CỐ ĐỊNH dùng chung mọi figure/bảng để nhất quán.
Decoder (mã -> tên thật) chỉ giữ nội bộ; paper xuất bản chỉ hiện mã.
"""
from __future__ import annotations

import pandas as pd

# bank (lowercase key trong dữ liệu) -> mã ẩn danh B_A..B_I (thứ tự CTI tăng dần).
BANK_CODE = {
    "agribank": "B_A",
    "ocb": "B_B",
    "bidv": "B_C",
    "mbbank": "B_D",
    "vpbank": "B_E",
    "vietcombank": "B_F",
    "shb": "B_G",
    "techcombank": "B_H",
    "viettinbank": "B_I",
}


def code_of(bank: str) -> str:
    return BANK_CODE.get(str(bank).strip().lower(), str(bank))


def anonymize(df: pd.DataFrame, col: str = "bank") -> pd.DataFrame:
    """Trả bản copy với cột `col` thay bằng mã ẩn danh."""
    if col not in df.columns:
        return df
    out = df.copy()
    out[col] = out[col].map(code_of)
    return out
