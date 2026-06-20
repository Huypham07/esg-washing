"""Mã hoá tên ngân hàng cho paper + figure (anonymisation).

Mã = chữ cái đầu + chữ cái cuối của tên ngân hàng (3 ngân hàng "V..." phân biệt
thành VMK/VNK/VK). Mapping CỐ ĐỊNH dùng chung mọi figure/bảng để nhất quán.
Decoder (mã -> tên thật) chỉ giữ nội bộ; paper xuất bản chỉ hiện mã.
"""
from __future__ import annotations

import pandas as pd

# bank (lowercase key trong dữ liệu) -> mã ẩn danh (chữ đầu + chữ cuối của tên,
# 3 ngân hàng "V..." phân biệt bằng VMK/VNK/VK).
BANK_CODE = {
    "agribank": "AK",
    "bidv": "BV",
    "mbbank": "MK",
    "ocb": "OB",
    "shb": "SB",
    "techcombank": "TM",
    "vietcombank": "VMK",
    "viettinbank": "VNK",
    "vpbank": "VK",
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
