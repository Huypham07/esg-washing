"""Candidate evidence pool (spec 03 #1) - thu hep, khong lay ca corpus.

Ung vien: cung doc_id, va (co so lieu | block_type table/list | specific-fact).
Loai chinh claim va cau cung block.
"""
NUMERIC_PATTERN = None  # TODO: regex so + don vi (%, ty, trieu, tan CO2, MWh, ha, ty dong)


def build_pool(claim_row, sentences_df, config: dict):
    raise NotImplementedError  # TODO(Phase C1)
