"""Dang ky stage end-to-end (spec 00 #4). Moi stage: ham thuan,
doc artifact truoc -> ghi artifact sau; chay doc lap duoc.

  build_corpus -> classify_topic -> classify_claim -> ground -> index
"""

STAGES = ["build_corpus", "classify_topic", "classify_claim", "ground", "index"]


def run_stage(name: str, config: dict) -> None:
    raise NotImplementedError  # TODO(Phase A1 wiring; tung stage theo phase)
