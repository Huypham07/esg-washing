import json
import pandas as pd
from esgwash.indices import figures_extract as fx


def test_parse_figures_keeps_quantified_with_figure():
    rub = json.dumps({"items": [
        {"action_or_event": "dư nợ tín dụng xanh", "figure": "74.000 tỷ",
         "is_quantified": True, "attributable_to_actor": True, "is_concrete_action": True},
        {"action_or_event": "nâng cao năng lực", "figure": None,
         "is_quantified": False, "attributable_to_actor": True, "is_concrete_action": False},
    ]})
    out = fx.parse_figures(rub)
    assert len(out) == 1
    assert out[0]["action"] == "dư nợ tín dụng xanh" and out[0]["figure"] == "74.000 tỷ"


def test_parse_figures_bad_input():
    assert fx.parse_figures(None) == []
    assert fx.parse_figures("not json") == []
    assert fx.parse_figures(json.dumps({"items": []})) == []


def test_categorize_action_priority():
    assert fx.categorize_action("dư nợ tín dụng xanh cho vay") == "green_credit"
    assert fx.categorize_action("giảm phát thải khí nhà kính") == "emissions"
    assert fx.categorize_action("lắp điện mặt trời") == "energy"
    assert fx.categorize_action("trồng 330.000 cây xanh") == "trees"
    assert fx.categorize_action("đào tạo cán bộ nhân viên") == "training"
    assert fx.categorize_action("ủng hộ quỹ an sinh xã hội") == "social"
    assert fx.categorize_action("một việc gì đó") == "other"


def test_figure_table_counts_by_type():
    rub = json.dumps({"items": [
        {"action_or_event": "dư nợ tín dụng xanh", "figure": "5 tỷ",
         "is_quantified": True, "attributable_to_actor": True, "is_concrete_action": True}]})
    clf = pd.DataFrame({
        "bank": ["a"], "year": [2023], "is_env": [1], "is_soc": [0], "is_gov": [0],
        "is_commitment": [1], "spec_level": [2], "spec_rubric": [rub]})
    out = fx.figure_table(clf)
    row = out.iloc[0]
    assert row["type"] == "green_credit" and row["n"] == 1


def test_pillar_say_do():
    # env: 2 vague + 1 quantified -> cti_p=2/3, qdr_p=1/3, say_do=1/3
    clf = pd.DataFrame({
        "bank": ["a", "a", "a"], "year": [2023, 2023, 2023],
        "is_env": [1, 1, 1], "is_soc": [0, 0, 0], "is_gov": [0, 0, 0],
        "is_commitment": [1, 1, 1], "spec_level": [0, 0, 2], "spec_rubric": [None, None, None]})
    out = fx.pillar_say_do(clf)
    env = out[out["pillar"] == "env"].iloc[0]
    assert env["n"] == 3
    assert abs(env["cti_p"] - 2/3) < 1e-9 and abs(env["qdr_p"] - 1/3) < 1e-9
    assert abs(env["say_do"] - 1/3) < 1e-9
