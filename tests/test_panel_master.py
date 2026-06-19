# tests/test_panel_master.py
import numpy as np
import pandas as pd
from esgwash.indices import panel_master as pm


def _panel():
    return pd.DataFrame({"bank": ["a", "b"], "year": [2023, 2023],
                         "cti": [0.5, 0.2], "nar": [0.3, 0.4], "qdr": [0.2, 0.4],
                         "n_commit": [10, 20]})


def _bri():
    return pd.DataFrame({"bank": ["a", "b"], "year": [2023, 2023], "bri": [0.6, 0.5]})


def _say_do():
    rows = []
    for bank, sd in [("a", {"env": 0.1, "soc": -0.1, "gov": 0.3}),
                     ("b", {"env": 0.0, "soc": 0.2, "gov": 0.1})]:
        for p, v in sd.items():
            rows.append({"bank": bank, "year": 2023, "pillar": p,
                         "cti_p": 0.0, "qdr_p": 0.0, "say_do": v, "n": 5})
    return pd.DataFrame(rows)


def test_merge_signals_columns_and_values():
    m = pm.merge_signals(_panel(), _bri(), _say_do())
    assert set(["bank", "year", "cti", "nar", "qdr", "n_commit", "bri",
                "say_do_env", "say_do_soc", "say_do_gov"]).issubset(m.columns)
    a = m[m["bank"] == "a"].iloc[0]
    assert a["bri"] == 0.6 and abs(a["say_do_gov"] - 0.3) < 1e-9


def test_rq4_correlations_shape():
    m = pm.merge_signals(_panel(), _bri(), _say_do())
    out = pm.rq4_correlations(m)
    assert "cti_bri" in out and "nar_bri" in out
    assert out["cti_bri"]["n"] == 2 and "rho" in out["cti_bri"]


def test_say_do_by_pillar_means():
    out = pm.say_do_by_pillar(_say_do())
    assert abs(out["gov"] - 0.2) < 1e-9 and abs(out["env"] - 0.05) < 1e-9


def test_build_master_smoke(tmp_path):
    import importlib.util, sys
    from pathlib import Path
    d = tmp_path
    _panel().to_csv(d / "panel.csv", index=False)
    _bri().to_csv(d / "embedding_signals.csv", index=False)
    _say_do().to_csv(d / "say_do.csv", index=False)
    spec = importlib.util.spec_from_file_location(
        "build_findings", Path(__file__).resolve().parents[1] / "experiments" / "build_findings.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["build_findings"] = mod
    spec.loader.exec_module(mod)
    m = mod.build_master(panel_dir=str(d))
    assert "bri" in m.columns and "say_do_gov" in m.columns and len(m) == 2
