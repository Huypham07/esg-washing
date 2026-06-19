import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from esgwash.eda import style


def test_palette_has_required_keys():
    for k in ["paper", "panel", "ink", "muted", "grid", "accent", "accent2", "highlight"]:
        assert k in style.PALETTE and style.PALETTE[k].startswith("#")


def test_entropy_uniform_two_states_is_one_bit():
    assert abs(style.shannon_entropy_bits([0.5, 0.5]) - 1.0) < 1e-9
    assert abs(style.effective_states([0.5, 0.5]) - 2.0) < 1e-9


def test_entropy_ignores_zero_prob():
    # zero-prob states must not produce NaN
    h = style.shannon_entropy_bits([0.0, 1.0])
    assert h == 0.0


def test_entropy_accepts_unnormalised_counts():
    # counts [1,1,1,1] -> 2 bits
    assert abs(style.shannon_entropy_bits([1, 1, 1, 1]) - 2.0) < 1e-9


def test_style_axes_returns_axes_and_sets_title():
    fig, ax = plt.subplots()
    out = style.style_axes(ax, title="Hello", subtitle="sub")
    assert out is ax
    assert ax.get_title(loc="left") == "Hello"
    plt.close(fig)


def test_delta_norm_centered_at_zero():
    assert style.DELTA_NORM.vcenter == 0.0
