import numpy as np
import pandas as pd

import classify
import forecast as F
import inventory as I


def test_wape_and_bias():
    assert F.wape([10, 10], [8, 12]) == 0.2
    assert F.bias([10, 10], [12, 12]) == 0.2


def test_safety_stock_formula():
    ss = I.safety_stock(sigma=10, service=0.95, lead_time=4)
    assert abs(ss - 1.6449 * 10 * 2) < 0.01


def test_seasonal_naive_uses_last_year_and_no_leakage():
    weeks = pd.date_range("2020-01-06", periods=110, freq="W-MON")
    rng = np.random.default_rng(0)
    Y = np.tile(np.arange(52) + 10.0, (3, 3))[:, :110] + rng.normal(0, 0.1, (3, 110))
    bt = F.backtest(Y, weeks)
    t0 = 110 - 13
    assert np.allclose(bt["forecasts"]["seasonal_naive"][:, 0], Y[:, t0 - 52])
    assert np.allclose(bt["forecasts"]["naive"][:, 0], Y[:, t0 - 1])
    # a perfectly repeating season should be forecast almost perfectly by seasonal naive
    assert F.wape(bt["actual"], bt["forecasts"]["seasonal_naive"]) < 0.02


def test_abc_xyz_classes():
    w = pd.DataFrame({"stock_code": ["a"] * 4 + ["b"] * 4 + ["c"] * 4, "description": "x",
                      "units": [10, 10, 10, 10, 1, 9, 0, 6, 0, 0, 0, 1],
                      "revenue": [100, 100, 100, 100, 10, 10, 10, 10, 1, 1, 1, 1]})
    t = classify.abc_xyz(w).set_index("stock_code")
    assert t.loc["a", "abc"] == "A" and t.loc["a", "xyz"] == "X"
    assert t.loc["c", "xyz"] == "Z"
