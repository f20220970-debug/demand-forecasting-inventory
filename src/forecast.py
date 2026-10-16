"""Step 3 - Forecast weekly demand and compare methods fairly.

Back-test: the last TEST_WEEKS weeks are held out. Each test week is forecast one week
ahead using only data up to the week before (a rolling origin), the way a planner
would forecast every Monday.

Methods:
  naive            - next week = this week
  seasonal_naive   - next week = same week last year
  moving_avg_4     - average of the last 4 weeks
  exp_smoothing    - simple exponential smoothing, smoothing weight chosen on training weeks
  gradient_boost   - one machine-learning model across all items (lags 1-4 and 52, rolling
                     means, week of year), each item scaled by its training average
Accuracy:
  WAPE = sum |actual - forecast| / sum actual   (lower is better; robust to small weeks)
  bias = sum (forecast - actual) / sum actual   (positive = over-forecasting)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import config as C

METHODS = ["naive", "seasonal_naive", "moving_avg_4", "exp_smoothing", "gradient_boost"]


def wape(actual, fc) -> float:
    actual, fc = np.asarray(actual, float), np.asarray(fc, float)
    return float(np.abs(actual - fc).sum() / actual.sum()) if actual.sum() else np.nan


def bias(actual, fc) -> float:
    actual, fc = np.asarray(actual, float), np.asarray(fc, float)
    return float((fc - actual).sum() / actual.sum()) if actual.sum() else np.nan


def ses_alpha(y: np.ndarray, grid=np.linspace(0.05, 0.95, 19)) -> float:
    """Pick the smoothing weight with the lowest one-step error on the training weeks."""
    best, best_err = 0.3, np.inf
    for a in grid:
        level, err = y[0], 0.0
        for v in y[1:]:
            err += abs(v - level)
            level = a * v + (1 - a) * level
        if err < best_err:
            best, best_err = a, err
    return float(best)


def features(Y: np.ndarray, t: int, scale: np.ndarray, weeks: pd.DatetimeIndex) -> np.ndarray:
    """Features for week t, using only weeks before t."""
    wk = weeks[t].isocalendar().week
    f = np.column_stack([
        Y[:, t - 1], Y[:, t - 2], Y[:, t - 3], Y[:, t - 4], Y[:, t - C.SEASON],
        Y[:, t - 4:t].mean(axis=1), Y[:, t - 8:t].mean(axis=1),
    ]) / scale[:, None]
    season = np.column_stack([np.full(len(Y), np.sin(2 * np.pi * wk / 52)), np.full(len(Y), np.cos(2 * np.pi * wk / 52))])
    return np.hstack([f, season])


def backtest(Y: np.ndarray, weeks: pd.DatetimeIndex, end: int = None) -> dict:
    """Forecast the TEST_WEEKS weeks ending at `end` (default: the last week), using only earlier data."""
    n_items = Y.shape[0]
    n_weeks = end if end is not None else Y.shape[1]
    test_start = n_weeks - C.TEST_WEEKS
    if test_start - 8 < C.SEASON:
        raise SystemExit("Not enough history for seasonal methods")
    train = Y[:, :test_start]
    scale = np.maximum(train.mean(axis=1), 1e-9)
    # train the ML model on all training weeks that have a full year of history
    X_tr, y_tr = [], []
    for t in range(C.SEASON, test_start):
        X_tr.append(features(Y, t, scale, weeks)); y_tr.append(Y[:, t] / scale)
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          random_state=C.SEED).fit(np.vstack(X_tr), np.concatenate(y_tr))
    alphas = np.array([ses_alpha(train[i]) for i in range(n_items)])
    fc = {m: np.zeros((n_items, C.TEST_WEEKS)) for m in METHODS}
    for j, t in enumerate(range(test_start, n_weeks)):
        fc["naive"][:, j] = Y[:, t - 1]
        fc["seasonal_naive"][:, j] = Y[:, t - C.SEASON]
        fc["moving_avg_4"][:, j] = Y[:, t - 4:t].mean(axis=1)
        for i in range(n_items):
            level = Y[i, 0]
            for v in Y[i, 1:t]:
                level = alphas[i] * v + (1 - alphas[i]) * level
            fc["exp_smoothing"][i, j] = level
        fc["gradient_boost"][:, j] = np.clip(model.predict(features(Y, t, scale, weeks)), 0, None) * scale
    return {"actual": Y[:, test_start:n_weeks], "forecasts": fc, "test_weeks": weeks[test_start:n_weeks], "alphas": alphas}


def select_methods(Y: np.ndarray, weeks: pd.DatetimeIndex, items: list) -> pd.Series:
    """Choose each item's method on a validation quarter BEFORE the test quarter, so the
    test result isn't flattered by picking the winner after seeing it."""
    val = backtest(Y, weeks, end=Y.shape[1] - C.TEST_WEEKS)
    w = pd.DataFrame({m: [wape(val["actual"][i], f[i]) for i in range(len(items))] for m, f in val["forecasts"].items()},
                     index=items)
    return w.idxmin(axis=1)


def scores(bt: dict, items: list) -> tuple:
    a = bt["actual"]
    overall = pd.DataFrame([{"method": m, "wape": wape(a, f), "bias": bias(a, f),
                             "mae_units": float(np.abs(a - f).mean())} for m, f in bt["forecasts"].items()])
    overall = overall.sort_values("wape").reset_index(drop=True)
    per_item = pd.DataFrame({m: [wape(a[i], f[i]) for i in range(len(items))] for m, f in bt["forecasts"].items()},
                            index=items)
    per_item["best_method"] = per_item[METHODS].idxmin(axis=1)
    return overall, per_item
