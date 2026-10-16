"""Step 4 - Turn forecast errors into safety stock and reorder points.

Safety stock covers forecast error during the lead time:
    safety_stock = z × σ(weekly forecast error) × √(lead time in weeks)
    reorder_point = expected demand during lead time + safety_stock
z comes from the service level (the chance of not running out during the lead time):
    90% → 1.28, 95% → 1.64, 99% → 2.33
A more accurate forecast has smaller errors, so it needs less safety stock for the same
service level: that saving is the business value of better forecasting.
"""
import numpy as np
import pandas as pd
from scipy import stats

import config as C


def safety_stock(sigma: float, service: float, lead_time: float = C.LEAD_TIME_WEEKS) -> float:
    return float(stats.norm.ppf(service) * sigma * np.sqrt(lead_time))


def plan(bt: dict, items: list, prices: pd.Series, chosen: pd.Series) -> pd.DataFrame:
    """Compare a naive plan with the plan using each item's method chosen on the validation quarter."""
    a = bt["actual"]
    rows = []
    for i, item in enumerate(items):
        errs = {m: a[i] - f[i] for m, f in bt["forecasts"].items()}
        for label, m in [("naive", "naive"), ("selected", chosen[item])]:
            sigma = float(np.std(errs[m], ddof=1))
            mean_fc = float(bt["forecasts"][m][i].mean())
            for s in C.SERVICE_LEVELS:
                ss = safety_stock(sigma, s)
                rows.append({"stock_code": item, "plan": label, "method": m, "service_level": s,
                             "sigma_units": sigma, "safety_stock_units": ss,
                             "reorder_point_units": mean_fc * C.LEAD_TIME_WEEKS + ss,
                             "safety_stock_value": ss * float(prices[item])})
    return pd.DataFrame(rows)


def savings(p: pd.DataFrame) -> pd.DataFrame:
    t = p.pivot_table(index="service_level", columns="plan", values="safety_stock_value", aggfunc="sum")
    t["saving_value"] = t["naive"] - t["selected"]
    t["saving_pct"] = t["saving_value"] / t["naive"]
    return t.reset_index()
