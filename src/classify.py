"""ABC-XYZ classification: how valuable (ABC) and how predictable (XYZ) each product is.

ABC: sort products by revenue; A = products making up the first 80% of revenue,
     B = the next 15%, C = the last 5%.
XYZ: coefficient of variation (CV = standard deviation / mean) of weekly units;
     X < 0.5 stable, Y 0.5-1.0 variable, Z > 1.0 erratic.
Planners use this to decide where to spend effort: AX items are worth forecasting
carefully; CZ items may be better ordered on demand.
"""
import numpy as np
import pandas as pd

import config as C


def abc_xyz(w: pd.DataFrame) -> pd.DataFrame:
    g = w.groupby("stock_code")
    t = pd.DataFrame({
        "description": g["description"].first(),
        "revenue": g["revenue"].sum(),
        "mean_units": g["units"].mean(),
        "std_units": g["units"].std(),
        "active_share": g["units"].apply(lambda s: (s > 0).mean()),
    })
    t = t[t["revenue"] > 0].sort_values("revenue", ascending=False)
    cum = t["revenue"].cumsum() / t["revenue"].sum()
    t["abc"] = np.where(cum <= C.ABC_CUTS[0], "A", np.where(cum <= C.ABC_CUTS[1], "B", "C"))
    t.iloc[0, t.columns.get_loc("abc")] = "A"          # the top product is always A
    t["cv"] = t["std_units"] / t["mean_units"]
    t["xyz"] = np.where(t["cv"] < C.XYZ_CUTS[0], "X", np.where(t["cv"] <= C.XYZ_CUTS[1], "Y", "Z"))
    t["class"] = t["abc"] + t["xyz"]
    return t.reset_index()


def matrix(t: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(t["abc"], t["xyz"]).reindex(index=list("ABC"), columns=list("XYZ"), fill_value=0)
    rev = t.pivot_table(index="abc", columns="xyz", values="revenue", aggfunc="sum", fill_value=0)
    rev = rev.reindex(index=list("ABC"), columns=list("XYZ"), fill_value=0) / t["revenue"].sum()
    return counts, rev


def pick_items(t: pd.DataFrame) -> list:
    """Highest-revenue A items that sell in most weeks (forecastable)."""
    cand = t[(t["abc"] == "A") & (t["active_share"] >= C.MIN_ACTIVE_SHARE)]
    return cand.head(C.TOP_ITEMS)["stock_code"].tolist()
