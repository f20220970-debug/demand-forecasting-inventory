"""Run everything: weekly demand -> ABC-XYZ -> forecast back-test -> inventory plan -> report."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import charts  # noqa: E402
import classify  # noqa: E402
import config as C  # noqa: E402
import forecast as F  # noqa: E402
import inventory as I  # noqa: E402
import prepare  # noqa: E402

MARK = "<!-- COMMENTARY -->"


def md(df):
    h = "| " + " | ".join(map(str, df.columns)) + " |"
    return "\n".join([h, "| " + " | ".join("---" for _ in df.columns) + " |"]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])


def kept(path):
    d = "_Write 4-6 sentences in your own words: which products matter, how predictable they are, which method won and why, and what the inventory saving means._"
    if path.exists():
        p = path.read_text(encoding="utf-8").split(MARK)
        if len(p) == 3:
            return p[1].strip()
    return d


def main():
    prepare.main()
    w = pd.read_parquet(C.WEEKLY)
    cls = classify.abc_xyz(w)
    counts, rev = classify.matrix(cls)
    items = classify.pick_items(cls)
    Y = (w[w["stock_code"].isin(items)].pivot(index="stock_code", columns="week", values="units")
           .reindex(items).to_numpy(float))
    weeks = pd.DatetimeIndex(sorted(w["week"].unique()))
    prices = w.groupby("stock_code")["unit_price"].first()

    chosen = F.select_methods(Y, weeks, items)
    bt = F.backtest(Y, weeks)
    overall, per_item = F.scores(bt, items)
    chosen_fc = np.vstack([bt["forecasts"][chosen[it]][i] for i, it in enumerate(items)])
    sel_row = {"method": "selected per item (validation)", "wape": F.wape(bt["actual"], chosen_fc),
               "bias": F.bias(bt["actual"], chosen_fc), "mae_units": float(np.abs(bt["actual"] - chosen_fc).mean())}
    overall = pd.concat([overall, pd.DataFrame([sel_row])], ignore_index=True).sort_values("wape").reset_index(drop=True)
    pl = I.plan(bt, items, prices, chosen)
    sv = I.savings(pl)

    C.OUTPUTS.mkdir(exist_ok=True)
    cls.round(4).to_csv(C.OUTPUTS / "abc_xyz.csv", index=False)
    overall.round(4).to_csv(C.OUTPUTS / "method_accuracy.csv", index=False)
    per_item.round(4).assign(selected=chosen).to_csv(C.OUTPUTS / "item_accuracy.csv")
    pl.round(2).to_csv(C.OUTPUTS / "inventory_plan.csv", index=False)
    sv.round(4).to_csv(C.OUTPUTS / "safety_stock_savings.csv", index=False)
    best_single = overall[overall["method"].isin(F.METHODS)].iloc[0]["method"]
    charts.total_forecast(Y, weeks, bt, best_single); charts.method_wape(overall[overall["method"].isin(F.METHODS)])
    charts.abc_xyz_heatmap(counts, rev); charts.safety_curve(sv)

    nv = overall.set_index("method").loc["naive", "wape"]
    sel = overall.set_index("method").loc["selected per item (validation)"]
    bs = overall.set_index("method").loc[best_single]
    s95 = sv.set_index("service_level").loc[0.95]
    a_share = cls.loc[cls["abc"] == "A", "revenue"].sum() / cls["revenue"].sum()
    facts = [
        f"{w['stock_code'].nunique():,} products over {len(weeks)} complete weeks ({weeks[0]:%Y-%m-%d} to {weeks[-1]:%Y-%m-%d}).",
        f"ABC: {int((cls['abc'] == 'A').sum()):,} A-products ({100 * (cls['abc'] == 'A').mean():.1f}% of products) make {100 * a_share:.0f}% of revenue; only {int(counts.loc['A', 'X'])} A-products are stable (AX), {int(counts.loc['A', 'Z'])} are erratic (AZ).",
        f"Forecast hold-out: last {C.TEST_WEEKS} weeks for the top {len(items)} products, one week ahead.",
        f"Best single method: {best_single} with WAPE {100 * bs['wape']:.1f}% vs naive {100 * nv:.1f}%; per-item methods chosen on a validation quarter give {100 * sel['wape']:.1f}% (bias {100 * sel['bias']:+.1f}%).",
        f"At a 95% service level and {C.LEAD_TIME_WEEKS}-week lead time, safety stock falls from £{s95['naive']:,.0f} (naive) to £{s95['selected']:,.0f} with the selected methods, a {100 * s95['saving_pct']:.0f}% saving.",
        f"Raising service from 95% to 99% increases safety stock by {100 * (I.safety_stock(1, 0.99) / I.safety_stock(1, 0.95) - 1):.0f}% for every plan, because z rises from 1.64 to 2.33.",
    ]
    out = C.REPORTS / "forecasting_report.md"
    text = f"""# Demand Forecasting & Inventory Planning Report

## 1. Key findings
""" + "\n".join(f"- {f}" for f in facts) + f"""

### Analyst commentary
{MARK}
{kept(out)}
{MARK}

## 2. ABC-XYZ classification
Items by class (rows: value, columns: predictability):

{md(counts.reset_index())}

![ABC-XYZ](../outputs/charts/abc_xyz.png)

## 3. Forecast accuracy (hold-out quarter)
{md(overall.round(4))}

![Method WAPE](../outputs/charts/method_wape.png)
![Forecast vs actual](../outputs/charts/forecast_vs_actual.png)

Methods chosen per item on the validation quarter: {chosen.value_counts().to_dict()}.

## 4. Inventory plan
{md(sv.round(4))}

![Safety stock](../outputs/charts/safety_stock.png)

## 5. Method notes
- Weekly units per product (Monday-Sunday), partial first/last weeks removed, weeks with no sales filled with 0.
- One-week-ahead rolling back-test on the last {C.TEST_WEEKS} weeks; each item's method is chosen on the previous {C.TEST_WEEKS} weeks (validation), never on the test weeks.
- Safety stock = z × σ(weekly forecast error) × √lead time; reorder point = forecast demand over the lead time + safety stock; valued at each product's average selling price.
- Limits: wholesale demand is lumpy; two years of history give only one prior season for seasonal methods; stock is valued at selling price, not cost.
"""
    out.write_text(text, encoding="utf-8")
    print("Analysis")
    for f in facts:
        print("  -", f)


if __name__ == "__main__":
    main()
