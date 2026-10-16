"""Step 1 - Clean transactions and build weekly demand per product.

Uses the same cleaning rules as the product analytics project (cancellations, missing
customers, non-positive quantity/price and duplicates removed). If the cleaned
transactions parquet is copied in, the raw download is skipped.
Weeks run Monday-Sunday; the first and last partial weeks are dropped so every
week in the series is complete.
"""
import io
import zipfile

import pandas as pd
import requests

import config as C


def download_and_clean() -> pd.DataFrame:
    if not C.RETAIL_XLSX.exists():
        print("  downloading Online Retail II from UCI ...")
        r = requests.get(C.RETAIL_URL, timeout=300)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            name = [n for n in z.namelist() if n.lower().endswith(".xlsx")][0]
            C.RETAIL_XLSX.write_bytes(z.read(name))
    raw = pd.concat(pd.read_excel(C.RETAIL_XLSX, sheet_name=None, dtype={"Invoice": str, "StockCode": str}).values())
    df = raw.rename(columns={"Customer ID": "customer_id", "InvoiceDate": "invoice_date", "Invoice": "invoice",
                             "StockCode": "stock_code", "Description": "description", "Quantity": "quantity",
                             "Price": "price", "Country": "country"})
    bad = (df["invoice"].astype(str).str.upper().str.startswith("C") | df["customer_id"].isna()
           | (df["quantity"] <= 0) | (df["price"] <= 0) | df.duplicated())
    df = df[~bad].copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["revenue"] = df["quantity"] * df["price"]
    return df


def weekly_demand(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    tx["stock_code"] = tx["stock_code"].astype(str).str.upper().str.strip()
    tx["week"] = tx["invoice_date"].dt.to_period("W-SUN").dt.start_time
    first_full = tx["invoice_date"].min().normalize()
    first_full = first_full + pd.Timedelta(days=(7 - first_full.weekday()) % 7)      # next Monday
    last_full = tx["week"].max() - pd.Timedelta(weeks=1)                              # drop partial last week
    tx = tx[(tx["week"] >= first_full) & (tx["week"] <= last_full)]
    w = (tx.groupby(["stock_code", "week"]).agg(units=("quantity", "sum"), revenue=("revenue", "sum"))
           .reset_index())
    weeks = pd.date_range(first_full, last_full, freq="W-MON")
    items = w["stock_code"].unique()
    full = pd.MultiIndex.from_product([items, weeks], names=["stock_code", "week"])
    w = w.set_index(["stock_code", "week"]).reindex(full, fill_value=0).reset_index()   # weeks with no sales = 0
    desc = tx.groupby("stock_code")["description"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else "")
    price = tx.groupby("stock_code").apply(lambda g: g["revenue"].sum() / g["quantity"].sum())
    w["description"] = w["stock_code"].map(desc)
    w["unit_price"] = w["stock_code"].map(price)
    return w


def main() -> None:
    for d in (C.RAW, C.PROCESSED, C.OUTPUTS):
        d.mkdir(parents=True, exist_ok=True)
    print("Prepare")
    if C.WEEKLY.exists() and not C.TRANSACTIONS.exists() and not C.RETAIL_XLSX.exists():
        print("  using committed weekly_demand.parquet")
        return
    tx = pd.read_parquet(C.TRANSACTIONS) if C.TRANSACTIONS.exists() else download_and_clean()
    w = weekly_demand(tx)
    w.to_parquet(C.WEEKLY, index=False)
    print(f"  {w['stock_code'].nunique():,} products × {w['week'].nunique()} complete weeks "
          f"({w['week'].min():%Y-%m-%d} to {w['week'].max():%Y-%m-%d})")


if __name__ == "__main__":
    main()
