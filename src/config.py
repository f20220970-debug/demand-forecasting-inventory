"""Settings for demand forecasting and inventory planning."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
CHARTS = OUTPUTS / "charts"
REPORTS = ROOT / "reports"

RETAIL_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RETAIL_XLSX = RAW / "online_retail_II.xlsx"
TRANSACTIONS = PROCESSED / "transactions.parquet"   # same cleaned file as the analytics project
WEEKLY = PROCESSED / "weekly_demand.parquet"

TOP_ITEMS = 20            # forecast the 20 highest-revenue products that sell almost every week
MIN_ACTIVE_SHARE = 0.80   # an item must sell in at least 80% of weeks to be forecastable
TEST_WEEKS = 13           # last quarter held out (includes the pre-Christmas peak)
SEASON = 52               # weeks in a year, for seasonal models

LEAD_TIME_WEEKS = 2       # time from placing an order to receiving stock
SERVICE_LEVELS = [0.90, 0.95, 0.99]   # chance of not running out during the lead time
ABC_CUTS = (0.80, 0.95)   # A = items making the first 80% of revenue, B = next 15%, C = rest
XYZ_CUTS = (0.5, 1.0)     # coefficient of variation of weekly demand: X stable, Y variable, Z erratic
SEED = 42
