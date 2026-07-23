"""Parse raw LBMA + Yahoo JSON into tidy daily CSVs. Run once."""
import json
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent / "data"

# LBMA fixes: list of {d: date, v: [USD, GBP, EUR]}
for src, out in [("gold_pm.json", "xau.csv"), ("silver.json", "xag.csv")]:
    rows = json.load(open(DATA / src))
    df = pd.DataFrame({"date": [r["d"] for r in rows],
                       "close": [r["v"][0] for r in rows]})
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().query("close > 0").set_index("date").sort_index()
    df.to_csv(DATA / out)
    print(out, df.index[0].date(), df.index[-1].date(), len(df))

# Yahoo chart JSON: adjclose
for sym in ["GLD", "SLV", "UGL"]:
    r = json.load(open(DATA / f"{sym}.json"))["chart"]["result"][0]
    ts = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
    adj = r["indicators"]["adjclose"][0]["adjclose"]
    close = r["indicators"]["quote"][0]["close"]
    df = pd.DataFrame({"date": ts, "adjclose": adj, "close": close}).dropna()
    df = df.drop_duplicates("date").set_index("date").sort_index()
    df.to_csv(DATA / f"{sym.lower()}.csv")
    print(sym, df.index[0].date(), df.index[-1].date(), len(df))
