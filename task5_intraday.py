"""Task 5: intraday high vs close for the 200DMA reclaim trigger.

The LBMA series is close-only, so this uses GLD OHLC (2004-11 -> present) as the
only intraday-capable gold proxy available. MA is computed on GLD closes.
Execution is held identical across variants (next-day close) so the ONLY thing
that varies is the signal definition.
"""
import urllib.request as u, json, datetime as dt
import pandas as pd
import numpy as np
from common import metrics, run_filtered, whipsaws, OUT, DATA

# ---------- fetch + cache OHLC ----------
def yahoo_ohlc(sym, path):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=25y&interval=1d"
    req = u.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    r = json.load(u.urlopen(req, timeout=40))["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": [dt.date.fromtimestamp(t) for t in r["timestamp"]],
        "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
    }).dropna()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.to_csv(path)
    return df

df = yahoo_ohlc("GLD", DATA / "gld_ohlc.csv")
px, hi, lo = df["close"], df["high"], df["low"]
ma = px.rolling(200).mean()
print(f"GLD OHLC {df.index[0].date()} -> {df.index[-1].date()}  n={len(df)}")

# ================================================================
# 1. DIAGNOSTIC: how often does an intraday cross survive to the close?
# ================================================================
print("\n=== 1. 盘中触及 200DMA 后，当日收盘能否守住？===")
prev_below = (px.shift(1) < ma.shift(1))
touch = prev_below & (hi > ma)          # was below, traded above intraday
held = touch & (px > ma)                # ...and closed above
t, h = int(touch.sum()), int(held.sum())
print(f"从下方盘中触及 200DMA 的交易日: {t} 次")
print(f"其中当日收盘守住的: {h} 次 ({h/t*100:.1f}%)  →  失守 {t-h} 次 ({(t-h)/t*100:.1f}%)")

# and of those that closed above, how many were still above 5 / 20 days later?
idx = px.index
for k in (5, 20, 60):
    surv_t = [px.iloc[min(idx.get_loc(d)+k, len(px)-1)] > ma.iloc[min(idx.get_loc(d)+k, len(px)-1)]
              for d in idx[touch]]
    surv_h = [px.iloc[min(idx.get_loc(d)+k, len(px)-1)] > ma.iloc[min(idx.get_loc(d)+k, len(px)-1)]
              for d in idx[held]]
    print(f"  {k:2d}日后仍在MA上方 — 盘中触及口径 {np.mean(surv_t)*100:4.1f}%  |  收盘站上口径 {np.mean(surv_h)*100:4.1f}%")

# how far above the MA does the intraday high typically poke?
poke = (hi[touch] / ma[touch] - 1)
print(f"\n盘中最高价高出MA的幅度: 中位 {poke.median()*100:.2f}%  75分位 {poke.quantile(.75)*100:.2f}%")
gap = (px[touch] / ma[touch] - 1)
print(f"同日收盘价相对MA:        中位 {gap.median()*100:+.2f}%  （负=盘中冲高后收回线下）")

# ================================================================
# 2. BACKTEST: same execution, different signal definition
# ================================================================
print("\n\n=== 2. 回测（GLD 2005-，5bp/边，次日收盘执行，仅信号定义不同）===")
V = {
    "盘中高点>MA":        (hi > ma).astype(float),
    "收盘>MA":            (px > ma).astype(float),
    "收盘>MA×1.01":       (px > ma * 1.01).astype(float),
    "收盘>MA×1.02":       (px > ma * 1.02).astype(float),
    "盘中低点>MA(全日在上)": (lo > ma).astype(float),
}
wk = (px > ma).astype(float).resample("W-FRI").last().dropna()
V["现规则 周五收盘>MA"] = wk.reindex(px.index).ffill().fillna(0.0)

rows = []
for nm, sig in V.items():
    eq, pos, n = run_filtered(px, sig.reindex(px.index).fillna(0.0))
    m = metrics(eq, pos, nm); m["交易次数"] = n
    _, nws, wsl = whipsaws(px, pos)
    m["whipsaw次数"] = nws; m["whipsaw损耗"] = wsl
    rows.append(m)
bt = pd.DataFrame(rows)
d = bt.copy()
for c in ("CAGR", "年化波动", "最大回撤", "在场比例", "whipsaw损耗"):
    d[c] = (d[c] * 100).round(1).astype(str) + "%"
d["Calmar"] = d["Calmar"].round(2)
print(d.to_string(index=False))
bt.to_csv(OUT / "t5_intraday_backtests.csv", index=False)

# buy-and-hold reference
eqA = px / px.iloc[0]
mA = metrics(eqA, None, "买入持有")
print(f"\n(参考) 买入持有: CAGR {mA['CAGR']*100:.1f}%  最大回撤 {mA['最大回撤']*100:.1f}%  Calmar {mA['Calmar']:.2f}")

# ================================================================
# 3. Fresh-entry event study by signal definition
# ================================================================
print("\n\n=== 3. 入场事件研究（信号 0->1，间隔≥60天）===")
def fresh(sig, min_gap=60):
    s = sig.fillna(0.0); f = s.diff().fillna(0)
    out, last = [], None
    for dte in s.index[f > 0]:
        if last is None or (dte - last).days >= min_gap:
            out.append(dte)
        last = dte
    return out

HOR = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}
res = []
for nm, sig in V.items():
    ds = fresh(sig)
    recs = []
    for dte in ds:
        i = idx.get_loc(dte); p0 = px.iloc[i]
        r = {}
        for hname, k in HOR.items():
            r[hname] = px.iloc[i+k]/p0 - 1 if i+k < len(px) else np.nan
        r["MAE6M"] = px.iloc[i:i+126].min()/p0 - 1
        r["hold3M"] = (px.iloc[i:i+63] > ma.iloc[i:i+63]).mean()
        recs.append(r)
    e = pd.DataFrame(recs)
    row = {"变体": nm, "入场次数": len(e)}
    for hname in HOR:
        row[f"{hname}中位"] = e[hname].median()
        row[f"{hname}胜率"] = (e[hname] > 0).mean()
    row["6M最大逆行"] = e["MAE6M"].median()
    row["3M维持率"] = e["hold3M"].median()
    res.append(row)
ev = pd.DataFrame(res)
d2 = ev.copy()
for c in d2.columns:
    if c not in ("变体", "入场次数"):
        d2[c] = (d2[c]*100).round(1).astype(str) + "%"
print(d2.to_string(index=False))
ev.to_csv(OUT / "t5_intraday_entries.csv", index=False)

# ================================================================
# 4. Where does today sit?
# ================================================================
print("\n\n=== 4. 当前状态（GLD）===")
print(f"日期 {idx[-1].date()}  收盘 {px.iloc[-1]:.2f}  200DMA {ma.iloc[-1]:.2f}  "
      f"相对 {(px.iloc[-1]/ma.iloc[-1]-1)*100:+.2f}%")
print(f"当日高点 {hi.iloc[-1]:.2f} ({(hi.iloc[-1]/ma.iloc[-1]-1)*100:+.2f}% vs MA)  "
      f"低点 {lo.iloc[-1]:.2f}")
print(f"1% 缓冲门槛 = {ma.iloc[-1]*1.01:.2f}   2% 缓冲门槛 = {ma.iloc[-1]*1.02:.2f}")
print("\n输出: output/t5_*.csv, data/gld_ohlc.csv")
