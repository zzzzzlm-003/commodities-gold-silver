"""Task 4: does the *definition* of the 200DMA reclaim trigger matter?

Tests the current rule (Friday close > 200DMA) against alternatives:
weekly vs daily close, N-day confirmation, and a % buffer above the MA.
"""
import pandas as pd
import numpy as np
from common import load, metrics, run_filtered, whipsaws, OUT

pd.set_option("display.width", 250)
xau = load("xau")
ma = xau.rolling(200).mean()
above = xau > ma

# ---------- signal variants (0/1 at each daily close) ----------
def weekly(sig_daily):
    """Evaluate only on Fridays, hold that state through the week."""
    wk = sig_daily.resample("W-FRI").last().dropna()
    return wk.reindex(sig_daily.index).ffill().fillna(0.0)

VARIANTS = {}
VARIANTS["现规则 周五收盘>MA"] = weekly(above.astype(float))
VARIANTS["任意日收盘>MA"] = above.astype(float)
for n in (3, 5, 10):
    VARIANTS[f"连续{n}日收盘>MA"] = (above.rolling(n).sum() == n).astype(float)
for b in (0.005, 0.01, 0.02):
    VARIANTS[f"收盘>MA×(1+{b*100:.1f}%)"] = (xau > ma * (1 + b)).astype(float)
# combos: buffer applied on the weekly check
VARIANTS["周五收盘>MA×1.01"] = weekly((xau > ma * 1.01).astype(float))

# ---------- A. full long/flat backtest ----------
rows = []
for start, lab in [("1975-01-01", "1975-"), ("1990-01-01", "1990-"), ("2005-01-01", "2005-")]:
    px = xau.loc[start:]
    for nm, sig in VARIANTS.items():
        eq, pos, n = run_filtered(px, sig.reindex(px.index).fillna(0.0))
        m = metrics(eq, pos, nm)
        m["交易次数"] = n
        _, nws, wsloss = whipsaws(px, pos)
        m["whipsaw次数"] = nws
        m["whipsaw损耗"] = wsloss
        m["样本"] = lab
        rows.append(m)
bt = pd.DataFrame(rows)[["样本", "策略", "CAGR", "年化波动", "最大回撤", "Calmar",
                          "在场比例", "交易次数", "whipsaw次数", "whipsaw损耗"]]
bt.to_csv(OUT / "t4_trigger_backtests.csv", index=False)
print("=== A. 长期多/空仓回测（5bp/边，次日执行）===")
for lab in ("1975-", "1990-", "2005-"):
    sub = bt[bt.样本 == lab].copy()
    for c in ("CAGR", "年化波动", "最大回撤", "在场比例"):
        sub[c] = (sub[c] * 100).round(1).astype(str) + "%"
    sub["Calmar"] = sub["Calmar"].round(2)
    sub["whipsaw损耗"] = (sub["whipsaw损耗"] * 100).round(1).astype(str) + "%"
    print(f"\n-- 样本 {lab} --")
    print(sub.drop(columns="样本").to_string(index=False))

# ---------- B. entry event study ----------
# A "fresh entry" = signal flips 0->1 after >=60 days flat (a genuine regime re-entry,
# not an intra-chop flicker).
def entries(sig, min_gap=60):
    s = sig.fillna(0.0)
    flips = s.diff().fillna(0)
    ups = list(s.index[flips > 0])
    out, last = [], None
    for d in ups:
        if last is None or (d - last).days >= min_gap:
            out.append(d)
        last = d
    return out

HOR = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}
ev_rows = []
for nm, sig in VARIANTS.items():
    ds = entries(sig)
    for d in ds:
        i = xau.index.get_loc(d)
        p0 = xau.iloc[i]
        rec = {"变体": nm, "日期": d.date(), "入场价": round(p0, 1),
               "高出MA": p0 / ma.iloc[i] - 1}
        for h, k in HOR.items():
            rec[h] = xau.iloc[i + k] / p0 - 1 if i + k < len(xau) else np.nan
        # max adverse excursion over next 6M
        fwd = xau.iloc[i:i + 126]
        rec["6M内最大逆行"] = fwd.min() / p0 - 1
        # did it stay above? fraction of next 63d with close > MA
        rec["3M内维持在MA上方比例"] = (xau.iloc[i:i + 63] > ma.iloc[i:i + 63]).mean()
        ev_rows.append(rec)
ev = pd.DataFrame(ev_rows)
ev.to_csv(OUT / "t4_entries.csv", index=False)

print("\n\n=== B. 入场事件研究（信号 0->1，间隔≥60天）===")
agg = []
for nm in VARIANTS:
    e = ev[ev.变体 == nm]
    r = {"变体": nm, "入场次数": len(e), "高出MA中位": e["高出MA"].median()}
    for h in HOR:
        r[f"{h}中位"] = e[h].median()
        r[f"{h}胜率"] = (e[h] > 0).mean()
    r["6M最大逆行中位"] = e["6M内最大逆行"].median()
    r["3M维持率"] = e["3M内维持在MA上方比例"].median()
    agg.append(r)
agg = pd.DataFrame(agg)
disp = agg.copy()
for c in disp.columns:
    if c not in ("变体", "入场次数"):
        disp[c] = (disp[c] * 100).round(1).astype(str) + "%"
print(disp.to_string(index=False))
agg.to_csv(OUT / "t4_entry_summary.csv", index=False)

# ---------- C. the specific question: does a thin margin matter? ----------
print("\n\n=== C. 首次站上时的\"幅度\"是否有信息量？（用任意日收盘变体的全部入场）===")
e = ev[ev.变体 == "任意日收盘>MA"].dropna(subset=["3M"])
bins = [(-1, 0.005, "<0.5% (贴线)"), (0.005, 0.015, "0.5-1.5%"), (0.015, 9, ">1.5%")]
for lo, hi, lab in bins:
    s = e[(e["高出MA"] > lo) & (e["高出MA"] <= hi)]
    if len(s) == 0:
        continue
    print(f"{lab:14s} n={len(s):3d}  3M中位={s['3M'].median()*100:+5.1f}%  "
          f"3M胜率={(s['3M']>0).mean()*100:4.1f}%  "
          f"12M中位={s['12M'].median()*100:+5.1f}%  "
          f"3M维持率中位={s['3M内维持在MA上方比例'].median()*100:4.1f}%")

# ---------- D. head-to-head: weekly vs daily, same entries? ----------
print("\n\n=== D. 周五收盘 vs 任意日收盘：入场时点差多少？===")
ew = ev[ev.变体 == "现规则 周五收盘>MA"].set_index("日期")
ed = ev[ev.变体 == "任意日收盘>MA"].set_index("日期")
pairs = []
for d in ed.index:
    later = [x for x in ew.index if 0 <= (pd.Timestamp(x) - pd.Timestamp(d)).days <= 30]
    if later:
        w = ew.loc[later[0]]
        pairs.append({"日线入场": d, "周线入场": later[0],
                      "延迟天数": (pd.Timestamp(later[0]) - pd.Timestamp(d)).days,
                      "价格差": w["入场价"] / ed.loc[d, "入场价"] - 1})
p = pd.DataFrame(pairs)
if len(p):
    print(f"配对数={len(p)}  中位延迟={p.延迟天数.median():.0f}天  "
          f"中位价格差={p.价格差.median()*100:+.2f}%  "
          f"平均价格差={p.价格差.mean()*100:+.2f}%")
    print(f"周线入场更贵的比例: {(p.价格差>0).mean()*100:.0f}%")
print("\n输出: output/t4_*.csv")
