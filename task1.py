"""Task 1: cost of right-side (200DMA reclaim) confirmation vs left-side laddering, gold."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, metrics, run_filtered, run_ladder, whipsaws, OUT

pd.set_option("display.width", 220)
xau = load("xau")
ma200 = xau.rolling(200).mean()

# ---------- 1-4: zigzag cycle identification ----------
def zigzag(px, thresh):
    """Return list of (peak_date, trough_date) pairs where decline >= thresh."""
    cycles = []
    peak_i = trough_i = 0
    state = "up"  # tracking a rising leg from last pivot
    vals = px.values; idx = px.index
    for i in range(1, len(px)):
        if state == "up":
            if vals[i] > vals[peak_i]:
                peak_i = i
            elif vals[i] <= vals[peak_i] * (1 - thresh):
                state = "down"; trough_i = i
        else:
            if vals[i] < vals[trough_i]:
                trough_i = i
            elif vals[i] >= vals[trough_i] * (1 + thresh):
                cycles.append((idx[peak_i], idx[trough_i]))
                state = "up"; peak_i = i
    if state == "down":  # ongoing episode
        cycles.append((idx[peak_i], idx[trough_i]))
    return cycles


def cycle_table(px, ma, cycles, ladder_pcts=(0.10, 0.15, 0.20, 0.25, 0.30)):
    rows = []
    for pk, tr in cycles:
        peak, trough = px.loc[pk], px.loc[tr]
        depth = trough / peak - 1
        after = px.loc[tr:].iloc[1:]
        ma_after = ma.loc[after.index]
        above = after[after > ma_after]
        # first close back above 200DMA after trough
        if len(above):
            cdate, cprice = above.index[0], above.iloc[0]
            conf_prem = cprice / trough - 1
            wait = (cdate - tr).days
        else:
            cdate = cprice = conf_prem = wait = None
        row = {"peak_date": pk.date(), "peak": round(peak, 1),
               "trough_date": tr.date(), "trough": round(trough, 1),
               "depth": depth,
               "confirm_date": cdate.date() if cdate is not None else "—",
               "confirm_px": round(cprice, 1) if cprice else "—",
               "confirm_prem": conf_prem, "wait_days": wait}
        for lp in ladder_pcts:
            lvl = peak * (1 - lp)
            row[f"L{int(lp*100)}"] = lvl / trough - 1 if trough <= lvl else None
        rows.append(row)
    return pd.DataFrame(rows)


cyc15 = zigzag(xau, 0.15)
cyc10 = zigzag(xau, 0.10)
t15 = cycle_table(xau, ma200, cyc15)
t10 = cycle_table(xau, ma200, cyc10)
t15.to_csv(OUT / "t1_cycles_15.csv", index=False)
t10.to_csv(OUT / "t1_cycles_10.csv", index=False)
print("=== cycles >=15% ==="); print(t15)
print("=== cycles >=10% ==="); print(t10)

# summary answers (>=15% table, completed cycles with confirmation)
done = t15.dropna(subset=["confirm_prem"])
print("\nconfirm premium mean/median/worst: %.1f%% / %.1f%% / %.1f%%" % (
    done.confirm_prem.mean()*100, done.confirm_prem.median()*100, done.confirm_prem.max()*100))
l20 = done["L20"].dropna()
print("L20 premium mean %.1f%% (n=%d triggered)" % (l20.mean()*100, len(l20)))

# ---------- 5: strategy backtests ----------
def run_all(px, start, label):
    px = px.loc[start:]
    ma = px.rolling(200).mean()  # NB: uses only in-sample window after start+200d
    # recompute MA on full history to avoid burn-in loss
    ma = xau.rolling(200).mean().reindex(px.index) if px.name == "xau" else px.rolling(200).mean()
    res, curves = [], {}

    eqA = px / px.iloc[0]
    res.append(metrics(eqA, None, "A 买入持有")); res[-1]["交易次数"] = 1
    curves["A 买入持有"] = eqA

    sigD = (px > ma).astype(float)
    eqB1, posB1, nB1 = run_filtered(px, sigD)
    m = metrics(eqB1, posB1, "B1 200DMA 日频"); m["交易次数"] = nB1
    trips, nws, wsloss = whipsaws(px, posB1)
    m["whipsaw次数"] = nws; m["whipsaw总损耗"] = wsloss
    res.append(m); curves["B1 日频"] = eqB1

    wk = px.resample("W-FRI").last().dropna()
    sigW_w = (wk > ma.reindex(wk.index)).astype(float)
    sigW = sigW_w.reindex(px.index).ffill().fillna(0)
    eqB2, posB2, nB2 = run_filtered(px, sigW)
    m = metrics(eqB2, posB2, "B2 200DMA 周频"); m["交易次数"] = nB2
    trips, nws, wsloss = whipsaws(px, posB2)
    m["whipsaw次数"] = nws; m["whipsaw总损耗"] = wsloss
    res.append(m); curves["B2 周频"] = eqB2

    for rungs, nm in [((0.05,0.10,0.15,0.20), "C1 -5/-10/-15/-20"),
                      ((0.08,0.16,0.24,0.32), "C2 -8/-16/-24/-32"),
                      ((0.10,0.20,0.30,0.40), "C3 -10/-20/-30/-40")]:
        eqC, posC, nC = run_ladder(px, rungs)
        m = metrics(eqC, posC, nm); m["交易次数"] = nC
        res.append(m); curves[nm] = eqC
        if nm.startswith("C2"):
            eqC2, posC2 = eqC, posC

    eqD = 0.5 * eqB2 + 0.5 * eqC2
    m = metrics(eqD, 0.5*posB2 + 0.5*posC2, "D 半右侧+半左侧(C2)")
    m["交易次数"] = nB2 + 4
    res.append(m); curves["D 混合"] = eqD

    df = pd.DataFrame(res)
    df.insert(0, "样本", label)
    return df, curves

r05, cv05 = run_all(xau, "2005-01-01", "XAU 2005-")
r90, cv90 = run_all(xau, "1990-01-01", "XAU 1990-")
r75, cv75 = run_all(xau, "1975-01-01", "XAU 1975-")
allres = pd.concat([r05, r90, r75])
allres.to_csv(OUT / "t1_backtests.csv", index=False)
print("\n", allres.to_string(index=False))

# ---------- charts ----------
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)
ax = axes[0]
ax.plot(xau.loc["2000":], lw=0.8, label="XAU (LBMA PM fix)")
ax.plot(ma200.loc["2000":], lw=0.8, label="200DMA", color="orange")
for _, row in t15.iterrows():
    if pd.Timestamp(row.peak_date) < pd.Timestamp("2000"): continue
    ax.axvspan(pd.Timestamp(row.peak_date), pd.Timestamp(row.trough_date), alpha=0.15, color="red")
    ax.scatter([pd.Timestamp(row.trough_date)], [row.trough], color="red", zorder=5, s=25)
    if row.confirm_date != "—":
        ax.scatter([pd.Timestamp(row.confirm_date)], [row.confirm_px], color="green", zorder=5, s=25)
ax.set_yscale("log"); ax.legend(); ax.set_title("Gold drawdown cycles (>=15%), troughs (red) vs 200DMA reclaim (green)")
ax = axes[1]
for nm, eq in cv05.items():
    ax.plot(eq, lw=1, label=nm)
ax.set_yscale("log"); ax.legend(fontsize=8); ax.set_title("Strategy equity curves, XAU 2005- (5bp/side, next-day exec)")
plt.tight_layout(); plt.savefig(OUT / "t1_gold.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(13, 5))
for nm, eq in cv75.items():
    ax.plot(eq, lw=1, label=nm)
ax.set_yscale("log"); ax.legend(fontsize=8); ax.set_title("Strategy equity curves, XAU 1975- (log)")
plt.tight_layout(); plt.savefig(OUT / "t1_gold_long.png", dpi=130); plt.close()
print("charts saved")
