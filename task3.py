"""Task 3: silver vs PV mean reversion; gold/silver ratio z-score."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from common import load, OUT

pd.set_option("display.width", 220)
xau, xag = load("xau"), load("xag")

# ---------- 1. annual data ----------
# Global new PV installations, GW-DC (BNEF / PV-Tech / IEA-PVPS; 2025 = PV Tech 647 GW)
pv_gw = {2010: 17, 2011: 31, 2012: 30, 2013: 39, 2014: 40, 2015: 51, 2016: 76,
         2017: 99, 2018: 104, 2019: 118, 2020: 139, 2021: 175, 2022: 239,
         2023: 456, 2024: 602, 2025: 647}
# PV silver demand, Moz (Silver Institute / Metals Focus World Silver Survey; pre-2019 approximate)
pv_ag = {2014: 59.9, 2015: 71.6, 2016: 93.7, 2017: 101.1, 2018: 92.5,
         2019: 98.7, 2020: 101.0, 2021: 113.7, 2022: 140.3, 2023: 193.5,
         2024: 197.6, 2025: 195.7}

ag_year = xag.resample("YE").mean()
ag_year.index = ag_year.index.year
df = pd.DataFrame({"ag_avg": ag_year})
df["pv_gw"] = pd.Series(pv_gw)
df["pv_ag_moz"] = pd.Series(pv_ag)
df = df.loc[2010:2025]
df["ag_per_gw"] = df.ag_avg / df.pv_gw          # $/oz per GW installed
df["ag_per_moz"] = df.ag_avg / df.pv_ag_moz     # $/oz per Moz PV silver demand
for c in ["ag_per_gw", "ag_per_moz"]:
    s = df[c].dropna()
    df[c + "_z"] = (df[c] - s.mean()) / s.std()
df.round(3).to_csv(OUT / "t3_silver_pv.csv")
print(df.round(3).to_string())

# ---------- 2. mean reversion stats ----------
def ar1_stats(s, name):
    s = np.log(s.dropna())
    x, y = s.shift(1).dropna(), s.iloc[1:]
    x, y = x.align(y, join="inner")
    rho = np.polyfit(x, y - x.mean(), 1)[0] if False else np.corrcoef(x, y)[0, 1]
    # proper AR1 coef via OLS on demeaned
    xd, yd = x - x.mean(), y - y.mean()
    phi = (xd * yd).sum() / (xd * xd).sum()
    hl = np.log(0.5) / np.log(abs(phi)) if 0 < phi < 1 else np.inf
    try:
        adf = adfuller(s, maxlag=1, regression="c")
        adf_p = adf[1]
    except Exception:
        adf_p = np.nan
    print(f"{name}: n={len(s)}, AR(1) phi={phi:.3f}, half-life={hl:.1f}y, ADF p={adf_p:.3f}")
    return phi, hl, adf_p

print("\n--- mean reversion (annual, SMALL SAMPLE) ---")
ar1_stats(df.ag_per_gw, "银价/装机GW")
ar1_stats(df.ag_per_moz.loc[2014:], "银价/光伏用银Moz")

# ---------- 3. gold/silver ratio ----------
m = pd.concat([xau, xag], axis=1).resample("ME").last().dropna()
gsr = (m.xau / m.xag).rename("gsr")
gsr_z = (gsr - gsr.mean()) / gsr.std()
fwd1y = m.xag.shift(-12) / m.xag - 1

hi = gsr > 85
lo = gsr < 65
res = {}
for nm, mask in [("GSR>85", hi), ("GSR<65", lo), ("全样本", gsr.notna())]:
    f = fwd1y[mask].dropna()
    res[nm] = {"n月": len(f), "1年后银价收益均值": f.mean(), "中位数": f.median(),
               "胜率>0": (f > 0).mean(), "p25": f.quantile(.25), "p75": f.quantile(.75)}
gsr_tbl = pd.DataFrame(res).T
print("\n", (gsr_tbl * 100).round(1).astype(str) + "%")
gsr_tbl.to_csv(OUT / "t3_gsr_fwd.csv")
pd.DataFrame({"gsr": gsr, "z": gsr_z, "fwd1y_ag": fwd1y}).to_csv(OUT / "t3_gsr.csv")
print("\ncurrent GSR: %.1f (z=%.2f)" % (gsr.iloc[-1], gsr_z.iloc[-1]))

# ---------- charts ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
ax = axes[0, 0]
ax.plot(df.index, df.ag_per_gw, "o-")
ax.set_title("银价年均 / 光伏年新增装机 ($/oz per GW)")
ax = axes[0, 1]
ax.plot(df.index, df.ag_per_moz, "o-", color="darkorange")
ax.set_title("银价年均 / 光伏用银量 ($/oz per Moz)")
ax = axes[1, 0]
ax.plot(df.index, df.ag_per_gw_z, "o-", label="/GW z")
ax.plot(df.index, df.ag_per_moz_z, "o-", label="/Moz z")
ax.axhline(0, color="k", lw=0.5); ax.axhline(1, color="gray", ls="--", lw=0.5); ax.axhline(-1, color="gray", ls="--", lw=0.5)
ax.legend(); ax.set_title("比值 z-score")
ax = axes[1, 1]
ax.plot(df.index, df.ag_avg, "o-", color="gray", label="银价年均")
ax2 = ax.twinx(); ax2.plot(df.index, df.pv_gw, "s--", color="green", label="PV GW")
ax.legend(loc="upper left"); ax2.legend(loc="lower right"); ax.set_title("银价 vs 光伏装机")
plt.tight_layout(); plt.savefig(OUT / "t3_silver_pv.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(gsr, lw=0.8)
ax.axhline(85, color="red", ls="--", lw=0.8); ax.axhline(65, color="green", ls="--", lw=0.8)
ax.fill_between(gsr.index, 85, gsr.where(gsr > 85), color="red", alpha=0.3)
ax.fill_between(gsr.index, gsr.where(gsr < 65), 65, color="green", alpha=0.3)
ax.set_title("金银比 (月频, 1968-)  红=GSR>85 (银相对便宜)  绿=GSR<65 (银相对贵)")
plt.tight_layout(); plt.savefig(OUT / "t3_gsr.png", dpi=130); plt.close()
print("charts saved")
