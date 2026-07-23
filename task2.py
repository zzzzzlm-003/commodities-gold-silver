"""Task 2: GLD tracking difference vs spot; UGL vs 2x GLD decay decomposition."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, OUT

pd.set_option("display.width", 220)
xau, gld, ugl = load("xau"), load("gld"), load("ugl")

# ---------- 1. GLD vs spot ----------
both = pd.concat([xau, gld], axis=1, join="inner").dropna()
yr_ret = both.resample("YE").last() / both.resample("YE").first() - 1  # approx: use year first/last common day
# better: annual returns from Dec-to-Dec last obs
ann = both.resample("YE").last().pct_change().dropna()
ann["TD"] = ann.gld - ann.xau
full_years = ann.index.year[(ann.index.year >= 2005) & (ann.index.year <= 2025)]
td = ann.loc[ann.index.year.isin(full_years), "TD"]
print("GLD-XAU annual tracking diff: mean %.2f%%, median %.2f%%, std %.2f%%" %
      (td.mean()*100, td.median()*100, td.std()*100))
# geometric over full period
n_yrs = (both.index[-1] - both.index[0]).days / 365.25
g_gld = (both.gld.iloc[-1]/both.gld.iloc[0])**(1/n_yrs)-1
g_xau = (both.xau.iloc[-1]/both.xau.iloc[0])**(1/n_yrs)-1
print("Full-period CAGR: GLD %.2f%%, XAU %.2f%%, diff %.2f%%/yr" % (g_gld*100, g_xau*100, (g_gld-g_xau)*100))
ann_out = ann.copy(); ann_out.index = ann_out.index.year
ann_out.to_csv(OUT / "t2_gld_tracking.csv")

# ---------- 2. UGL vs 2x GLD ----------
pair = pd.concat([gld, ugl], axis=1, join="inner").dropna()
rg, ru = pair.gld.pct_change().fillna(0), pair.ugl.pct_change().fillna(0)
synth = (1 + 2 * rg).cumprod()  # ideal 2x daily, costless

def yearly(x):
    return x.resample("YE").apply(lambda s: (1 + s).prod() - 1)

Y = pd.DataFrame({
    "GLD": yearly(rg), "UGL": yearly(ru),
    "synth2x_daily": yearly(2 * rg),
})
Y["2xGLD_ann"] = 2 * Y.GLD
Y["总差值 UGL-2xGLD_ann"] = Y.UGL - Y["2xGLD_ann"]
Y["费用融资差 UGL-synth"] = Y.UGL - Y.synth2x_daily          # fees + financing + tracking noise
Y["波动损耗 synth-2xann"] = Y.synth2x_daily - Y["2xGLD_ann"]  # compounding / vol drag (can be + in trends)
Y["GLD年化波动"] = rg.resample("YE").std() * np.sqrt(252)
Y.index = Y.index.year
Y = Y[(Y.index >= 2009) & (Y.index <= 2026)]
Y.to_csv(OUT / "t2_ugl_yearly.csv")
print("\n", (Y * 100).round(1).to_string())

# rolling 1y tracking difference (UGL vs ideal 2x daily)
rel = (pair.ugl / pair.ugl.iloc[0]) / synth
roll_td = rel.pct_change(252).dropna()  # 1y drift of UGL vs ideal 2x-daily
roll_vs2x = ((1 + ru).rolling(252).apply(np.prod, raw=True) -
             (1 + 2 * (1 + rg).rolling(252).apply(np.prod, raw=True) - 2)).dropna()
print("\nRolling 1y: UGL vs ideal-2x-daily drift: mean %.2f%%, p5 %.2f%%, p95 %.2f%%" %
      (roll_td.mean()*100, roll_td.quantile(.05)*100, roll_td.quantile(.95)*100))
print("Rolling 1y: UGL ret minus 2x(GLD 1y ret): mean %.2f%%, median %.2f%%, p5 %.2f%%, p95 %.2f%%" %
      (roll_vs2x.mean()*100, roll_vs2x.median()*100, roll_vs2x.quantile(.05)*100, roll_vs2x.quantile(.95)*100))
pd.DataFrame({"roll_fee": roll_td, "roll_vs2x": roll_vs2x}).to_csv(OUT / "t2_ugl_rolling.csv")

# trend vs chop split: |GLD annual ret| vs vol
Y2 = Y.dropna()
trend = Y2[Y2.GLD.abs() >= 0.15]
chop = Y2[Y2.GLD.abs() < 0.15]
for nm, g in [("趋势年 |GLD|>=15%", trend), ("震荡年 |GLD|<15%", chop)]:
    print("%s (n=%d): 总差 %.1f%%, 费用差 %.1f%%, 波动损耗 %.1f%%" % (
        nm, len(g), g["总差值 UGL-2xGLD_ann"].mean()*100,
        g["费用融资差 UGL-synth"].mean()*100, g["波动损耗 synth-2xann"].mean()*100))

# ---------- charts ----------
fig, axes = plt.subplots(2, 1, figsize=(13, 9))
ax = axes[0]
x = np.arange(len(Y2))
ax.bar(x - 0.2, Y2["UGL"] * 100, 0.4, label="UGL 实际年收益")
ax.bar(x + 0.2, Y2["2xGLD_ann"] * 100, 0.4, label="2×GLD 年收益")
ax.set_xticks(x); ax.set_xticklabels(Y2.index, rotation=45)
ax.legend(); ax.set_ylabel("%"); ax.set_title("UGL vs 2×GLD 逐年收益")
ax2 = axes[1]
ax2.plot(roll_vs2x * 100, lw=0.9, label="滚动1年: UGL − 2×GLD(1年收益)")
ax2.plot(roll_td * 100, lw=0.9, label="滚动1年: UGL − 理想2×日复利 (费用+融资)")
ax2.axhline(0, color="k", lw=0.5)
ax2.legend(); ax2.set_ylabel("%"); ax2.set_title("UGL 滚动1年跟踪差")
plt.tight_layout(); plt.savefig(OUT / "t2_ugl.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.bar(td.index.year, td * 100)
ax.set_title("GLD − XAU 现货 年度跟踪差 (%)"); ax.set_ylabel("%")
plt.tight_layout(); plt.savefig(OUT / "t2_gld_td.png", dpi=130); plt.close()
print("charts saved")
