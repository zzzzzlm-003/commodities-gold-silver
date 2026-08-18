"""Task 6: 东吴/芦哲月度黄金报告的季节性主张能否证伪？

他们的主张（《黄金ETF 2026年7月复盘与8月展望》，标的=沪金 SHFE 人民币/克）：
  7月 历史平均 +0.92%，正收益概率 55.50%
  8月 历史平均 +2.09%，胜率 54.66%，"全年表现最好的月份之一"

这里用伦敦金（LBMA 定盘价，美元/盎司）复核。口径不同点已知：
标的（沪金 vs 伦敦金）、计价货币（含 USDCNY）、样本起点（SHFE 2008 上市）。
所以同时报全样本与 2008+ 子样本。
"""
import pandas as pd
from common import load, OUT

pd.set_option("display.width", 250)
xau = load("xau")

# 月度收益：月末收盘对月末收盘
m = xau.resample("ME").last()
mret = m.pct_change().dropna()

CLAIM = {7: (0.0092, 0.5550), 8: (0.0209, 0.5466)}
MONTHS = ["1月", "2月", "3月", "4月", "5月", "6月",
          "7月", "8月", "9月", "10月", "11月", "12月"]


def table(r, label):
    rows = []
    for mo in range(1, 13):
        s = r[r.index.month == mo]
        rows.append({"月份": MONTHS[mo - 1], "n": len(s),
                     "平均": s.mean(), "中位": s.median(),
                     "胜率": (s > 0).mean()})
    df = pd.DataFrame(rows)
    df["平均排名"] = df["平均"].rank(ascending=False).astype(int)
    print(f"\n=== {label}（伦敦金 LBMA，美元/盎司）===")
    disp = df.copy()
    for c in ("平均", "中位", "胜率"):
        disp[c] = (disp[c] * 100).round(2).astype(str) + "%"
    print(disp.to_string(index=False))
    return df


full = table(mret, f"A. 全样本 {mret.index[0].year}-{mret.index[-1].year}")
recent = table(mret[mret.index >= "2008-01-01"], "B. 2008- 子样本（沪金上市后，可比口径）")

print("\n\n=== C. 对账：他们的沪金数字 vs 伦敦金实测 ===")
for mo, (c_mean, c_win) in CLAIM.items():
    nm = MONTHS[mo - 1]
    for df, lab in ((full, "全样本"), (recent, "2008-")):
        row = df[df.月份 == nm].iloc[0]
        print(f"{nm} {lab:6s}: 平均 {row['平均']*100:+5.2f}% (他们 {c_mean*100:+.2f}%, "
              f"差 {(row['平均']-c_mean)*100:+5.2f}pp)  |  "
              f"胜率 {row['胜率']*100:4.1f}% (他们 {c_win*100:.2f}%, "
              f"差 {(row['胜率']-c_win)*100:+5.1f}pp)  |  "
              f"平均排名 {row['平均排名']}/12  n={row['n']}")

print("\n=== D. \"8月是全年最好月份之一\"是否成立？===")
for df, lab in ((full, "全样本"), (recent, "2008-")):
    top3 = df.nlargest(3, "平均")["月份"].tolist()
    aug = df[df.月份 == "8月"].iloc[0]
    print(f"{lab:6s}: 平均收益前三 = {', '.join(top3)}；"
          f"8月排名 {aug['平均排名']}/12 → {'成立' if aug['平均排名'] <= 4 else '不成立'}")

full.to_csv(OUT / "t6_seasonality_full.csv", index=False)
recent.to_csv(OUT / "t6_seasonality_2008.csv", index=False)
print("\n输出: output/t6_seasonality_*.csv")
