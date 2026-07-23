"""Weekly checklist: refresh data then print the 6 numbers from report.md."""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent
if "--refresh" in sys.argv:
    subprocess.run(["curl", "-s", "-A", "Mozilla/5.0",
                    "https://prices.lbma.org.uk/json/gold_pm.json", "-o", ROOT / "data/gold_pm.json"])
    subprocess.run(["curl", "-s", "-A", "Mozilla/5.0",
                    "https://prices.lbma.org.uk/json/silver.json", "-o", ROOT / "data/silver.json"])
    subprocess.run([sys.executable, ROOT / "prep_data.py"])

from common import load
xau, xag = load("xau"), load("xag")
mg, ms = xau.rolling(200).mean().iloc[-1], xag.rolling(200).mean().iloc[-1]
g, s = xau.iloc[-1], xag.iloc[-1]
athg, aths = xau.max(), xag.max()
print(f"数据截至 {xau.index[-1].date()}")
print(f"1) 金 {g:.0f} vs 200DMA {mg:.0f} ({g/mg-1:+.1%}) — {'上方✓右侧确认' if g > mg else '下方，等待确认'}")
low = xau.loc[xau.idxmax():].min()  # lowest since ATH: rung is triggered once crossed
print(f"2) 金距 ATH {athg:.0f}: {g/athg-1:+.1%} (本轮最低 {low:.0f}) | 左侧档位 " +
      " ".join(f"-{int(r*100)}%={athg*(1-r):.0f}{'✓已触发' if low <= athg*(1-r) else ''}" for r in (.08, .16, .24, .32)))
print(f"3) 金银比 GSR = {g/s:.1f}  (>85 做多银信号 / <65 银偏贵)")
print(f"4) 银 {s:.2f} vs 200DMA {ms:.2f} ({s/ms-1:+.1%}) | 距 ATH {aths:.1f}: {s/aths-1:+.1%}")
vol = xau.pct_change().iloc[-63:].std() * (252 ** 0.5)
print(f"5) 金 3个月滚动年化波动 {vol:.0%} — {'UGL 磨损高危区(>20%且横盘)' if vol > 0.20 else '正常'}")
print(f"6) 光伏比值：每年4月 WSS 出版后更新 task3.py")
