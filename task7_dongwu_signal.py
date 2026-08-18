"""Task 7: 东吴/芦哲的"局部底右侧信号" vs 我们的 200DMA×1.01 —— 哪个入场点更好？

背景与口径限制（重要，别当成原版复刻）：
  东吴《技术分析系列：双维框架研究之动能驱动与风险管控》(2025-05-05) 公布了信号条件：
    局部底左侧：TR<20, tmp_xl>50, jax_xl<50, trend>50
    局部底右侧：TR<20, tmp_xl>50, jax_xl>50
  其中 JAX/TMP（济安线）公式是公开的，可精确复刻；
  但 TR（风险度）只公布了"75/20 交易期窗口 + OHLC 四价 + 相关平滑计算"，**公式未披露**。
  我们用 7 个公开的沪金 TR 月末实测值反解，最好只能到 RMSE≈5.6，
  且最优参数(60,MA5)与它自己公布的(75,20)矛盾 —— 判定为**不可忠实复现**。
  因此这里用"透明位置分"POS = MA(75日stoch, 20) 替代 TR，按其公布参数，不调参。
  结论只能读作"东吴式"信号，不是东吴原版。

  斜率标准化：原文"斜率数据标准化处理"，>50 即上行。信号只用到符号，
  故直接以 JAX/TMP 的一阶差分符号实现（xl>50 ⟺ 该线上行）。
"""
import pandas as pd
import numpy as np
from common import load, OUT

pd.set_option("display.width", 250)

# ---------- 济安线 JAX(慢) / TMP(快) ----------
# 月报 6 期一致写"济安线的慢线JAX与快线TMP" → JAX=慢线, TMP=快线
# （方法论报告 2.2 节写成"快线JAX与慢线TMP"，与月报矛盾，取月报）
def dma(x, a):
    """通达信 DMA(X,A)：变权移动平均 Y_t = a_t*X_t + (1-a_t)*Y_{t-1}"""
    a = a.clip(lower=1e-6, upper=1.0)
    out = np.empty(len(x))
    prev = x.iloc[0]
    for i, (xi, ai) in enumerate(zip(x.to_numpy(), a.to_numpy())):
        if np.isnan(xi) or np.isnan(ai):
            out[i] = prev
            continue
        prev = ai * xi + (1 - ai) * prev
        out[i] = prev
    return pd.Series(out, index=x.index)


def jiaan(o, h, l, c, n):
    """济安线：JAX 与辅助快线 TMP（公开通达信公式）"""
    typ = (2 * c + h + l) / 4
    aa = (typ - c.rolling(n).mean()).abs() / c.rolling(n).mean()
    jax = dma((2 * c + l + h) / 4, aa)
    ma1 = (c / jax * typ).rolling(3).mean()
    maaa = ((ma1 - jax) / jax) / 3
    tmp = ma1 - maaa * ma1
    return jax, tmp


def pos_score(h, l, c, win=75, smooth=20):
    """透明位置分（替代未披露的 TR），按东吴公布的 75/20 参数"""
    hh, ll = h.rolling(win).max(), l.rolling(win).min()
    return (100 * (c - ll) / (hh - ll)).rolling(smooth).mean()


# ---------- A. 用他们月报的趋势描述校准济安线参数 N ----------
au = pd.read_csv("data/au0.csv", parse_dates=["date"], index_col="date")
# 他们 6 期月报对"快线(TMP)/慢线(JAX)"方向的定性描述（月末）
TREND_DESC = {
    "2026-01-30": ("up", "up"),      # 快线在慢线上方、大正向开口
    "2026-02-27": ("up", "up"),      # 快线上穿慢线，新多头信号
    "2026-03-31": ("up", "down"),    # 快线上行、慢线下行
    "2026-04-30": ("down", "down"),  # 快慢线均下行
    "2026-05-29": ("down", "down"),  # 快慢线均下行
}


def at(s, d):
    i = s.index[s.index <= pd.Timestamp(d)]
    return s.loc[i[-1]] if len(i) else np.nan


print("=== A. 用月报趋势描述反推济安线参数 N ===")
best = None
for n in (5, 10, 13, 21, 26, 34):
    jax, tmp = jiaan(au.open, au.high, au.low, au.close, n)
    dj, dt = jax.diff(), tmp.diff()
    hit = 0
    for d, (want_t, want_j) in TREND_DESC.items():
        gt, gj = at(dt, d), at(dj, d)
        if (gt > 0) == (want_t == "up"):
            hit += 1
        if (gj > 0) == (want_j == "up"):
            hit += 1
    print(f"  N={n:2d}  命中 {hit}/{2*len(TREND_DESC)} 个方向描述")
    if best is None or hit > best[1]:
        best = (n, hit)
N = best[0]
print(f"  → 取 N={N}（命中 {best[1]}/{2*len(TREND_DESC)}）")

# ---------- B. 两个"右侧"在同一标的上对撞 ----------
HOR = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}


def entries(sig, min_gap=60):
    """信号 0->1 且距上次入场 >=min_gap 天，避免同一段行情反复计数"""
    s = sig.fillna(False).astype(float)
    ups = list(s.index[s.diff().fillna(0) > 0])
    out, last = [], None
    for d in ups:
        if last is None or (d - last).days >= min_gap:
            out.append(d)
            last = d
    return out


def study(px, ds, label, ma=None):
    rows = []
    for d in ds:
        i = px.index.get_loc(d)
        p0 = px.iloc[i]
        rec = {"信号": label, "日期": d.date(), "价格": round(p0, 1)}
        for h, k in HOR.items():
            rec[h] = px.iloc[i + k] / p0 - 1 if i + k < len(px) else np.nan
        # 相对"事后真底"的溢价：入场后 6 个月内最低价
        fwd = px.iloc[i:i + 126]
        rec["6M内最大逆行"] = fwd.min() / p0 - 1
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(ev, label):
    r = {"信号": label, "入场次数": len(ev)}
    for h in HOR:
        r[f"{h}中位"] = ev[h].median()
        r[f"{h}胜率"] = (ev[h] > 0).mean()
    r["6M最大逆行中位"] = ev["6M内最大逆行"].median()
    return r


all_ev, summ = [], []
for nm, df, pxcol in (("沪金主连(他们的标的)", au, "close"),):
    o, h, l, c = df.open, df.high, df.low, df[pxcol]
    jax, tmp = jiaan(o, h, l, c, N)
    POS = pos_score(h, l, c)
    right = (POS < 20) & (tmp.diff() > 0) & (jax.diff() > 0)
    left = (POS < 20) & (tmp.diff() > 0) & (jax.diff() < 0)
    ma200 = c.rolling(200).mean()
    mine = c > ma200 * 1.01
    for sig, lab in ((right, "东吴式右侧"), (left, "东吴式左侧"), (mine, "我们 200DMA×1.01")):
        ev = study(c, entries(sig), f"{lab}@{nm}")
        if len(ev):
            all_ev.append(ev)
            summ.append(summarize(ev, f"{lab}@{nm}"))

# 同样规则跑伦敦金（我们真正决策的标的），OHLC 用 GLD 代理
gld = pd.read_csv("data/gld_ohlc.csv", parse_dates=["date"], index_col="date")
jax, tmp = jiaan(gld.open, gld.high, gld.low, gld.close, N)
POS = pos_score(gld.high, gld.low, gld.close)
right = (POS < 20) & (tmp.diff() > 0) & (jax.diff() > 0)
left = (POS < 20) & (tmp.diff() > 0) & (jax.diff() < 0)
ma200 = gld.close.rolling(200).mean()
mine = gld.close > ma200 * 1.01
for sig, lab in ((right, "东吴式右侧"), (left, "东吴式左侧"), (mine, "我们 200DMA×1.01")):
    ev = study(gld.close, entries(sig), f"{lab}@GLD")
    if len(ev):
        all_ev.append(ev)
        summ.append(summarize(ev, f"{lab}@GLD"))

ev_all = pd.concat(all_ev, ignore_index=True)
s = pd.DataFrame(summ)
disp = s.copy()
for col in disp.columns:
    if col not in ("信号", "入场次数"):
        disp[col] = (disp[col] * 100).round(1).astype(str) + "%"
print("\n=== B. 入场事件对比（出场用固定持有期，不用'后续最近局部高点'）===")
print(disp.to_string(index=False))

ev_all.to_csv(OUT / "t7_entries.csv", index=False)
s.to_csv(OUT / "t7_summary.csv", index=False)

# ---------- C. 当下状态 ----------
print("\n=== C. 当下读数 ===")
for nm, df in (("沪金主连", au), ("GLD", gld)):
    o, h, l, c = df.open, df.high, df.low, df.close
    jax, tmp = jiaan(o, h, l, c, N)
    POS = pos_score(h, l, c)
    d = c.index[-1].date()
    print(f"{nm} @{d}: 位置分={POS.iloc[-1]:.1f} "
          f"TMP{'↑' if tmp.diff().iloc[-1]>0 else '↓'} "
          f"JAX{'↑' if jax.diff().iloc[-1]>0 else '↓'} → "
          f"右侧{'✓' if POS.iloc[-1]<20 and tmp.diff().iloc[-1]>0 and jax.diff().iloc[-1]>0 else '✗'}")

print("\n输出: output/t7_entries.csv, output/t7_summary.csv")
