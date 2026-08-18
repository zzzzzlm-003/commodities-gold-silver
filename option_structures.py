"""Task 8: 期权结构 —— 「永不卖 call」前提下的候选结构定价、情景表与按月滚动模拟。

口径（沿用报告全文约定，补充期权特有项）：
- 报价：CBOE delayed quotes 快照，存 data/opt_{SYM}_{YYYYMMDD}.json 以便复现。
  基准成交 = mid；同时报最差成交（买方付 ask、卖方拿 bid）。
- 无风险利率：不假设，从 put-call parity 反解（GLD/SLV 无分红；GDX 有分红，
  反解出的是 r-q，用于定价正好一致）。取合规腿的中位数。
- 换算：金→ETF 用当日实测比率。GDX/SLV 对金价的敏感度用 beta（Yahoo 5 年日收益
  回归 GLD）线性化，**仅用于情景表**，不是精确定价。
- 流动性门槛：OI >= 100 且 相对价差 <= 10%；不达标的腿标 illiq 且不进结论。
- 历史左尾：LBMA 现货，H = round(DTE*252/365) 交易日前瞻，**条件在「现价<200DMA」**
  （= 今天的状态）。ETF 所需跌幅经 beta 折算回金价幅度后取频率。
- 滚动模拟：50% 止盈 / 到期前 7 天展期 / 否则持到期；路径重定价用入场 IV 不变
  （不建模 IV 路径，这会低估亏损周期的幅度，是已知的乐观偏差）。
"""
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from common import DATA, OUT, load

SYMS = ["GLD", "GDX", "SLV"]
CBOE = "https://cdn.cboe.com/api/global/delayed_quotes/options/{}.json"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{}?range=5y&interval=1d"
UA = "Mozilla/5.0"
MIN_OI, MAX_REL_SPREAD = 100, 0.10
# 情景档：左侧档 3675 / 本轮低点 3994 附近 / 现价 / 右侧触发 4535 / 5000 / 前高 5405
GOLD_SCENARIOS = [3675, 4000, 4200, 4406, 4535, 4800, 5000, 5405]
TRIGGER_GOLD = 4535        # 200DMA x 1.01，清单规则 #1
BUDGET = 2000.0            # 对照表用的统一投入资金（美元）
NAV = 40000.0              # 总资产，来自 portfolio_review.md（≈$40k），改这里即可
TODAY = date.today().strftime("%Y%m%d")


# ---------------------------------------------------------------- 数据

def fetch_json(url, path):
    """下载并缓存当日快照；已存在就直接读，保证结果可复现。"""
    if path.exists():
        return json.load(open(path))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    raw = urllib.request.urlopen(req, timeout=60).read()
    if len(raw) < 5000:
        raise SystemExit(f"payload too small ({len(raw)}B) for {url}")
    path.write_bytes(raw)
    return json.loads(raw)


def refresh_lbma():
    for name in ("gold_pm", "silver"):
        subprocess.run(["curl", "-s", "-A", UA,
                        f"https://prices.lbma.org.uk/json/{name}.json",
                        "-o", str(DATA / f"{name}.json")], check=True)
    subprocess.run([sys.executable, str(Path(__file__).parent / "prep_data.py")],
                   check=True)


def load_chain(sym):
    d = fetch_json(CBOE.format(sym), DATA / f"opt_{sym}_{TODAY}.json")["data"]
    rows = []
    for o in d["options"]:
        tail = o["option"][len(sym):] if o["option"].startswith(sym) else ""
        if len(tail) < 8 or not tail[:6].isdigit() or tail[6] not in "CP":
            continue
        bid, ask = o["bid"], o["ask"]
        mid = (bid + ask) / 2
        rows.append({
            "exp": datetime.strptime(tail[:6], "%y%m%d").date(),
            "cp": tail[6], "strike": int(tail[7:]) / 1000,
            "bid": bid, "ask": ask, "mid": mid, "iv": o["iv"],
            "delta": o["delta"], "theta": o["theta"], "vega": o["vega"],
            "oi": o["open_interest"] or 0, "vol": o["volume"] or 0,
            "rel_spread": (ask - bid) / mid if mid > 0 else np.inf,
        })
    df = pd.DataFrame(rows)
    df["dte"] = [(e - date.today()).days for e in df.exp]
    df["illiq"] = (df.oi < MIN_OI) | (df.rel_spread > MAX_REL_SPREAD)
    return d["current_price"], df, d.get("iv30")


def imply_rate(df, spot):
    """put-call parity: K*exp(-rT) = S - C + P。取中位数。"""
    rs = []
    for (exp, k), g in df[~df.illiq].groupby(["exp", "strike"]):
        if set(g.cp) != {"C", "P"}:
            continue
        T = (exp - date.today()).days / 365
        pv = spot - g[g.cp == "C"].mid.iloc[0] + g[g.cp == "P"].mid.iloc[0]
        if T > 0.05 and 0 < pv < k * 1.5:
            rs.append(-np.log(pv / k) / T)
    return float(np.median(rs)) if rs else 0.04


def yahoo_close(sym):
    r = fetch_json(YAHOO.format(sym), DATA / f"opt_yh_{sym}_{TODAY}.json")
    r = r["chart"]["result"][0]
    ts = pd.to_datetime(r["timestamp"], unit="s", utc=True)
    ts = ts.tz_convert("America/New_York").normalize().tz_localize(None)
    s = pd.Series(r["indicators"]["adjclose"][0]["adjclose"], index=ts,
                  name=sym).dropna()
    return s[~s.index.duplicated()]


def compute_betas():
    """GDX/SLV 对 GLD 的日收益 beta（5 年）；GLD 自己 = 1。"""
    px = pd.concat([yahoo_close(s) for s in SYMS], axis=1).dropna()
    r = np.log(px).diff().dropna()
    b = {"GLD": 1.0}
    for s in ("GDX", "SLV"):
        b[s] = float(np.cov(r[s], r.GLD)[0, 1] / np.var(r.GLD))
    return b, len(r)


# ---------------------------------------------------------------- 结构

def pick_delta(df, exp, cp, target, tol=0.08):
    """取 |delta| 最接近 target 的合规腿；偏离超 tol 就返回 None（不塌到最近一档）。"""
    g = df[(df.exp == exp) & (df.cp == cp) & ~df.illiq & (df.bid > 0)]
    if g.empty:
        return None
    leg = g.loc[(g.delta.abs() - target).abs().idxmin()]
    return leg if abs(abs(leg.delta) - target) <= tol else None


def pick_strike(df, exp, cp, strike, tol_pct=0.02):
    """取行权价最接近的合规腿；偏离超现价 tol_pct 就返回 None。"""
    g = df[(df.exp == exp) & (df.cp == cp) & ~df.illiq & (df.bid > 0)]
    if g.empty:
        return None
    leg = g.loc[(g.strike - strike).abs().idxmin()]
    return leg if abs(leg.strike - strike) <= strike * tol_pct else None


def entry_cost(legs, worst=False):
    """入场净现金流（正 = 净支出，负 = 净收权利金），美元/组。"""
    tot = 0.0
    for qty, leg in legs:
        px = (leg.ask if qty > 0 else leg.bid) if worst else leg.mid
        tot += qty * px
    return tot * 100


def payoff(legs, U, cost):
    U = np.asarray(U, dtype=float)
    v = np.zeros_like(U)
    for qty, leg in legs:
        intr = (np.maximum(U - leg.strike, 0) if leg.cp == "C"
                else np.maximum(leg.strike - U, 0))
        v += qty * intr * 100
    return v - cost


def profile(legs, cost, spot):
    grid = np.linspace(0.01, spot * 3, 8000)
    pl = payoff(legs, grid, cost)
    capped = sum(q for q, l in legs if l.cp == "C") <= 0
    bes = [round(float(grid[i]), 2) for i in range(1, len(grid))
           if np.sign(pl[i]) != np.sign(pl[i - 1])]
    short_calls = [l.strike for q, l in legs if q < 0 and l.cp == "C"]
    return {"max_loss": float(pl.min()),
            "max_gain": float(pl.max()) if capped else np.inf,
            "upside_capped": capped,
            "cap_strike": max(short_calls) if (capped and short_calls) else np.nan,
            "short_leg": any(q < 0 for q, _ in legs),
            "breakevens": bes}


def build(sym, df, spot, beta, gold_last):
    """候选结构。A/B/C/E 走近月两个周期（对应按月轮换）；D 另外覆盖远月。"""
    out = []
    exps = sorted(df[~df.illiq].exp.unique())
    dte = {e: (e - date.today()).days for e in exps}
    near = [e for e in exps if 22 <= dte[e] <= 45][:1]
    nxt = [e for e in exps if 46 <= dte[e] <= 75][:1]
    far = [e for e in exps if 100 <= dte[e] <= 165][:2]
    cycles = near + nxt or exps[:1]

    for exp in cycles:
        for tgt in (0.10, 0.20, 0.30):          # A. put credit spread 三档
            s = pick_delta(df, exp, "P", tgt)
            if s is None:
                continue
            l = pick_strike(df, exp, "P", s.strike - spot * 0.025)
            if l is None or l.strike >= s.strike:
                continue
            out.append((f"A put价差 Δ{tgt:.2f}", exp, [(-1, s), (1, l)]))

        sp = pick_delta(df, exp, "P", 0.20)
        lp = (pick_strike(df, exp, "P", sp.strike - spot * 0.025)
              if sp is not None else None)
        lc = pick_delta(df, exp, "C", 0.35)
        sc = pick_delta(df, exp, "C", 0.15)
        have = all(x is not None for x in (sp, lp, lc, sc))
        if have and sc.strike > lc.strike and lp.strike < sp.strike:
            out.append(("B1 RR定义风险 Δ0.35/0.15", exp,
                        [(-1, sp), (1, lp), (1, lc), (-1, sc)]))
        # B2：封顶位挂在她自己的目标（金 5405 = 前高），封顶就不咬目标
        lc2 = pick_strike(df, exp, "C", spot * (1 + beta * (TRIGGER_GOLD / gold_last - 1)))
        sc2 = pick_strike(df, exp, "C", spot * (1 + beta * (5405 / gold_last - 1)), 0.04)
        if all(x is not None for x in (sp, lp, lc2, sc2)) and sc2.strike > lc2.strike:
            out.append(("B2 RR定义风险 封顶挂金5405", exp,
                        [(-1, sp), (1, lp), (1, lc2), (-1, sc2)]))
        if sp is not None and lc is not None:
            out.append(("C RR裸版 卖put买call", exp, [(-1, sp), (1, lc)]))
        # F iron condor：她提名的结构。算出来只为把"封顶位"和"能收多少"写实
        scall = pick_delta(df, exp, "C", 0.20)
        lcall = (pick_strike(df, exp, "C", scall.strike + spot * 0.025)
                 if scall is not None else None)
        if all(x is not None for x in (sp, lp, scall, lcall)) and lcall.strike > scall.strike:
            out.append(("F iron condor Δ0.20/0.20", exp,
                        [(-1, sp), (1, lp), (-1, scall), (1, lcall)]))
        s45, l22 = pick_delta(df, exp, "C", 0.45), pick_delta(df, exp, "C", 0.22)
        if s45 is not None and l22 is not None and l22.strike > s45.strike:
            out.append(("E call比率反向 1×2", exp, [(-1, s45), (2, l22)]))

        # B3 自融资 RR：put 价差的 credit 最多能买到多高的封顶位？
        if all(x is not None for x in (sp, lp, lc2)):
            credit = -entry_cost([(-1, sp), (1, lp)]) / 100
            cands = df[(df.exp == exp) & (df.cp == "C") & ~df.illiq
                       & (df.bid > 0) & (df.strike > lc2.strike)]
            best = None
            for _, sc3 in cands.sort_values("strike", ascending=False).iterrows():
                if lc2.mid - sc3.mid <= credit:
                    best = sc3
                    break
            if best is not None:
                out.append(("B3 RR自融资 净成本≈0", exp,
                            [(-1, sp), (1, lp), (1, lc2), (-1, best)]))

    for exp in cycles + far:                     # D. 单腿 long call
        for gk in (TRIGGER_GOLD, 5000, 5405):
            k = spot * (1 + beta * (gk / gold_last - 1))
            c = pick_strike(df, exp, "C", k)
            if c is not None:
                out.append((f"D 单腿call 金{gk}", exp, [(1, c)]))
    return out


# ---------------------------------------------------------------- 概率 / 定价

def bs_put(S, K, T, r, iv):
    if T <= 1e-9:
        return max(K - S, 0.0)
    sd = iv * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / sd
    return K * np.exp(-r * T) * norm.cdf(sd - d1) - S * norm.cdf(-d1)


def p_itm_put(S, K, T, r, iv):
    """风险中性 P(S_T < K) = N(-d2)。"""
    d2 = (np.log(S / K) + (r - 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
    return float(norm.cdf(-d2))


def hist_tail(xau, H, move):
    """条件在「现价 < 200DMA」时，未来 H 交易日涨跌 <= move 的历史频率。"""
    px, ma = xau.to_numpy(), xau.rolling(200).mean().to_numpy()
    fwd = px[H:] / px[:-H] - 1
    mask = (~np.isnan(ma[:-H])) & (px[:-H] < ma[:-H])
    sub = fwd[mask]
    return float((sub <= move).mean()), int(mask.sum())


# ---------------------------------------------------------------- 滚动模拟

def roll_sim(xau, spot, sl, ll, credit, r, beta, dte, tp=0.50, roll_dte=7):
    """按月滚动卖 put 价差。返回每周期盈亏（美元/组）+ 退出方式 + 起点位置。"""
    px, ma = xau.to_numpy(), xau.rolling(200).mean().to_numpy()
    H = max(2, round(dte * 252 / 365))
    roll_at = max(1, round(roll_dte * 252 / 365))
    ks, kl, ivs, ivl = sl.strike, ll.strike, sl.iv, ll.iv
    rows = []
    for i in range(200, len(px) - H):
        if np.isnan(ma[i]) or px[i] >= ma[i]:
            continue
        pl, how = None, None
        for j in range(1, H + 1):
            U = spot * (1 + beta * (px[i + j] / px[i] - 1))
            T = (H - j) / 252
            val = max(bs_put(U, ks, T, r, ivs) - bs_put(U, kl, T, r, ivl), 0.0) * 100
            if val <= credit * (1 - tp):
                # 实际是挂 GTC 限价单在 50%，只能成交在 50%；按天检查会跳穿阈值，
                # 用 credit-val 会高估（实测 $81 vs $66.75），所以按限价成交价记。
                pl, how = credit * tp, "止盈"
                break
            if (H - j) <= roll_at:
                pl, how = credit - val, "展期"
                break
        if pl is None:
            U = spot * (1 + beta * (px[i + H] / px[i] - 1))
            intr = (max(ks - U, 0) - max(kl - U, 0)) * 100
            pl, how = credit - intr, "到期"
        rows.append({"i": i, "start": xau.index[i], "pl": pl, "exit": how})
    return pd.DataFrame(rows), H


def chain_year(sim, H, days=252):
    """串成约 days 个交易日（≈一年）的不重叠周期链 → 「滚一年」分布。

    注意：要求链上每一期起点都满足「现价<200DMA」，所以这个分布条件在
    「整年都待在 200DMA 下方」——正是她现在的处境，也是偏悲观的子样本。
    """
    n = max(1, round(days / H))
    by_pos = dict(zip(sim.i, sim.pl))
    tot = []
    for i in sim.i:
        seq = [by_pos.get(i + k * H) for k in range(n)]
        if all(v is not None for v in seq):
            tot.append(sum(seq))
    return np.array(tot), n


# ---------------------------------------------------------------- 主流程

if __name__ == "__main__":
    if "--refresh" in sys.argv:
        refresh_lbma()

    xau = load("xau")
    gold_last, gold_asof = float(xau.iloc[-1]), xau.index[-1].date()
    dma200 = float(xau.rolling(200).mean().iloc[-1])
    BETA, beta_n = compute_betas()

    print(f"LBMA 金 {gold_last:.1f}（截至 {gold_asof}），200DMA {dma200:.0f}，"
          f"触发门槛 {dma200 * 1.01:.0f}，ATH {xau.max():.0f}")
    print(f"beta vs GLD（5y 日收益 n={beta_n}）：" +
          "  ".join(f"{k}={v:.2f}" for k, v in BETA.items()))

    chains, spots, rates = {}, {}, {}
    for sym in SYMS:
        spot, df, iv30 = load_chain(sym)
        chains[sym], spots[sym], rates[sym] = df, spot, imply_rate(df, spot)
        print(f"{sym} spot {spot:.2f}  iv30 {iv30}  腿 {len(df)}"
              f"（合规 {int((~df.illiq).sum())}）  r(parity) {rates[sym]:.2%}"
              f"  金→ETF 比率 {spot / gold_last:.5f}")

    summary, scen_rows, leg_rows, keep = [], [], [], {}
    for sym in SYMS:
        df, spot, r = chains[sym], spots[sym], rates[sym]
        for name, exp, legs in build(sym, df, spot, BETA[sym], gold_last):
            dte = (exp - date.today()).days
            c_mid, c_worst = entry_cost(legs), entry_cost(legs, worst=True)
            prof = profile(legs, c_mid, spot)
            H = max(2, round(dte * 252 / 365))
            edge, q_rn, q_hist = np.nan, np.nan, np.nan
            shorts = [l for q, l in legs if q < 0 and l.cp == "P"]
            if shorts:
                sp = shorts[0]
                q_rn = p_itm_put(spot, sp.strike, dte / 365, r, sp.iv)
                q_hist, _ = hist_tail(xau, H, (sp.strike / spot - 1) / BETA[sym])
                edge = q_rn / q_hist if q_hist > 0 else np.inf
            key = f"{sym}|{name}|{exp}"
            keep[key] = (legs, c_mid, spot, sym)
            summary.append({
                "标的": sym, "结构": name, "到期": exp, "DTE": dte,
                "净收权利金_mid": round(-c_mid, 2),
                "净收权利金_最差": round(-c_worst, 2),
                "最大亏": round(prof["max_loss"], 2),
                "最大赚": prof["max_gain"],
                "上行封顶": prof["upside_capped"],
                "封顶位_金价": (round(gold_last * (1 + (prof["cap_strike"] / spot - 1)
                                                   / BETA[sym]))
                             if prof["cap_strike"] == prof["cap_strike"] else np.nan),
                "最大亏占净资产": round(prof["max_loss"] / NAV, 4),
                "需空头腿": prof["short_leg"],
                "盈亏平衡": prof["breakevens"],
                "隐含P(ITM)": round(q_rn, 4) if q_rn == q_rn else np.nan,
                "历史P(<200DMA条件)": round(q_hist, 4) if q_hist == q_hist else np.nan,
                "卖方溢价比": round(edge, 2) if np.isfinite(edge) else np.nan,
            })
            for qty, leg in legs:
                leg_rows.append({"key": key, "qty": qty, "cp": leg.cp,
                                 "strike": leg.strike, "bid": leg.bid,
                                 "ask": leg.ask, "iv": round(leg.iv, 4),
                                 "delta": round(leg.delta, 4), "oi": leg.oi})
            sc = {"标的": sym, "结构": name, "到期": exp}
            for g in GOLD_SCENARIOS:
                U = spot * (1 + BETA[sym] * (g / gold_last - 1))
                sc[f"金{g}"] = round(float(payoff(legs, U, c_mid)), 2)
            scen_rows.append(sc)

    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "t8_structures.csv", index=False)
    pd.DataFrame(scen_rows).to_csv(OUT / "t8_scenarios.csv", index=False)
    pd.DataFrame(leg_rows).to_csv(OUT / "t8_legs.csv", index=False)
    print(f"\n=== 结构 {len(summary)} 个 ===")
    print(summary[["标的", "结构", "DTE", "净收权利金_mid", "最大亏", "上行封顶",
                   "隐含P(ITM)", "历史P(<200DMA条件)", "卖方溢价比"]].to_string(index=False))

    # ------------------------------------------------------ 按月滚动模拟
    roll_rows, roll_year = [], []
    todo = [(sym, t, d) for sym in ("GLD", "GDX", "SLV")
            for t in (0.10, 0.20, 0.30) for d in ("近月", "次月")]
    for sym, tgt, cyc in todo:
        cand = summary[(summary.标的 == sym)
                       & (summary.结构 == f"A put价差 Δ{tgt:.2f}")].sort_values("DTE")
        if cand.empty or (cyc == "次月" and len(cand) < 2):
            continue
        row = cand.iloc[0 if cyc == "近月" else 1]
        legs, c_mid, spot, _ = keep[f"{sym}|{row.结构}|{row.到期}"]
        sl = [l for q, l in legs if q < 0][0]
        ll = [l for q, l in legs if q > 0][0]
        credit = -c_mid
        sim, H = roll_sim(xau, spot, sl, ll, credit, rates[sym], BETA[sym], row.DTE)
        yr, n_cyc = chain_year(sim, H)
        pl = sim.pl.to_numpy()
        roll_rows.append({
            "标的": sym, "结构": row.结构, "DTE": row.DTE, "H(交易日)": H,
            "入场credit": round(credit, 2), "最大亏": row.最大亏,
            "周期数": len(sim), "胜率": round(float((pl > 0).mean()), 4),
            "均值": round(float(pl.mean()), 2), "中位": round(float(np.median(pl)), 2),
            "p5": round(float(np.percentile(pl, 5)), 2),
            "最差": round(float(pl.min()), 2),
            "止盈占比": round(float((sim.exit == "止盈").mean()), 4),
            "展期占比": round(float((sim.exit == "展期").mean()), 4),
            "到期占比": round(float((sim.exit == "到期").mean()), 4),
            "滚一年_期数": n_cyc,
            "滚一年_均值": round(float(yr.mean()), 2) if len(yr) else np.nan,
            "滚一年_中位": round(float(np.median(yr)), 2) if len(yr) else np.nan,
            "滚一年_p5": round(float(np.percentile(yr, 5)), 2) if len(yr) else np.nan,
            "滚一年_最差": round(float(yr.min()), 2) if len(yr) else np.nan,
            "滚一年_亏损年占比": round(float((yr < 0).mean()), 4) if len(yr) else np.nan,
            "滚一年_样本": len(yr),
        })
        roll_year.append(pd.DataFrame({"标的": sym, "结构": row.结构,
                                       "DTE": row.DTE, "滚一年盈亏": yr}))
    roll = pd.DataFrame(roll_rows)
    roll.to_csv(OUT / "t8_roll_sim.csv", index=False)
    if roll_year:
        pd.concat(roll_year).to_csv(OUT / "t8_roll_year.csv", index=False)
    print("\n=== 按月滚动模拟（LBMA 历史，只取「现价<200DMA」起点）===")
    print(roll.to_string(index=False))

    # ------------------------------------------------------ 与正股对照（同资金 $2000）
    spot_g = spots["GLD"]
    trig_px = spot_g * (1 + TRIGGER_GOLD / gold_last - 1)
    picks = [("GLD", "A put价差 Δ0.20", 24),
             ("GLD", "B3 RR自融资 净成本≈0", 24),
             ("GLD", "B1 RR定义风险 Δ0.35/0.15", 59),
             ("GLD", "B2 RR定义风险 封顶挂金5405", 59),
             ("GLD", "E call比率反向 1×2", 24),
             ("GLD", "D 单腿call 金4535", 122),
             ("GLD", "C RR裸版 卖put买call", 59)]
    cmp_rows = [
        {"方案": f"现在买 GLD 正股 @{spot_g:.2f}", "张/股数": round(BUDGET / spot_g, 3),
         **{f"金{g}": round(BUDGET / spot_g * (spot_g * g / gold_last - spot_g), 2)
            for g in GOLD_SCENARIOS}},
        {"方案": f"等触发后买 GLD 正股 @{trig_px:.2f}", "张/股数": round(BUDGET / trig_px, 3),
         **{f"金{g}": (round(BUDGET / trig_px * (spot_g * g / gold_last - trig_px), 2)
                      if g >= TRIGGER_GOLD else 0.0) for g in GOLD_SCENARIOS}},
    ]
    for sym, name, dte in picks:
        m = summary[(summary.标的 == sym) & (summary.结构 == name) & (summary.DTE == dte)]
        if m.empty:
            continue
        row = m.iloc[0]
        legs, c_mid, spot, _ = keep[f"{sym}|{name}|{row.到期}"]
        risk = abs(row.最大亏)
        n = int(BUDGET // risk) if risk > 0 else 0
        if n == 0:
            cmp_rows.append({"方案": f"{name} {dte}DTE —— 单张最大亏 ${risk:,.0f}"
                                     f" > 预算 ${BUDGET:,.0f}，开不了",
                             "张/股数": 0, **{f"金{g}": np.nan for g in GOLD_SCENARIOS}})
            continue
        cmp_rows.append({"方案": f"{name} {dte}DTE ×{n}张", "张/股数": n,
                         **{f"金{g}": round(n * float(payoff(
                             legs, spot * (1 + BETA[sym] * (g / gold_last - 1)), c_mid)), 2)
                            for g in GOLD_SCENARIOS}})
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(OUT / "t8_vs_spot.csv", index=False)
    print(f"\n=== 同资金 ${BUDGET:.0f} 对照（到期/年底金价情景，美元盈亏）===")
    print("注：「等触发后买」假设终值 >= 4535 才算触发过，是简化。")
    print(cmp_df.to_string(index=False))

    # ------------------------------------------------------ 损益图
    gg = np.linspace(3400, 5900, 500)
    curves = []
    for sym, name, dte in picks:
        m = summary[(summary.标的 == sym) & (summary.结构 == name) & (summary.DTE == dte)]
        if m.empty:
            continue
        row = m.iloc[0]
        legs, c_mid, spot, _ = keep[f"{sym}|{name}|{row.到期}"]
        U = spot * (1 + BETA[sym] * (gg / gold_last - 1))
        curves.append((f"{name} {dte}DTE", payoff(legs, U, c_mid)))

    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
    for ax, ylim, tag in ((axes[0], None, "全幅"),
                          (axes[1], (-2500, 2500), "放大：±$2500")):
        for lab, y in curves:
            ax.plot(gg, y, label=lab, lw=1.4)
        ax.axhline(0, color="k", lw=0.8)
        if ylim:
            ax.set_ylim(*ylim)
        for x, lab in ((gold_last, f"现价 {gold_last:.0f}"),
                       (TRIGGER_GOLD, f"右侧触发 {TRIGGER_GOLD}"),
                       (3675, "左侧档 3675"), (5405, "前高 5405")):
            ax.axvline(x, ls="--", lw=0.8, color="gray")
            ax.text(x, ax.get_ylim()[1], lab, rotation=90, va="top", fontsize=8)
        ax.set_ylabel("每组盈亏 (USD)")
        ax.set_title(f"{tag}", fontsize=10, loc="left")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].set_xlabel("到期时 LBMA 金价 (USD/oz)")
    fig.suptitle(f"GLD 期权结构到期损益（每结构 1 张；快照 {gold_asof}，金 {gold_last:.0f}）")
    fig.tight_layout()
    fig.savefig(OUT / "t8_payoff.png", dpi=130)
    print("\n图 → output/t8_payoff.png")
