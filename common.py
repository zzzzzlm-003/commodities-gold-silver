"""Shared helpers: data loading, backtest engine, metrics."""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

COST = 0.0005  # 5bp one-way


def load(name, col=None):
    df = pd.read_csv(DATA / f"{name}.csv", parse_dates=["date"], index_col="date")
    if col is None:
        col = "adjclose" if "adjclose" in df.columns else "close"
    return df[col].rename(name)


def metrics(equity, pos=None, label=""):
    """equity: daily equity curve (pd.Series). pos: invested fraction series."""
    r = equity.pct_change().dropna()
    yrs = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    dd = (equity / equity.cummax() - 1).min()
    calmar = cagr / abs(dd) if dd < 0 else np.nan
    tim = pos.mean() if pos is not None else 1.0
    return {"策略": label, "CAGR": cagr, "年化波动": vol, "最大回撤": dd,
            "Calmar": calmar, "在场比例": tim}


def run_filtered(px, sig, exec_lag=1):
    """Long/flat backtest. sig: 0/1 computed at close t; executed at close t+exec_lag,
    so exposure applies to the day AFTER execution (pos shifted exec_lag+1)."""
    pos = sig.shift(exec_lag + 1).reindex(px.index).fillna(0.0)
    r = px.pct_change().fillna(0.0)
    trade = pos.diff().abs().fillna(pos.iloc[0] if len(pos) else 0)
    ret = pos * r - trade * COST
    eq = (1 + ret).cumprod()
    n_trades = int((pos.diff().abs() > 0).sum())
    return eq, pos, n_trades


def whipsaws(px, pos):
    """Round trips (entry->exit) with net loss including 2-way cost."""
    changes = pos.diff().fillna(0)
    entries = px.index[changes > 0]
    exits = px.index[changes < 0]
    trips = []
    for e in entries:
        ex = exits[exits > e]
        if len(ex) == 0:
            break
        x = ex[0]
        ret = px.loc[x] / px.loc[e] - 1 - 2 * COST
        trips.append({"entry": e, "exit": x, "ret": ret,
                      "days": (x - e).days})
    trips = pd.DataFrame(trips)
    if trips.empty:
        return trips, 0, 0.0
    ws = trips[trips.ret < 0]
    return trips, len(ws), ws.ret.sum()


def run_ladder(px, rungs, exec_lag=1):
    """Left-side ladder: 25% of initial capital per rung below running ATH.
    Signal at close t, executed at close t+exec_lag. Buy-only. Cash earns 0."""
    r = px.pct_change().fillna(0.0)
    ath = px.cummax()
    dd = px / ath - 1
    # target invested fraction signal at each close
    frac = pd.Series(0.0, index=px.index)
    deployed = 0.0
    armed = [True] * len(rungs)  # re-arm on new ATH only if cash remains
    prev_ath = -np.inf
    for i, (d, p) in enumerate(px.items()):
        if ath.iloc[i] > prev_ath and deployed < 1.0:
            armed = [True] * len(rungs)
            prev_ath = ath.iloc[i]
        for j, rung in enumerate(rungs):
            if armed[j] and dd.iloc[i] <= -rung and deployed < 0.999:
                deployed = min(1.0, deployed + 0.25)
                armed[j] = False
        frac.iloc[i] = deployed
    # execute at close t+exec_lag: exact unit accounting (spend fraction of INITIAL capital)
    target = frac.shift(exec_lag).fillna(0.0)
    cash, units = 1.0, 0.0
    eq = pd.Series(1.0, index=px.index)
    invested = pd.Series(0.0, index=px.index)
    prev_t = 0.0
    n_trades = 0
    for i, (d, p) in enumerate(px.items()):
        buy = target.iloc[i] - prev_t
        if buy > 1e-9 and cash > 1e-9:
            spend = min(buy, cash)  # fraction of initial capital (=1)
            units += spend * (1 - COST) / p
            cash -= spend
            n_trades += 1
        prev_t = max(prev_t, target.iloc[i])
        eq.iloc[i] = cash + units * p
        invested.iloc[i] = units * p / eq.iloc[i]
    return eq, invested, n_trades

import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
