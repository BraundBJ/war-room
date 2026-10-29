"""Extend projections to 2030 and 2031 with historical aging curves (rebuilders care about them).

From every pair of consecutive seasons 2012-2025, by position and age (age at Sept 1 of season t):
  s(a) = P(healthy in t+1 | healthy in t)          healthy = 8+ games
  u(a) = P(healthy in t+1 | 1-7 games in t)
  r(a) = mean PPG(t+1) / PPG(t), healthy both years (ratio clipped to 0.3-2.0)
Pooled over +-1 year of age (wider where thin), and from age 27 on forced to hold or fall.
Then, stepping from the 2029 projection (h3) at the player's 2029 age:
  P(healthy h+1) = P(healthy h) s(a) + (1 - P(healthy h)) u(a)
  PPG(h+1) = PPG(h) r(a), spread widened 5% a year.

Usage: python3 -I extend_horizons.py FEATURES PLAYERS_CSV PROJECTIONS_CSV
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from wrconf import ROOT, SEASON, WEEK, REST, LATE_MIN, BACK_BY, LAST_WEEK, PLAYOFF  # noqa: E402
import sys
import numpy as np
import pandas as pd

FEAT, PLAYERS, PROJ = sys.argv[1:4]
Z90 = 1.2816
T = pd.read_parquet(FEAT)
T = T[T.season <= 2025]
pl = pd.read_csv(PLAYERS, usecols=["gsis_id", "birth_date"]).dropna().drop_duplicates("gsis_id")
bd = dict(zip(pl.gsis_id, pd.to_datetime(pl.birth_date, errors="coerce")))
T["age"] = [((pd.Timestamp(f"{s}-09-01") - bd[p]).days / 365.25) if p in bd and pd.notna(bd[p]) else np.nan
            for p, s in zip(T.player_id, T.season)]
nxt = T.set_index(["player_id", "season"])[["full_games", "full_ppg"]]
idx = pd.MultiIndex.from_arrays([T.player_id, T.season + 1])
n1 = nxt.reindex(idx)
T["g1"] = n1.full_games.fillna(0).values
T["ppg1"] = n1.full_ppg.values
T = T[(T.season <= 2024) & T.age.notna()]
T["a"] = T.age.astype(int).clip(21, 40)
T["h0"] = T.full_games >= 8
T["h1"] = T.g1 >= 8
T["ratio"] = (T.ppg1 / T.full_ppg).clip(0.3, 2.0)

curves = {}
for pos, g in T.groupby("pos"):
    rows = {}
    for a in range(21, 41):
        for wdt in (1, 2, 3):
            w = g[(g.a - a).abs() <= wdt]
            hh, hn = w[w.h0], w[~w.h0]
            if len(hh) >= 25 or wdt == 3:
                break
        both = hh[hh.h1]
        rows[a] = {"s": hh.h1.mean() if len(hh) else np.nan, "u": hn.h1.mean() if len(hn) else np.nan,
                   "r": both.ratio.mean() if len(both) else np.nan, "n": len(hh)}
    c = pd.DataFrame(rows).T.astype(float).ffill().bfill()
    for k in ("s", "u", "r"):  # from 27 on: hold or fall
        tail = c.loc[27:, k].cummin()
        c.loc[27:, k] = tail
    curves[pos] = c
    print(pos, c.loc[[22, 25, 27, 29, 31, 33, 35]].round(2).to_string(), sep="\n")

P = pd.read_csv(PROJ)
for h in (4, 5):
    a = (P.age + h - 1).round().astype(int).clip(21, 40)
    s = np.array([curves[p].at[x, "s"] for p, x in zip(P.pos, a)])
    u = np.array([curves[p].at[x, "u"] for p, x in zip(P.pos, a)])
    r = np.array([curves[p].at[x, "r"] for p, x in zip(P.pos, a)])
    ph = P[f"phealthy{h - 1}"]
    P[f"phealthy{h}"] = ph * s + (1 - ph) * u
    P[f"ppg{h}"] = P[f"ppg{h - 1}"] * r
    sd = (P[f"ppg{h - 1}_p90"] - P[f"ppg{h - 1}_p10"]) / (2 * Z90) * 1.05
    P[f"ppg{h}_p10"] = P[f"ppg{h}"] - Z90 * sd
    P[f"ppg{h}_p90"] = P[f"ppg{h}"] + Z90 * sd
    P[f"games{h}"] = P["games3"]
P.to_csv(PROJ, index=False)
show = P[P.name.isin(["George Kittle", "Travis Kelce", "Derrick Henry", "Ashton Jeanty", "Jeremiyah Love", "Puka Nacua"])]
print(show[["name", "age", "ppg3", "ppg4", "ppg5", "phealthy3", "phealthy4", "phealthy5"]].round(2).to_string(index=False))
