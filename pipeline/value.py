"""Actual value for every projected 2026 player in Brady's league format.

Replacement levels come from the 2025 season (healthy players, 8+ games), filling 12 lineups of
1QB / 2RB / 2WR / 1TE / 3FLEX greedily by PPG; replacement = best non-starter at each position.
Held constant across horizons (the v3.5 fix: re-solving per year inflates long-horizon value).

Season value  = P(healthy season) x max(0, projected PPG - replacement) x games if healthy
Contender     = 1.0 x rest of 2026 + 0.5 x 2027 + 0.25 x 2028 + 0.1 x 2029
Rebuilder     = 0 x rest of 2026 + 0.3 x 2027 + 0.8 x 2028 + 1.0 x 2029 + 1.0 x 2030 + 0.8 x 2031
(2030-2031 come from extend_horizons.py aging curves)
Low / high    = the same with each horizon's 10th / 90th percentile PPG.

Usage: python3 -I value.py PROJECTIONS FEATURES IDMAP ROSTERS USERS OUT_CSV
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from wrconf import ROOT, SEASON, WEEK, REST, LATE_MIN, BACK_BY, LAST_WEEK, PLAYOFF  # noqa: E402
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

PROJ, FEAT, IDMAP, ROSTERS, USERS, OUT = sys.argv[1:7]
TEAMS = 12
SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
FLEX = 3
W_CONTEND = {0: 1.0, 1: 0.5, 2: 0.25, 3: 0.1, 4: 0.0, 5: 0.0}
W_REBUILD = {0: 0.0, 1: 0.3, 2: 0.8, 3: 1.0, 4: 1.0, 5: 0.8}
# Same total weight for both modes (in full-season units; the rest of 2026 is 10 of 17 weeks), so a
# value means points per season of that team's window and contender vs rebuilder only reflects WHEN
# a player produces, not how many seasons are being added up.
_SEASON = {0: REST / 17, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
for _W in (W_CONTEND, W_REBUILD):
    _tot = sum(_W[h] * _SEASON[h] for h in _W)
    for h in _W:
        _W[h] = _W[h] / _tot

T = pd.read_parquet(FEAT)
pool = T[(T.season == 2025) & (T.full_games >= 8)][["player_id", "pos", "full_ppg"]].sort_values("full_ppg", ascending=False)
start = set()
for pos, n in SLOTS.items():
    start |= set(pool[pool.pos == pos].head(n * TEAMS).player_id)
flex = pool[~pool.player_id.isin(start) & pool.pos.isin(["RB", "WR", "TE"])].head(FLEX * TEAMS)
start |= set(flex.player_id)
repl = {pos: float(pool[(pool.pos == pos) & ~pool.player_id.isin(start)].full_ppg.iloc[0]) for pos in SLOTS}
starters = {pos: int(pool[(pool.pos == pos) & pool.player_id.isin(start)].shape[0]) for pos in SLOTS}
print("replacement PPG", {k: round(v, 2) for k, v in repl.items()}, "starters", starters)

P = pd.read_csv(PROJ)
# rest-of-2026 availability from the return-timing model: a healthy rest of season needs 4+ games,
# so he must be back by week 14
import os
_io = json.load(open(os.path.join(os.path.dirname(PROJ), "injury_outlook.json")))
_ids = pd.read_csv(IDMAP, low_memory=False).dropna(subset=["sleeper_id", "gsis_id"])
_g2s = dict(zip(_ids.gsis_id, _ids.sleeper_id.astype(int).astype(str)))
def _cap(row):
    st = _io["status"].get(_g2s.get(row.player_id))
    if not st or st["kind"] == "q":
        return row.phealthy0
    return min(row.phealthy0, st["back"].get(str(BACK_BY), 0.0))
P["phealthy0"] = P.apply(_cap, axis=1)
r = P.pos.map(repl)
Z90 = 1.2816  # normal 90th percentile
for h in range(6):
    # spread of a healthy season's PPG, from the calibrated 10th-90th percentile band
    P[f"sd{h}"] = ((P[f"ppg{h}_p90"] - P[f"ppg{h}_p10"]) / (2 * Z90)).clip(lower=0.5)
    # expected points over replacement counting upside: E[max(0, PPG - repl)] for normal PPG.
    # A bad season costs nothing beyond being benched; a breakout counts in full.
    z = (P[f"ppg{h}"] - r) / P[f"sd{h}"]
    pos_part = P[f"sd{h}"] * norm.pdf(z) + (P[f"ppg{h}"] - r) * norm.cdf(z)
    P[f"v{h}"] = P[f"phealthy{h}"] * pos_part * P[f"games{h}"]
    P[f"v{h}_mean"] = P[f"phealthy{h}"] * (P[f"ppg{h}"] - r).clip(lower=0) * P[f"games{h}"]
    for tag, col in (("_lo", f"ppg{h}_p10"), ("_hi", f"ppg{h}_p90")):
        P[f"v{h}{tag}"] = P[f"phealthy{h}"] * (P[col] - r).clip(lower=0) * P[f"games{h}"]
for name, wts in (("contender", W_CONTEND), ("rebuilder", W_REBUILD)):
    for tag in ("", "_lo", "_hi", "_mean"):
        P[f"{name}{tag}"] = sum(wts[h] * P[f"v{h}{tag}"] for h in wts)

ids = pd.read_csv(IDMAP, low_memory=False).dropna(subset=["sleeper_id", "gsis_id"])
ids["sleeper_id"] = ids.sleeper_id.astype(int).astype(str)
P["sleeper_id"] = P.player_id.map(dict(zip(ids.gsis_id, ids.sleeper_id)))
users = {u["user_id"]: u.get("display_name") for u in json.load(open(USERS))}
owner = {pid: users.get(ro["owner_id"]) for ro in json.load(open(ROSTERS)) for pid in (ro.get("players") or [])}
P["owner"] = P.sleeper_id.map(owner).fillna("Free agent")
P["contender_rank"] = P.contender.rank(ascending=False, method="min").astype(int)
P["rebuilder_rank"] = P.rebuilder.rank(ascending=False, method="min").astype(int)
P.round(3).sort_values("contender", ascending=False).to_csv(OUT, index=False)
json.dump({"replacement_ppg": repl, "starters": starters, "w_contender": W_CONTEND, "w_rebuilder": W_REBUILD},
          open(OUT.replace(".csv", "_settings.json"), "w"), indent=1)

pd.set_option("display.width", 230)
cols = ["name", "pos", "age", "owner", "ppg0", "ppg1", "ppg2", "phealthy1", "contender", "contender_lo",
        "contender_hi", "rebuilder", "contender_rank", "rebuilder_rank"]
print("\nTOP 25 CONTENDER\n", P.sort_values("contender", ascending=False).head(25)[cols].round(1).to_string(index=False))
print("\nTOP 25 REBUILDER\n", P.sort_values("rebuilder", ascending=False).head(25)[cols].round(1).to_string(index=False))
print("\nBRADY\n", P[P.owner == "BraundJB"].sort_values("contender", ascending=False)[cols].round(1).to_string(index=False))
