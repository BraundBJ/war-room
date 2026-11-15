"""Projected fantasy-playoff production (weeks 15-17) for every player.

For each playoff week: his healthy rest-of-season PPG, scaled by that week's opponent (points the defense
allows to his position vs league average, raised to a position elasticity measured on 2016-2025
player-weeks: QB 0.37, RB 0.47, WR 0.27, TE 0.21), times his chance of being available that week (the
model's per-game availability times, for injured players, the chance he is back by then).

Writes proj/playoff_proj.json: sleeper id -> {"ppg": expected points per playoff week, "h": healthy PPG before matchups,
"ifp": PPG if he plays, "av": average availability, "wk": [[week, opp, home, factor, availability], ...]}

Usage: python3 -I playoff_proj.py PROJ_DIR GAMES_CSV SLEEPER_PLAYERS_JSON
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from wrconf import SEASON, WEEK, REST  # noqa: E402
import json
import sys
import numpy as np
import pandas as pd

PROJ, GAMES, SPJ = sys.argv[1:4]
PO_WEEKS = [15, 16, 17]
ELAST = {"QB": 0.37, "RB": 0.47, "WR": 0.27, "TE": 0.21}
TEAMFIX = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS"}

V = pd.read_csv(f"{PROJ}/actual_values.csv", dtype={"sleeper_id": str})
V = V[V.sleeper_id.notna()].drop_duplicates("sleeper_id")
raw = pd.read_csv(f"{PROJ}/projections_2026.csv")[["player_id", "phealthy0", "games0", "ppg0"]].rename(
    columns={"phealthy0": "ph_m", "games0": "g_m", "ppg0": "ppg_m"})
V = V.merge(raw, on="player_id", how="left")
io = json.load(open(f"{PROJ}/injury_outlook.json"))["status"]
allow = pd.read_parquet(f"{PROJ}/week_allow.parquet") if _os.path.exists(f"{PROJ}/week_allow.parquet") else pd.DataFrame(columns=["opp", "pos", "opp_allow_rel"])
allow = {(o, p): r for o, p, r in allow[["opp", "pos", "opp_allow_rel"]].itertuples(index=False)}
G = pd.read_csv(GAMES)
G = G[(G.season == SEASON) & G.week.isin(PO_WEEKS) & (G.game_type == "REG")]
sched = {}
for g in G.itertuples():
    sched[(g.home_team, g.week)] = (g.away_team, 1)
    sched[(g.away_team, g.week)] = (g.home_team, 0)
SP = json.load(open(SPJ))

out = {}
for r in V.itertuples():
    team = TEAMFIX.get((SP.get(r.sleeper_id) or {}).get("team") or "", (SP.get(r.sleeper_id) or {}).get("team"))
    if not team or pd.isna(r.ppg_m):
        continue
    # per-game availability before any current injury (the model's), then the injury's return timing
    base = float(np.clip((r.ph_m if pd.notna(r.ph_m) else 0.9) * (r.g_m if pd.notna(r.g_m) else REST) / max(REST, 1), 0, 1))
    st = io.get(r.sleeper_id)
    wk, tot, fsum, n = [], 0.0, 0.0, 0
    for w in PO_WEEKS:
        if (team, w) not in sched:
            wk.append([w, None, None, 0.0, 0.0]); continue
        opp, home = sched[(team, w)]
        rel = allow.get((opp, r.pos), 1.0)
        f = float(rel ** ELAST.get(r.pos, 0.3))
        back = 1.0
        if st and st.get("kind") != "q":
            back = float(st["back"].get(str(w), 1.0))
        a = base * back
        tot += a * r.ppg_m * f; fsum += f; n += 1
        wk.append([w, opp, home, round(f, 3), round(a, 3)])
    if not n:
        continue
    out[r.sleeper_id] = {"ppg": round(tot / len(PO_WEEKS), 2), "h": round(float(r.ppg_m), 2), "ifp": round(float(r.ppg_m) * fsum / n, 2),
                         "av": round(float(np.mean([x[4] for x in wk])), 3), "wk": wk}
json.dump({"weeks": PO_WEEKS, "players": out}, open(f"{PROJ}/playoff_proj.json", "w"), separators=(",", ":"))
top = sorted(out.items(), key=lambda kv: -kv[1]["ppg"])[:10]
print(len(out), "players;", [(SP.get(k, {}).get("full_name"), v["ppg"], [x[1] for x in v["wk"]]) for k, v in top])
