"""Bundle model outputs into one JSON blob for the league page.

Usage: python3 -I build_site_data.py OUT_JSON
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from wrconf import ROOT, SEASON, WEEK, REST, LATE_MIN, BACK_BY, LAST_WEEK, PLAYOFF  # noqa: E402
import datetime
import glob
import json
import sys
import numpy as np
import pandas as pd

OUT = sys.argv[1]
LG = ROOT + "/league/raw"
V = pd.read_csv(ROOT + "/stats/proj/actual_values.csv", dtype={"sleeper_id": str})
V = V[V.sleeper_id.notna()].drop_duplicates("sleeper_id").reset_index(drop=True)
S = pd.read_csv(ROOT + "/stats/results/scores_2026.csv")[["player_id", "stat_signal", "exp_ppg", "early_tgt_share",
                                                               "early_car_share", "early_snap_pct", "early_td_pg"]]
V = V.merge(S, on="player_id", how="left")
TV = pd.read_csv(ROOT + "/stats/proj/team_values_all.csv", dtype={"sleeper_id": str})
modes = pd.read_csv(ROOT + "/stats/proj/team_modes.csv", index_col=0)
settings = json.load(open(ROOT + "/stats/proj/actual_values_settings.json"))
cv = pd.read_csv(ROOT + "/stats/proj/projection_cv.csv")
nflteam = {p: d.get("team") for p, d in json.load(open(f"{LG}/players_nfl.json")).items()}

r = lambda x, k=1: None if pd.isna(x) else round(float(x), k)
# one linear 0-10,000 scale for every value on the page: the league's most valuable player = 10,000.
# Linear keeps proportions: 6,000 is worth exactly two 3,000s.
# Each mode gets its own scale so the two are comparable: the top 150 players carry the same total
# value in both. (Raw rebuilder numbers run smaller only because 2028-2031 are less certain, which
# made every player look worth more to a contender.) Then one factor puts the top value at 10,000.
TOPN = 150
_c = V.contender.nlargest(TOPN).sum(); _r = V.rebuilder.nlargest(TOPN).sum()
SC_C, SC_R = 1.0, _c / _r
_top = max(V.contender.max() * SC_C, V.rebuilder.max() * SC_R)
SC_C, SC_R = 10000 * SC_C / _top, 10000 * SC_R / _top
SCALE = SC_C
for c in ["contender", "contender_lo", "contender_hi"]:
    V[c] = V[c] * SC_C
for c in ["rebuilder", "rebuilder_lo", "rebuilder_hi"]:
    V[c] = V[c] * SC_R
TV["team_contender"] = TV["team_contender"] * SC_C
TV["team_rebuilder"] = TV["team_rebuilder"] * SC_R
players = []
for row in V.itertuples():
    players.append({
        "id": row.sleeper_id, "n": row.name, "p": row.pos, "t": nflteam.get(row.sleeper_id) or "",
        "age": r(row.age), "o": row.owner, "g": int(row.e_games), "ppg": r(row.e_ppg),
        "proj": [r(getattr(row, f"ppg{h}")) for h in range(6)],
        "lo": [r(getattr(row, f"ppg{h}_p10")) for h in range(6)],
        "hi": [r(getattr(row, f"ppg{h}_p90")) for h in range(6)],
        "ph": [r(getattr(row, f"phealthy{h}"), 3) for h in range(6)],
        "gm": [r(getattr(row, f"games{h}"), 2) for h in range(6)],
        "sd": [r(getattr(row, f"sd{h}"), 2) for h in range(6)],
        "c": r(row.contender, 0), "cl": r(row.contender_lo, 0), "ch": r(row.contender_hi, 0),
        "rb": r(row.rebuilder, 0), "rl": r(row.rebuilder_lo, 0), "rh": r(row.rebuilder_hi, 0),
        "sig": r(row.stat_signal, 2), "tgt": r(row.early_tgt_share, 3), "car": r(row.early_car_share, 3),
        "snap": r(row.early_snap_pct, 3), "td": r(row.early_td_pg, 2),
    })
index = {p["id"]: i for i, p in enumerate(players)}

# weekly trend data (2026 to date and 2025) and recent-role numbers for each player
wk = pd.read_parquet(ROOT + "/stats/features_weekly.parquet")
wk["tshare"] = wk.targets / wk.team_targets.where(wk.team_targets > 0)
wk["cshare"] = wk.carries / wk.team_carries.where(wk.team_carries > 0)
F = pd.read_parquet(ROOT + "/stats/features.parquet")
F26 = F[F.season == 2026].set_index("player_id")
gid = dict(zip(V.sleeper_id, V.player_id))
wkg = {k: g for k, g in wk.groupby("player_id")}
for p in players:
    g = wkg.get(gid.get(p["id"]))
    p["w26"], p["w25"] = [], []
    if g is not None:
        for row in g[g.season == 2026].sort_values("week").itertuples():
            p["w26"].append([int(row.week), r(row.pts), r(row.offense_pct, 2),
                             r(row.cshare if p["p"] == "RB" else row.tshare, 3), int(bool(row.left_early))])
        p["w25"] = [[int(row.week), r(row.pts)] for row in g[g.season == 2025].sort_values("week").itertuples()]
    f = F26.loc[gid[p["id"]]] if gid.get(p["id"]) in F26.index else None
    if f is not None and f.recent_games >= 3 and f.early_games >= 4:
        key = "car_share" if p["p"] == "RB" else "tgt_share"
        p["rc"] = {"ppg": r(f.recent_ppg), "share": r(f[f"recent_{key}"], 3), "seasonShare": r(f[f"early_{key}"], 3)}

users = {u["user_id"]: u for u in json.load(open(f"{LG}/league_1313707182407024640_users.json"))}
teams = []
for ro in json.load(open(f"{LG}/league_1313707182407024640_rosters.json")):
    u = users.get(ro["owner_id"], {})
    name = u.get("display_name")
    s = ro.get("settings") or {}
    tv = TV[TV.team == name].set_index("sleeper_id")
    vals = [[r(tv.at[p["id"], "team_contender"], 0), r(tv.at[p["id"], "team_rebuilder"], 0)] if p["id"] in tv.index else [0, 0]
            for p in players]
    teams.append({
        "name": name, "label": (u.get("metadata") or {}).get("team_name") or name, "rid": ro["roster_id"],
        "w": s.get("wins", 0), "l": s.get("losses", 0), "pf": round(s.get("fpts", 0) + s.get("fpts_decimal", 0) / 100, 1),
        "roster": [p for p in (ro.get("players") or []) if p in index], "taxi": [p for p in (ro.get("taxi") or []) if p in index],
        "mode": modes.at[name, "mode"], "projRos": round(float(modes.at[name, "proj"]), 0), "vals": vals,
    })

# genuine league trades (round trips, FAAB-only and startup pick swaps removed; Brady's manual excludes)
P = json.load(open(f"{LG}/players_nfl.json"))
rid2name = {t["rid"]: t["name"] for t in teams}
tx = [t for f in glob.glob(f"{LG}/tx_*.json") for t in (json.load(open(f)) or [])]
trades = sorted([t for t in tx if t["type"] == "trade" and t["status"] == "complete"], key=lambda t: t["status_updated"])
flag = set()
for i, a in enumerate(trades):
    for b in trades[i + 1:]:
        if sorted(a["roster_ids"]) == sorted(b["roster_ids"]) and set(a.get("adds") or {}) & set(b.get("adds") or {}) \
                and b["status_updated"] - a["status_updated"] < 21 * 86400 * 1000:
            flag |= {a["transaction_id"], b["transaction_id"]}
manual = {("J.J. McCarthy", "Carson Beck")}
log = []
for t in trades:
    names = {P.get(p, {}).get("full_name", p) for p in (t.get("adds") or {})}
    picks = t.get("draft_picks") or []
    if t["transaction_id"] in flag or not (t.get("adds") or picks):
        continue
    if picks and all(d["season"] == "2026" for d in picks) and not t.get("adds"):
        continue
    if any(set(m) <= names for m in manual):
        continue
    sides = []
    for rid in t["roster_ids"]:
        got = [{"id": p, "n": P.get(p, {}).get("full_name", p)} for p, rr in (t.get("adds") or {}).items() if rr == rid]
        gp = [f"{d['season']} round {d['round']}" for d in picks if d["owner_id"] == rid]
        sides.append({"team": rid2name.get(rid), "players": got, "picks": gp})
    log.append({"date": datetime.datetime.fromtimestamp(t["status_updated"] / 1000).strftime("%b %-d"), "sides": sides})

INJ = json.load(open(ROOT + "/stats/proj/injury_outlook.json"))
for p in players:
    st = INJ["status"].get(p["id"])
    if st:
        p["inj"] = {"s": st["status"], "b": st["body"]}
    ol = INJ["outlook"].get(p["id"])
    if ol:
        p["ol"] = ol

# rookie picks (picks_value.py): same 0-10,000 scale; plus slot hit rates from the rookie study
PK = json.load(open(ROOT + "/stats/proj/picks.json"))
ordn = {1: "1st", 2: "2nd", 3: "3rd"}
pick_list = []
for k in PK["picks"]:
    pick_list.append({"id": k["id"], "season": k["season"], "round": k["round"], "orig": k["orig"], "o": k["owner"],
                      "n": f"{k['season']} {ordn[k['round']]}" + ("" if k["orig"] == k["owner"] else f" (from {k['orig']})"),
                      "exp": k["expSlot"], "slotP": k["slotP"], "c": round(k["c"] * SC_C), "rb": round(k["rb"] * SC_R)})
for t in teams:
    t["picks"] = [k["id"] for k in pick_list if k["o"] == t["name"]]
hit = pd.read_csv(ROOT + "/rookie/results/by_slot.csv", dtype={"slot": str}).set_index("slot")
slot_rows = []
for sl in PK["slots"]:
    h = hit.loc[sl["slot"]] if sl["slot"] in hit.index else None
    slot_rows.append({"slot": sl["slot"], "c27": round(sl["c27"] * SC_C), "r27": round(sl["r27"] * SC_R),
                      "hit": None if h is None else round(float(h["hit_rate%"]) / 100, 3),
                      "star": None if h is None else round(float(h["Star%"]) / 100, 3)})

data = {
    "meta": {"built": datetime.date.today().isoformat(), "throughWeek": WEEK, "restWeeks": REST, "season": 2026,
             "repl": settings["replacement_ppg"], "wC": settings["w_contender"], "wR": settings["w_rebuilder"],
             "cv": cv[["position", "horizon", "r2_calibrated", "auc_healthy", "resid_sd"]].to_dict("records"),
             "me": "BraundJB", "scale": SCALE, "scaleC": SC_C, "scaleR": SC_R},
    "players": players, "teams": teams, "trades": log, "picks": pick_list, "slots": slot_rows,
    # rookie slot study (rookie/scripts/build_study_site.py) for the Draft study tab
    "study": json.load(open(ROOT + "/rookie/results/site_study.json")),
}
json.dump(data, open(OUT, "w"), separators=(",", ":"))
print(len(players), "players", len(teams), "teams", len(log), "trades", round(len(json.dumps(data)) / 1024), "KB")
