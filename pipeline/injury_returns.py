"""Return timing and opportunity outlook for 2026.

1. History (2012-2025): every absence spell for a QB/RB/WR/TE who had played earlier that season and
   was listed Out or Doubtful when the spell began. Spell length = consecutive team games missed
   until he played again (censored at season end). Grouped by injury type from the injury report.
2. Now: for each player Sleeper lists as IR / Out / PUP / Doubtful / Questionable, the chance of
   being back by each remaining week, conditional on games already missed (IR = 4-game minimum;
   torn ACL or Achilles = season over).
3. Outlook: a player "gains" while a teammate at his position who held a real role is out, and that
   gain is "at risk" by the week the teammate is likely back.

Usage: python3 -I injury_returns.py RAW_DIR FEATURES SLEEPER_PLAYERS_JSON IDMAP OUT_JSON
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from wrconf import ROOT, SEASON, WEEK, REST, LATE_MIN, BACK_BY, LAST_WEEK, PLAYOFF  # noqa: E402
import json
import sys
import numpy as np
import pandas as pd

RAW, FEAT, SP, IDMAP, OUT = sys.argv[1:6]
POS = ["QB", "RB", "WR", "TE"]
CUR_WEEK = WEEK  # LAST_WEEK and PLAYOFF come from wrconf
CATS = [("season", ["acl", "achilles"]), ("concussion", ["concussion"]), ("hamstring", ["hamstring"]),
        ("ankle", ["ankle"]), ("knee", ["knee", "mcl", "meniscus"]), ("foot", ["foot", "toe", "lisfranc"]),
        ("shoulder", ["shoulder", "collarbone", "clavicle"]), ("groin/hip", ["groin", "hip", "abdom", "oblique"]),
        ("calf/quad", ["calf", "quad", "thigh"]), ("back/ribs", ["back", "rib", "chest", "neck"])]


def cat(text):
    t = str(text or "").lower()
    for name, keys in CATS:
        if any(k in t for k in keys):
            return name
    return "other"


# ---- history ---------------------------------------------------------------------------------
rows = []
for y in range(2012, 2026):
    w = pd.read_csv(f"{RAW}/week_{y}.csv", low_memory=False, usecols=["player_id", "position", "team", "week", "season_type"])
    w = w[(w.season_type == "REG") & w.position.isin(POS)]
    inj = pd.read_csv(f"{RAW}/injuries_{y}.csv", low_memory=False)
    inj = inj[(inj.game_type == "REG") & inj.report_status.isin(["Out", "Doubtful"])]
    inj = inj.drop_duplicates(["gsis_id", "week"]).set_index(["gsis_id", "week"]).report_primary_injury
    team_weeks = w.groupby("team").week.apply(lambda s: sorted(set(s))).to_dict()
    for (pid, team), g in w.groupby(["player_id", "team"]):
        weeks = sorted(set(g.week))
        tw = team_weeks.get(team, [])
        played = set(weeks)
        i = 0
        while i < len(tw):
            wk = tw[i]
            if wk not in played and any(x < wk for x in weeks) and (pid, wk) in inj.index:
                kind = cat(inj.loc[(pid, wk)])
                j = i
                while j < len(tw) and tw[j] not in played:
                    j += 1
                rows.append({"cat": kind, "missed": j - i, "returned": j < len(tw)})
                i = j
            i += 1
H = pd.DataFrame(rows)
print("absence spells:", len(H))
print(H.groupby("cat").agg(n=("missed", "size"), median=("missed", "median"),
                           back=("returned", "mean")).round(2).to_string())


def surv(c):
    """Discrete survival S(k) = P(still out after k games), Kaplan-Meier with season-end censoring."""
    d = H[H.cat == c] if (H.cat == c).sum() >= 30 else H
    S, out = [1.0], 1.0
    for k in range(1, 18):
        at_risk = (d.missed >= k).sum()
        events = ((d.missed == k) & d.returned).sum()  # back after missing exactly k games
        out *= 1 - (events / at_risk if at_risk else 0)
        S.append(out)
    return np.array(S)


SURV = {c: surv(c) for c in set(H.cat) | {"other"}}

# ---- now ---------------------------------------------------------------------------------------
T = pd.read_parquet(FEAT)
T26 = T[T.season == 2026].set_index("player_id")
ids = pd.read_csv(IDMAP, low_memory=False).dropna(subset=["sleeper_id", "gsis_id"])
s2g = dict(zip(ids.sleeper_id.astype(int).astype(str), ids.gsis_id))
SPL = json.load(open(SP))
remaining = list(range(CUR_WEEK + 1, LAST_WEEK + 1))  # team games left, byes ignored


def p_back_by(status, body, notes, missed):
    """P(back for the game in week w) for each remaining week."""
    c = cat(body)
    if status == "Questionable":
        return {w: 0.8 for w in remaining}, "q"
    if c == "season" or (status == "IR" and "surgery" in str(notes).lower() and "knee" in str(body).lower()):
        return {w: 0.0 for w in remaining}, "season"
    S = SURV.get(c, SURV["other"])
    m = int(missed or 0)
    floor = 4 if status == "IR" else (1 if status in ("Out", "Doubtful", "PUP") else 0)
    out = {}
    for i, w in enumerate(remaining, start=1):
        k = m + i - 1  # games he will have missed before week w
        if k < floor:
            out[w] = 0.0
            continue
        base = S[min(m, 17)] if S[min(m, 17)] > 0 else 1e-9
        out[w] = float(np.clip(1 - S[min(k, 17)] / base, 0, 1))
    return out, "out"


status = {}
for sid, d in SPL.items():
    st = d.get("injury_status")
    if d.get("position") in POS and d.get("team") and st in ("IR", "Out", "PUP", "Doubtful", "Questionable"):
        g = s2g.get(sid)
        missed = T26.at[g, "early_wks_since_last"] if g in T26.index else CUR_WEEK
        probs, kind = p_back_by(st, d.get("injury_body_part"), d.get("injury_notes"), missed if pd.notna(missed) else CUR_WEEK)
        likely = next((w for w in remaining if probs[w] >= 0.5), None)
        status[sid] = {"status": st, "body": d.get("injury_body_part"), "kind": kind, "back": probs,
                       "likelyWeek": likely, "pPlayoffs": round(probs.get(max(PLAYOFF, CUR_WEEK + 1), 0.0), 2), "name": d.get("full_name"),
                       "team": d.get("team"), "pos": d.get("position")}

# role shares this season for the outlook
role = {}
for sid, g in s2g.items():
    if g in T26.index:
        r = T26.loc[g]
        share = r.early_car_share if r.pos == "RB" else (np.nan if r.pos == "QB" else r.early_tgt_share)
        role[sid] = {"share": None if pd.isna(share) else float(share),
                     "recent": None if pd.isna(r.get("recent_car_share" if r.pos == "RB" else "recent_tgt_share")) else
                     float(r.get("recent_car_share" if r.pos == "RB" else "recent_tgt_share")),
                     "depth": None if pd.isna(r.early_depth_rank) else int(r.early_depth_rank)}

outlook = {}
for sid, d in SPL.items():
    if d.get("position") not in POS or not d.get("team"):
        continue
    if sid in status and status[sid]["kind"] != "q":
        s = status[sid]
        if s["kind"] == "season":
            outlook[sid] = {"kind": "out", "text": f"{s['status']} ({s['body'] or 'injury'}): likely out for the season"}
        else:
            lw = s["likelyWeek"]
            outlook[sid] = {"kind": "out", "text": f"{s['status']} ({s['body'] or 'injury'}): "
                            + (f"likely back week {lw}, " if lw else "may not return this season, ")
                            + f"{round(100 * s['pPlayoffs'])}% by week {PLAYOFF}"}
        continue
    me = role.get(sid, {})
    mates = [o for o, s in status.items() if s["kind"] != "q" and s["team"] == d.get("team") and s["pos"] == d.get("position") and o != sid]
    mates = [o for o in mates if (role.get(o, {}).get("share") or 0) >= 0.15 or (role.get(o, {}).get("depth") or 9) <= 1]
    if mates:
        o = max(mates, key=lambda x: role.get(x, {}).get("share") or 0)
        s = status[o]
        mine = me.get("depth") or 9
        theirs = role.get(o, {}).get("depth") or 9
        if (role.get(o, {}).get("share") or 0) > (me.get("share") or 0) * 0.8 or theirs < mine or (me.get("recent") or 0) > (me.get("share") or 0) + 0.05:
            if s["kind"] == "season":
                outlook[sid] = {"kind": "gain", "text": f"Gaining while {s['name']} is out; he's likely out for the season"}
            else:
                lw = s["likelyWeek"]
                outlook[sid] = {"kind": "risk", "text": f"Gaining while {s['name']} is out; "
                                + (f"he's likely back week {lw}" if lw else "he may not return this season")
                                + f" ({round(100 * s['pPlayoffs'])}% by week {PLAYOFF})"}
    elif sid in status:
        outlook[sid] = {"kind": "q", "text": f"Questionable ({status[sid]['body'] or 'injury'})"}

json.dump({"status": status, "outlook": outlook}, open(OUT, "w"))
pd.set_option("display.width", 200)
for n in ["Ollie Gordon II", "De'Von Achane", "Rashee Rice", "De'Zhaun Stribling", "Jahmyr Gibbs", "Jalen Coker"]:
    sid = next((k for k, v in SPL.items() if v.get("full_name") == n), None)
    print(n, "|", outlook.get(sid, {}).get("text"), "|", status.get(sid, {}).get("likelyWeek"))
print(len(status), "injured listed,", len(outlook), "outlook notes")
