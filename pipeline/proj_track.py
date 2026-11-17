"""Projection tracker: save each week's pre-game projections, then score them against what happened.

  save      WEEK_PROJ_JSON TRACK_DIR
            Merge the current week_proj.json into TRACK_DIR/<season>-w<NN>.json. A player's projection is
            only updated before his game kicks off (kickoff is US Eastern), so the file ends up holding the
            last projection made before each game.
  backfill  ROWS_PARQUET MODEL_PKL IDMAP TRACK_DIR
            For this season's completed weeks with no saved file: what the weekly model would have said,
            from a model trained only on 2016 to last season (src "backtest"). Pre-game features only.
  score     TRACK_DIR OUT_JSON
            For each completed week with projections: pulls actual points and Sleeper's own projections
            (cached in TRACK_DIR), and writes the accuracy summary the page shows.

Scored set: players who played, with ours or Sleeper's projection at 5+ points (both must exist).
Points are Sleeper's full-PPR (the league's scoring).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from wrconf import SEASON, WEEK  # noqa: E402
import datetime
import glob
import json
import os
import pickle
import sys
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np

ET = ZoneInfo("America/New_York")
fn = lambda d, w: os.path.join(d, f"{SEASON}-w{int(w):02d}.json")


def save(src, d):
    W = json.load(open(src))
    os.makedirs(d, exist_ok=True)
    path = fn(d, W["week"])
    old = json.load(open(path)) if os.path.exists(path) else {"season": W["season"], "week": W["week"], "src": "live", "players": {}}
    now = datetime.datetime.now(ET)
    n = 0
    for sid, x in W["players"].items():
        if x.get("bye"):
            continue
        ko = x.get("ko")
        try:
            kt = datetime.datetime.strptime(ko, "%Y-%m-%d %H:%M").replace(tzinfo=ET) if ko else None
        except ValueError:
            kt = None
        if kt is not None and now >= kt:
            continue  # game started: keep the last pre-game projection
        old["players"][sid] = {"p": x["p"], "pl": x["pl"], "x": x["x"], "lo": x["lo"], "hi": x["hi"], "st": x.get("st"),
                               "at": now.strftime("%Y-%m-%d %H:%M")}
        n += 1
    json.dump(old, open(path, "w"), separators=(",", ":"))
    print(path, n, "projections updated,", len(old["players"]), "saved")


def backfill(rows, pkl, idmap, d):
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    from week_proj import FEATS, KW, POSC, PLAY
    need = [w for w in range(1, WEEK + 1) if not os.path.exists(fn(d, w))]
    if not need:
        print("backfill: nothing to do"); return
    R = pd.read_parquet(rows)
    R["pos_c"] = R.pos.map(POSC)
    T = R[(R.played == 1) & R.pts.notna() & (R.season < SEASON)]
    m = HistGradientBoostingRegressor(**KW).fit(T[FEATS], T.pts)
    M = pickle.load(open(pkl, "rb"))
    c = M["corr"]
    ids = pd.read_csv(idmap, low_memory=False).dropna(subset=["sleeper_id", "gsis_id"])
    g2s = dict(zip(ids.gsis_id, ids.sleeper_id.astype(int).astype(str)))
    os.makedirs(d, exist_ok=True)
    for w in need:
        X = R[(R.season == SEASON) & (R.week == w)].copy()
        if X.empty:
            continue
        pred = np.clip(m.predict(X[FEATS].astype(float)), 0, None)
        pred = pred * np.where(X.q == 1, np.where(X.prac_dnp == 1, c["q_dnp"], c["q_other"]), c["healthy"])
        play = np.where(X.o == 1, 0.0, np.where(X.d == 1, PLAY["Doubtful"], np.where(X.q == 1, np.where(X.prac_dnp == 1, PLAY["Q_dnp"],
               np.where(X.prac_lim == 1, PLAY["Q_lim"], np.where(X.prac_full == 1, PLAY["Q_full"], PLAY["Questionable"]))), PLAY["healthy"])))
        band = np.clip(np.digitize(pred, M["bands"]) - 1, 0, len(M["q10"]) - 1)
        lo, hi = pred * np.array(M["q10"])[band], pred * np.array(M["q90"])[band]
        out = {}
        for i, (pid, st) in enumerate(zip(X.player_id, X.report_status)):
            sid = g2s.get(pid)
            if sid:
                out[sid] = {"p": round(float(pred[i]), 1), "pl": round(float(play[i]), 3), "x": round(float(pred[i] * play[i]), 1),
                            "lo": round(float(lo[i]), 1), "hi": round(float(hi[i]), 1), "st": st if isinstance(st, str) else None}
        json.dump({"season": SEASON, "week": w, "src": "backtest", "players": out}, open(fn(d, w), "w"), separators=(",", ":"))
        print("backfill week", w, len(out), "players")


def sleeper(kind, w):
    url = f"https://api.sleeper.app/{kind}/nfl/{SEASON}/{w}?season_type=regular"
    req = urllib.request.Request(url, headers={"User-Agent": "war-room"})
    data = json.load(urllib.request.urlopen(req, timeout=60))
    out = {}
    for r in data:
        s = r.get("stats") or {}
        if kind == "stats" and not s.get("gp"):
            continue
        if s.get("pts_ppr") is not None:
            out[str(r["player_id"])] = round(float(s["pts_ppr"]), 2)
    return out


def score(d, out_path):
    weeks, pos_rows, misses = [], [], []
    pos_of = {}
    for path in sorted(glob.glob(os.path.join(d, f"{SEASON}-w*.json"))):
        F = json.load(open(path))
        w = F["week"]
        if w > WEEK:
            continue  # not played yet
        cache = {}
        for kind, name in (("stats", "actual"), ("projections", "sleeper")):
            cp = os.path.join(d, f"{name}-{SEASON}-w{w:02d}.json")
            # re-pull the latest completed week (stat corrections), otherwise use the cache
            if os.path.exists(cp) and (w < WEEK or name == "sleeper"):
                cache[name] = json.load(open(cp))
            else:
                try:
                    cache[name] = sleeper(kind, w)
                    json.dump(cache[name], open(cp, "w"), separators=(",", ":"))
                except Exception as e:  # network trouble: try next run
                    print("week", w, name, "unavailable:", e)
                    cache[name] = json.load(open(cp)) if os.path.exists(cp) else {}
        A, S = cache["actual"], cache["sleeper"]
        rows = []
        for sid, x in F["players"].items():
            if sid not in A or sid not in S:
                continue
            if max(x["p"], S[sid]) < 5:
                continue
            rows.append((sid, x["p"], S[sid], A[sid], x["lo"], x["hi"], x.get("st")))
        if not rows:
            continue
        p = np.array([r[1] for r in rows]); s = np.array([r[2] for r in rows]); a = np.array([r[3] for r in rows])
        lo = np.array([r[4] for r in rows]); hi = np.array([r[5] for r in rows])
        q = np.array([r[6] == "Questionable" for r in rows])
        weeks.append({"w": w, "src": F.get("src", "live"), "n": len(rows),
                      "mae": round(float(np.abs(a - p).mean()), 2), "maeS": round(float(np.abs(a - s).mean()), 2),
                      "bias": round(float((a - p).mean()), 2), "biasS": round(float((a - s).mean()), 2),
                      "cover": round(float(((a >= lo) & (a <= hi)).mean()), 3),
                      "closer": round(float((np.abs(a - p) < np.abs(a - s)).mean()), 3),
                      "nq": int(q.sum()), "biasQ": round(float((a[q] - p[q]).mean()), 2) if q.any() else None})
        for r in rows:
            pos_rows.append((w, r[0]) + r[1:4])
        if w == max(x["w"] for x in weeks):
            misses = sorted(rows, key=lambda r: -abs(r[3] - r[1]))[:10]
            misses = [{"id": r[0], "p": r[1], "s": r[2], "a": r[3]} for r in misses]
    # by position, all weeks together (position from the site's player list when available)
    try:
        SP = json.load(open(os.path.join(os.environ.get("WR_ROOT", "."), "league/raw/players_nfl.json")))
        pos_of = {k: v.get("position") for k, v in SP.items()}
    except Exception:
        pass
    by_pos = {}
    for pos in ("QB", "RB", "WR", "TE"):
        g = [r for r in pos_rows if pos_of.get(r[1]) == pos]
        if g:
            p = np.array([r[2] for r in g]); s = np.array([r[3] for r in g]); a = np.array([r[4] for r in g])
            by_pos[pos] = {"n": len(g), "mae": round(float(np.abs(a - p).mean()), 2), "maeS": round(float(np.abs(a - s).mean()), 2),
                           "bias": round(float((a - p).mean()), 2)}
    json.dump({"season": SEASON, "weeks": weeks, "pos": by_pos, "misses": misses}, open(out_path, "w"), separators=(",", ":"))
    for x in weeks:
        print(x)
    print(by_pos)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "save":
        save(*sys.argv[2:4])
    elif cmd == "backfill":
        backfill(*sys.argv[2:6])
    elif cmd == "score":
        score(*sys.argv[2:4])
