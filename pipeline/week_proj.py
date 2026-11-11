"""Next-week projections for every player.

  train   (full weekly build): fit the weekly model on every played game 2016 to now, plus held-out
          corrections: per-injury-status multipliers and floor/ceiling ratios learned from each season
          predicted by a model that never saw it (2018-2025). Saves proj/week_model.pkl and each player's
          form/usage going into next week (proj/week_base.parquet).
  predict (full and light builds): combine those with the latest lines (nflverse schedule), Sleeper
          injury statuses and teammates out, and write proj/week_proj.json for the target week
          (the week after the last completed one).

Play chance (injured players): Questionable 73% (Full practice 91%, Limited 74%, did not practice 52%),
Doubtful 2%, Out/IR 0, from the 2016-2026 Questionable study.

Usage: python3 -I week_proj.py train ROWS_PARQUET OUT_DIR
       python3 -I week_proj.py predict OUT_DIR GAMES_CSV SLEEPER_PLAYERS_JSON IDMAP
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from wrconf import ROOT, SEASON, WEEK  # noqa: E402
import json
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATS = ["std_pts", "l3_pts", "last_pts", "ps_ppg", "games_so_far", "std_tshare", "l3_tshare", "std_cshare", "l3_cshare",
         "std_ashare", "l3_ashare", "std_snap", "l3_snap", "ps_tshare", "ps_cshare", "implied", "opp_implied", "spread",
         "total_line", "home", "dome", "rest", "opp_allow_rel", "q", "prac_dnp", "prac_lim", "prac_full", "hammy",
         "concussion", "missed_last", "gap", "mates_out_usage", "mates_out_pts", "age", "pos_c"]
KW = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
POSC = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
PLAY = {"healthy": 0.985, "Questionable": 0.73, "Q_full": 0.905, "Q_lim": 0.736, "Q_dnp": 0.519, "Doubtful": 0.016}
mode = sys.argv[1]


def train(rows_path, out):
    R = pd.read_parquet(rows_path)
    R["pos_c"] = R.pos.map(POSC)
    D = R[(R.played == 1) & R.pts.notna()].copy()
    # held-out predictions, 2018-2025, for the corrections
    oos = []
    for s in range(2018, 2026):
        m = HistGradientBoostingRegressor(**KW).fit(D[D.season != s][FEATS], D[D.season != s].pts)
        oos.append(pd.Series(m.predict(D[D.season == s][FEATS]), index=D[D.season == s].index))
    D["oos"] = pd.concat(oos)
    E = D[D.oos.notna() & (D.oos >= 3)]
    # status corrections: actual / predicted for injured-but-played games (shrunk toward 1 by sample size)
    corr = {}
    for nm, m in [("q_dnp", (E.q == 1) & (E.prac_dnp == 1)), ("q_other", (E.q == 1) & (E.prac_dnp == 0)), ("healthy", E.q == 0)]:
        x = E[m]
        raw = x.pts.sum() / x.oos.sum()
        corr[nm] = float(1 + (raw - 1) * len(x) / (len(x) + 200))
    # floor / ceiling: 10th and 90th percentile of actual/predicted by projection band
    E = E.assign(ratio=E.pts / E.oos, band=pd.cut(E.oos, [0, 6, 9, 12, 15, 18, 100], labels=False))
    q = E.groupby("band").ratio.quantile([0.1, 0.9]).unstack().round(3)
    final = HistGradientBoostingRegressor(**KW).fit(D[FEATS], D.pts)
    bands = [0, 6, 9, 12, 15, 18, 100]
    pickle.dump({"model": final, "corr": corr, "bands": bands, "q10": q[0.1].tolist(), "q90": q[0.9].tolist(),
                 "mae_oos": float((E.pts - E.oos).abs().mean())}, open(f"{out}/week_model.pkl", "wb"))
    print("held-out MAE", round(float((E.pts - E.oos).abs().mean()), 2), "corrections", {k: round(v, 3) for k, v in corr.items()})
    # each player's form going into the next week: the last row of this season, rolled forward one game
    cur = R[R.season == SEASON].sort_values("week")
    base = []
    for pid, g in cur.groupby("player_id"):
        pl = g[g.played == 1]
        last = g.iloc[-1]
        pts, ts, cs, as_, sn = pl.pts, pl.tshare, pl.cshare, pl.ashare, pl.snap
        base.append({"player_id": pid, "name": last["name"], "pos": last.pos, "team": last.team, "age": last.age,
                     "std_pts": pts.mean() if len(pl) else np.nan, "l3_pts": pts.tail(3).mean() if len(pl) else np.nan,
                     "last_pts": pts.iloc[-1] if len(pl) else np.nan, "games_so_far": float(len(pl)),
                     "std_tshare": ts.mean(), "l3_tshare": ts.tail(3).mean(), "std_cshare": cs.mean(), "l3_cshare": cs.tail(3).mean(),
                     "std_ashare": as_.mean(), "l3_ashare": as_.tail(3).mean(), "std_snap": sn.mean(), "l3_snap": sn.tail(3).mean(),
                     "ps_ppg": last.ps_ppg, "ps_tshare": last.ps_tshare, "ps_cshare": last.ps_cshare,
                     "last_week": int(last.week), "played_last": int(last.played)})
    pd.DataFrame(base).to_parquet(f"{out}/week_base.parquet")
    print(len(base), "players carried into next week")


def predict(out, games_csv, sleeper_json, idmap):
    M = pickle.load(open(f"{out}/week_model.pkl", "rb"))
    B = pd.read_parquet(f"{out}/week_base.parquet")
    target = WEEK + 1
    G = pd.read_csv(games_csv)
    G = G[(G.season == SEASON) & (G.week == target) & (G.game_type == "REG")]
    rows = []
    for _, g in G.iterrows():
        for team, opp, home, sgn, rest in ((g.home_team, g.away_team, 1, 1, g.home_rest), (g.away_team, g.home_team, 0, -1, g.away_rest)):
            tl, sp = g.total_line, g.spread_line
            have = pd.notna(tl) and pd.notna(sp)
            rows.append({"team": team, "opp": opp, "home": home, "rest": rest, "dome": float(g.roof in ("dome", "closed")),
                         "implied": (tl + sgn * sp) / 2 if have else np.nan, "opp_implied": (tl - sgn * sp) / 2 if have else np.nan,
                         "spread": sgn * sp if have else np.nan, "total_line": tl if have else np.nan, "kickoff": f"{g.gameday} {g.gametime}"})
    GC = pd.DataFrame(rows)
    # opponent strength vs position, from this season's games so far (weekly rows, same definition as training)
    allow = pd.read_parquet(f"{out}/week_allow.parquet") if _os.path.exists(f"{out}/week_allow.parquet") else None
    ids = pd.read_csv(idmap, low_memory=False).dropna(subset=["sleeper_id", "gsis_id"])
    g2s = dict(zip(ids.gsis_id, ids.sleeper_id.astype(int).astype(str)))
    SP = json.load(open(sleeper_json))
    B["sid"] = B.player_id.map(g2s)
    B = B[B.sid.notna()].copy()
    B["team_now"] = [SP.get(s, {}).get("team") or t for s, t in zip(B.sid, B.team)]
    B["team_now"] = B.team_now.replace({"LAR": "LA", "JAC": "JAX", "WSH": "WAS"})
    st = {s: (SP.get(s, {}).get("injury_status") or "") for s in B.sid}
    prac = {s: str(SP.get(s, {}).get("practice_participation") or "") for s in B.sid}
    body = {s: str(SP.get(s, {}).get("injury_body_part") or "") for s in B.sid}
    B["status"] = B.sid.map(st)
    B["q"] = (B.status == "Questionable").astype(float)
    B["prac_dnp"] = B.sid.map(lambda s: float("dnp" in prac[s].lower() or "did not" in prac[s].lower()))
    B["prac_lim"] = B.sid.map(lambda s: float("lim" in prac[s].lower()))
    B["prac_full"] = B.sid.map(lambda s: float("full" in prac[s].lower()))
    B["hammy"] = B.sid.map(lambda s: float("hamstring" in body[s].lower()))
    B["concussion"] = B.sid.map(lambda s: float("concussion" in body[s].lower()))
    B["missed_last"] = (B.played_last == 0).astype(float)
    B["gap"] = target - B.last_week
    B["pos_c"] = B.pos.map(POSC)
    B = B.merge(GC, left_on="team_now", right_on="team", how="left", suffixes=("", "_g"))
    # teammates at the position out this week
    out_now = B[B.status.isin(["Out", "IR", "Doubtful", "PUP", "Sus"])]
    use = np.where(out_now.pos == "RB", out_now.std_cshare, out_now.std_tshare)
    mo = out_now.assign(u=use).groupby(["team_now", "pos"]).agg(mates_out_usage=("u", "sum"), mates_out_pts=("std_pts", "sum")).reset_index()
    B = B.merge(mo, on=["team_now", "pos"], how="left")
    selfo = B.status.isin(["Out", "IR", "Doubtful", "PUP", "Sus"])
    own_use = pd.Series(np.where(B.pos == "RB", B.std_cshare, B.std_tshare), index=B.index).fillna(0)
    B["mates_out_usage"] = np.where(selfo, (B.mates_out_usage.fillna(0) - own_use).clip(lower=0), B.mates_out_usage.fillna(0))
    B["mates_out_pts"] = np.where(selfo, (B.mates_out_pts.fillna(0) - B.std_pts.fillna(0)).clip(lower=0), B.mates_out_pts.fillna(0))
    if allow is not None:
        B = B.merge(allow, left_on=["opp", "pos"], right_on=["opp", "pos"], how="left")
    B["opp_allow_rel"] = B.get("opp_allow_rel", pd.Series(1.0, index=B.index)).fillna(1.0)
    bye = B.opp.isna()
    X = B[FEATS].astype(float)
    pred = np.clip(M["model"].predict(X), 0, None)
    c = M["corr"]
    mult = np.where(B.q == 1, np.where(B.prac_dnp == 1, c["q_dnp"], c["q_other"]), c["healthy"])
    pred = pred * mult
    play = np.where(B.status.isin(["Out", "IR", "PUP", "Sus", "NA"]), 0.0,
           np.where(B.status == "Doubtful", PLAY["Doubtful"],
           np.where(B.q == 1, np.where(B.prac_dnp == 1, PLAY["Q_dnp"], np.where(B.prac_lim == 1, PLAY["Q_lim"], np.where(B.prac_full == 1, PLAY["Q_full"], PLAY["Questionable"]))), PLAY["healthy"])))
    play = np.where(bye, 0.0, play)
    band = np.clip(np.digitize(pred, M["bands"]) - 1, 0, len(M["q10"]) - 1)
    lo, hi = pred * np.array(M["q10"])[band], pred * np.array(M["q90"])[band]
    res = {}
    for i, r in B.reset_index(drop=True).iterrows():
        res[r.sid] = {"p": round(float(pred[i]), 1), "pl": round(float(play[i]), 3), "x": round(float(pred[i] * play[i]), 1),
                      "lo": round(float(lo[i]), 1), "hi": round(float(hi[i]), 1), "opp": None if bye[i] else r.opp,
                      "h": None if bye[i] else int(r.home), "it": None if pd.isna(r.implied) else round(float(r.implied), 1),
                      "sp": None if pd.isna(r.spread) else round(float(r.spread), 1), "oa": round(float(r.opp_allow_rel), 2),
                      "bye": bool(bye[i]), "st": r.status or None, "ko": None if bye[i] else r.kickoff,
                      "mo": round(float(r.mates_out_usage), 2) if r.mates_out_usage else 0}
    json.dump({"week": target, "season": SEASON, "linesPosted": int(G.total_line.notna().sum()), "games": int(len(G)),
               "mae": round(M["mae_oos"], 2), "players": res}, open(f"{out}/week_proj.json", "w"), separators=(",", ":"))
    top = sorted(res.items(), key=lambda kv: -kv[1]["x"])[:12]
    print("week", target, "lines", int(G.total_line.notna().sum()), "/", len(G), "games;", len(res), "players")
    print([(SP.get(k, {}).get("full_name"), v["x"], v["opp"]) for k, v in top])


def save_allow(rows_path, out):
    """Opponent strength vs position going into next week (this season, mixed with last early on)."""
    R = pd.read_parquet(rows_path)
    W = R[(R.played == 1) & R.season.isin([SEASON - 1, SEASON])]
    al = W.groupby(["season", "week", "opp", "pos"]).pts.sum().reset_index()
    cur = al[al.season == SEASON].groupby(["opp", "pos"]).pts.agg(["mean", "size"]).reset_index()
    prev = al[al.season == SEASON - 1].groupby(["opp", "pos"]).pts.mean().rename("prev").reset_index()
    lg = al[al.season == SEASON].groupby("pos").pts.mean().rename("lg").reset_index()
    x = cur.merge(prev, on=["opp", "pos"], how="left").merge(lg, on="pos")
    k = x["size"] / (x["size"] + 4)
    x["opp_allow_rel"] = (k * x["mean"] + (1 - k) * x.prev.fillna(x.lg)) / x.lg
    x[["opp", "pos", "opp_allow_rel"]].to_parquet(f"{out}/week_allow.parquet")


if mode == "train":
    train(sys.argv[2], sys.argv[3])
    save_allow(sys.argv[2], sys.argv[3])
elif mode == "predict":
    predict(*sys.argv[2:6])
