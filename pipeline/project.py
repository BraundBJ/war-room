"""In-season projections from a week-7 vantage point (2026 has weeks 1-7).

For each player-season t, features = weeks 1-7 of t, full seasons t-1 and t-2, age,
experience and draft capital. Targets per horizon h:
  h=0  rest of season t (weeks 8+), PPG if 4+ games
  h=1..3  season t+h, PPG if 8+ games (a healthy season)
Two models per position x horizon (gradient boosting, NaN-native):
  PPG given a healthy season, and the probability of a healthy season.
Season-held-out CV gives honest accuracy and residual spread (binned by prediction level).
Recency weight: 6-season half-life, the v3.5 finding.

Usage: python3 -I project.py FEATURES PLAYERS_CSV OUT_DIR
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from wrconf import ROOT, SEASON, WEEK, REST, LATE_MIN, BACK_BY, LAST_WEEK, PLAYOFF  # noqa: E402
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
FEAT, PLAYERS, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
CUR, LAST_FULL = 2026, 2025
POS = ["QB", "RB", "WR", "TE"]

T = pd.read_parquet(FEAT)
pl = pd.read_csv(PLAYERS, usecols=["gsis_id", "display_name", "position", "birth_date",
                                   "rookie_season", "draft_pick", "draft_year"])
pl = pl.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")

E_COLS = ["ppg", "games", "tgt_share", "car_share", "snap_pct", "opp_pg", "wopr", "td_pg", "yds_pg",
          "rz_car_pg", "rz_tgt_pg", "att_pg", "pass_epa_per_db", "car_pg", "qb_rush_yds_pg", "pass_adot",
          "wks_since_last", "comp_absent", "comp_absent_long", "depth_rank"]
P_COLS = ["ppg", "games", "tgt_share", "car_share", "snap_pct", "opp_pg", "att_pg", "qb_rush_yds_pg"]

full = T.set_index(["player_id", "season"])
# base rows: present in season t, or healthy (4+ games) in t-1 and t <= CUR
present = T[["player_id", "season"]]
carried = T[T.full_games >= 4][["player_id", "season"]].assign(season=lambda d: d.season + 1)
# plus every drafted player in his rookie and second seasons, stats or not, so injured or inactive
# young players (e.g. a high pick who hasn't played) are trained on and projected from draft capital
dr = pl[pl.draft_year.notna() & pl.position.isin(POS)].reset_index()[["gsis_id", "draft_year"]]
young = pd.concat([dr.assign(season=dr.draft_year + k) for k in (0, 1)])
young = young.rename(columns={"gsis_id": "player_id"})[["player_id", "season"]].astype({"season": int})
base = pd.concat([present, carried[carried.season <= CUR], young[young.season <= CUR]]).drop_duplicates()
base = base[base.season >= 2013].reset_index(drop=True)


def take(cols, prefix, shift, newp):
    idx = pd.MultiIndex.from_arrays([base.player_id, base.season - shift])
    got = full.reindex(idx)[[f"{prefix}_{c}" for c in cols]]
    got.columns = [f"{newp}_{c}" for c in cols]
    return got.reset_index(drop=True)


R_COLS = ["ppg", "games", "tgt_share", "car_share", "snap_pct", "opp_pg"]
X = pd.concat([base, take(E_COLS, "early", 0, "e"), take(R_COLS, "recent", 0, "rc"), take(P_COLS, "full", 1, "p1"),
               take(["ppg", "games"], "full", 2, "p2")], axis=1)
X["e_games"] = X.e_games.fillna(0)
X["p1_games"] = X.p1_games.fillna(0)
X["p2_games"] = X.p2_games.fillna(0)
info = pl.reindex(X.player_id)
bd = pd.to_datetime(info.birth_date.values, errors="coerce")
X["age"] = ((pd.to_datetime(X.season.astype(str) + "-09-01").values - bd.values) / np.timedelta64(1, "D") / 365.25)
X["exp"] = (X.season - pd.Series(info.rookie_season.values).fillna(pd.Series(info.draft_year.values))).clip(lower=0)
X["draft_pick"] = info.draft_pick.fillna(260).values
X["name"] = info.display_name.values
# position: latest seen in the stats, else the player file
lastpos = T.sort_values("season").groupby("player_id").pos.last()
X["pos"] = X.player_id.map(lastpos).fillna(pd.Series(info.position.values))
X = X[X.pos.isin(POS)].reset_index(drop=True)

# targets
nxt = lambda c, k: full.reindex(pd.MultiIndex.from_arrays([X.player_id, X.season + k]))[c].values
X["y0"] = np.where(nxt("late_games", 0) >= LATE_MIN, nxt("late_ppg", 0), np.nan)
X["q0"] = (np.nan_to_num(nxt("late_games", 0)) >= LATE_MIN).astype(float)
X["g0"] = np.nan_to_num(nxt("late_games", 0))
for k in (1, 2, 3):
    X[f"y{k}"] = np.where(nxt("full_games", k) >= 8, nxt("full_ppg", k), np.nan)
    X[f"q{k}"] = (np.nan_to_num(nxt("full_games", k)) >= 8).astype(float)
    X[f"g{k}"] = np.nan_to_num(nxt("full_games", k))

# recent minus full-window usage: the size of a role change
for c in ["ppg", "tgt_share", "car_share", "snap_pct", "opp_pg"]:
    X[f"rc_d_{c}"] = X[f"rc_{c}"] - X[f"e_{c}"]
X["rc_games"] = X.rc_games.fillna(0)
# players whose same-position teammate is out for the season (from injury_returns.py)
import json as _json
SEASON_GONE_MATES = set()
try:
    _io = _json.load(open(os.path.join(OUT, "injury_outlook.json")))
    _ids = pd.read_csv(ROOT + "/rookie/raw/db_playerids.csv", low_memory=False).dropna(subset=["sleeper_id", "gsis_id"])
    _s2g = dict(zip(_ids.sleeper_id.astype(int).astype(str), _ids.gsis_id))
    SEASON_GONE_MATES = {_s2g[k] for k, v in _io["outlook"].items()
                         if v["kind"] == "gain" and "out for the season" in v["text"] and k in _s2g}
    print("season-ending teammate adjustments:", len(SEASON_GONE_MATES))
except FileNotFoundError:
    pass
# age split at 27: free to rise before, only allowed to hold or fall after (monotone constraint),
# so sparse data on old players can't project a 36-year-old as flat or rising
X["age_young"] = X.age.clip(upper=27)
X["age_old"] = X.age.clip(lower=27)
X["exp_young"] = X.exp.clip(upper=6)
X["exp_old"] = X.exp.clip(lower=6)
FEATS = [c for c in X.columns if c.startswith(("e_", "rc_", "p1_", "p2_"))] + ["age_young", "age_old", "exp_young", "exp_old", "draft_pick"]
MONO = [-1 if f in ("age_old", "exp_old") else 0 for f in FEATS]
LAST_T = {0: LAST_FULL, 1: LAST_FULL - 1, 2: LAST_FULL - 2, 3: LAST_FULL - 3}
reg_kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=25, l2_regularization=1.0,
              monotonic_cst=MONO)

cv_rows, preds = [], []
cur = X[X.season == CUR].copy()
OOF_ROWS, CAL_ROWS = [], []
CAL_KEEP = 0.5  # share of the extra calibration terms kept
for pos in POS:
    D = X[X.pos == pos]
    out = cur[cur.pos == pos][["player_id", "name", "pos", "age", "exp", "e_games", "e_ppg", "p1_ppg"]].copy()
    for h in (0, 1, 2, 3):
        tr = D[D.season <= LAST_T[h]]
        w = 0.5 ** ((LAST_T[h] - tr.season) / 6)
        healthy = tr[tr[f"y{h}"].notna()]
        # season-held-out CV
        oof = pd.Series(np.nan, index=healthy.index)
        oofq = pd.Series(np.nan, index=tr.index)
        for s in sorted(tr.season.unique()):
            a, b = healthy[healthy.season != s], healthy[healthy.season == s]
            if len(b):
                m = HistGradientBoostingRegressor(**reg_kw).fit(a[FEATS], a[f"y{h}"], sample_weight=w[a.index])
                oof[b.index] = m.predict(b[FEATS])
            a, b = tr[tr.season != s], tr[tr.season == s]
            c = HistGradientBoostingClassifier(**reg_kw).fit(a[FEATS], a[f"q{h}"], sample_weight=w[a.index])
            oofq[b.index] = c.predict_proba(b[FEATS])[:, 1]
        yv = healthy[f"y{h}"]
        r2 = 1 - ((yv - oof) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        auc = roc_auc_score(tr[f"q{h}"], oofq)
        # residual spread by prediction tercile (heteroscedastic bands)
        res = (yv - oof)
        bins = pd.qcut(oof, 3, labels=False, duplicates="drop")
        lo = res.groupby(bins).quantile(0.1)
        hi = res.groupby(bins).quantile(0.9)
        edges = oof.quantile([1 / 3, 2 / 3]).values
        games_q = tr.loc[tr[f"q{h}"] == 1, f"g{h}"].mean()
        # calibration (v2): held-out forecasts vs what happened. A straight-line shrink toward the mean
        # under-called strong players, and WRs with 1-3 games overreacted, so the fit gets its own slope
        # above the 80th percentile of forecasts and its own slope for short samples. Both extra terms
        # are pulled halfway back to zero (CAL_KEEP) so the correction stays modest.
        mu = float(np.average(yv, weights=w[yv.index]))
        slope = float(np.clip(np.cov(yv, oof)[0, 1] / np.var(oof, ddof=1), 0, 1))
        q80 = float(oof.quantile(0.8))
        few = (healthy.e_games.fillna(0) <= 3).astype(float).values
        def _basis(o, f):
            o = np.asarray(o, float)
            return np.column_stack([np.ones_like(o), o - mu, np.maximum(0, o - q80), f * (o - mu)])
        Bm = _basis(oof, few)
        sw = np.sqrt(w[yv.index].values)
        coef = np.linalg.lstsq(Bm * sw[:, None], yv.values * sw, rcond=None)[0]
        coef[1] = float(np.clip(coef[1], 0, 1))
        coef[2] *= CAL_KEEP; coef[3] *= CAL_KEEP
        calf = lambda o, f: _basis(o, f) @ coef
        CAL_ROWS.append({"position": pos, "horizon": h, "base_slope": round(slope, 3), "slope": round(coef[1], 3),
                         "top_extra": round(coef[2], 3), "few_games": round(coef[3], 3), "q80": round(q80, 2)})
        cal_oof = pd.Series(calf(oof.values, few), index=oof.index)
        res = yv - cal_oof
        OOF_ROWS.append(pd.DataFrame({"pos": pos, "h": h, "season": healthy.season.values, "player_id": healthy.player_id.values,
                                      "y": yv.values, "oof": oof.values, "cal": cal_oof.values, "w": w[yv.index].values}))
        r2_cal = 1 - (res ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        bins = pd.qcut(cal_oof, 3, labels=False, duplicates="drop")
        lo = res.groupby(bins).quantile(0.1)
        hi = res.groupby(bins).quantile(0.9)
        edges = cal_oof.quantile([1 / 3, 2 / 3]).values
        cv_rows.append({"position": pos, "horizon": h, "train_rows": len(tr), "healthy_rows": len(healthy),
                        "r2_ppg": round(r2, 3), "r2_calibrated": round(r2_cal, 3), "slope": round(slope, 3), "auc_healthy": round(auc, 3),
                        "resid_sd": round(res.std(), 2), "games_if_healthy": round(games_q, 2)})
        # final fit and 2026 prediction
        m = HistGradientBoostingRegressor(**reg_kw).fit(healthy[FEATS], healthy[f"y{h}"], sample_weight=w[healthy.index])
        c = HistGradientBoostingClassifier(**reg_kw).fit(tr[FEATS], tr[f"q{h}"], sample_weight=w[tr.index])
        cp = cur[cur.pos == pos].copy()
        if h == 0:
            # rest of 2026: a teammate whose injury is season-ending is gone, so project as if absent
            gone = cp.player_id.isin(SEASON_GONE_MATES)
            cp.loc[gone, ["e_comp_absent", "e_comp_absent_long"]] = 0.0
        ppg = calf(m.predict(cp[FEATS]), (cp.e_games.fillna(0) <= 3).astype(float).values)
        b = np.digitize(ppg, edges)
        out[f"ppg{h}"] = ppg
        out[f"ppg{h}_p10"] = ppg + lo.reindex(b).values
        out[f"ppg{h}_p90"] = ppg + hi.reindex(b).values
        out[f"phealthy{h}"] = c.predict_proba(cp[FEATS])[:, 1]
        out[f"games{h}"] = games_q
    preds.append(out)

pd.concat(OOF_ROWS).to_parquet(f"{OUT}/projection_oof.parquet")
pd.DataFrame(CAL_ROWS).to_csv(f"{OUT}/projection_calibration.csv", index=False)
print(pd.DataFrame(CAL_ROWS).to_string(index=False))
cv = pd.DataFrame(cv_rows)
cv.to_csv(f"{OUT}/projection_cv.csv", index=False)
P = pd.concat(preds)
P.to_csv(f"{OUT}/projections_2026.csv", index=False)
print(cv.to_string(index=False))
print(len(P), "players projected")
