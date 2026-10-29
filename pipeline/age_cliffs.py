"""Age cliffs: empirical aging and drop-off for each position, applied on top of the projections.

The projection model can't learn the cliff on its own: few players reach 33+, the trees can't
extrapolate past the oldest ages in training, and the old players who are left are survivors
(Henry, Kelce), so the model ends up thinking old stars keep going. Here, with 1999-2025 history:

  base  = every fantasy-relevant season (8+ games and a top-48 RB/WR or top-24 QB/TE PPG finish)
  S(k)  = P(8+ game season k years later), with retirement, release or injury all counted as 0.
          Logistic in age (hinges at 27/30/33) plus the player's PPG level, per position and k.
  R(k)  = log(PPG k years later / PPG now) for those who play, same features (linear).

For each projected player and horizon h (2027 = 1 ... 2031 = 5), the empirical number is computed
from his age now and his current level (rest-of-2026 projection). From the position's cliff onset,
healthy chance and PPG are pulled down to the empirical value whenever the model is higher
(phased in over two years). The model is never pushed up. Originals are kept as *_model columns.

Usage: python3 -I age_cliffs.py RAW_DIR PROJECTIONS_CSV
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from wrconf import ROOT, SEASON, WEEK, REST, LATE_MIN, BACK_BY, LAST_WEEK, PLAYOFF  # noqa: E402
import glob
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

RAW, PROJ = sys.argv[1:3]
POS = ["QB", "RB", "WR", "TE"]
TOPN = {"QB": 24, "RB": 48, "WR": 48, "TE": 24}
ONSET = {"RB": 27, "WR": 29, "TE": 29, "QB": 32}  # age (at the horizon season) where the pull starts
MAXK = 5


def ppr(d):
    g = lambda c: d[c].fillna(0) if c in d else 0
    return (0.04 * g("passing_yards") + 4 * g("passing_tds") - g("passing_interceptions") + 2 * g("passing_2pt_conversions")
            + 0.1 * g("rushing_yards") + 6 * g("rushing_tds") + 2 * g("rushing_2pt_conversions")
            + g("receptions") + 0.1 * g("receiving_yards") + 6 * g("receiving_tds") + 2 * g("receiving_2pt_conversions")
            + 6 * g("special_teams_tds")
            - 2 * (g("rushing_fumbles_lost") + g("receiving_fumbles_lost") + g("sack_fumbles_lost")))


rows = []
for f in sorted(glob.glob(f"{RAW}/week_*.csv")):
    y = int(f.split("_")[-1][:4])
    if y > 2025:
        continue
    d = pd.read_csv(f, low_memory=False)
    d = d[(d.season_type == "REG") & d.position.isin(POS)]
    d = d.copy(); d["pts"] = ppr(d)
    rows.append(d.groupby(["player_id", "season"]).agg(pos=("position", "last"), games=("week", "nunique"), pts=("pts", "sum")).reset_index())
S = pd.concat(rows)
S["ppg"] = S.pts / S.games
pl = pd.read_csv(f"{RAW}/players.csv", low_memory=False)
bd = dict(zip(pl.gsis_id, pd.to_datetime(pl.birth_date, errors="coerce")))
S["age"] = [(pd.Timestamp(f"{s}-09-01") - bd[p]).days / 365.25 if p in bd and pd.notna(bd[p]) else np.nan for p, s in zip(S.player_id, S.season)]
S = S[S.age.notna()]
q = S[S.games >= 8].copy()
q["rank"] = q.groupby(["season", "pos"]).ppg.rank(ascending=False)
base = q[q["rank"] <= q.pos.map(TOPN)].copy()
lvl = base.groupby("pos").ppg.median()
idx = S.set_index(["player_id", "season"])


def feats(age, ppg, pos):
    a = np.asarray(age, float)
    return np.column_stack([a, np.maximum(0, a - 27), np.maximum(0, a - 30), np.maximum(0, a - 33),
                            np.log(np.maximum(np.asarray(ppg, float), 1) / lvl[pos])])


models = {}
print("Healthy-season chance k years out for a typical starter (8+ games), by age now")
for pos in POS:
    b = base[base.pos == pos]
    for k in range(1, MAXK + 1):
        bb = b[b.season + k <= 2025]
        nx = idx.reindex(pd.MultiIndex.from_arrays([bb.player_id, bb.season + k]))
        g = nx.games.fillna(0).values
        X = feats(bb.age, bb.ppg, pos)
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, (g >= 8).astype(int))
        ok = g >= 8
        reg = LinearRegression().fit(X[ok], np.log(np.maximum(nx.ppg.values[ok], 1) / bb.ppg.values[ok]))
        models[(pos, k)] = (clf, reg)
    ages = np.arange(23, 39, 2)
    tab = pd.DataFrame({f"k{k}": models[(pos, k)][0].predict_proba(feats(ages, np.full(len(ages), lvl[pos]), pos))[:, 1] for k in range(1, 6)}, index=ages)
    print(pos, f"(n={len(b)})\n", tab.round(2).to_string())

P = pd.read_csv(PROJ)
for h in range(1, MAXK + 1):
    for c in (f"phealthy{h}", f"ppg{h}", f"ppg{h}_p10", f"ppg{h}_p90"):
        if f"{c}_model" not in P:
            P[f"{c}_model"] = P[c]
level = P["ppg0_model"] if "ppg0_model" in P else P["ppg0"]
for h in range(1, MAXK + 1):
    for pos in POS:
        m = (P.pos == pos) & P.age.notna()
        if not m.any():
            continue
        clf, reg = models[(pos, h)]
        X = feats(P.loc[m, "age"], level[m], pos)
        e_ph = clf.predict_proba(X)[:, 1]
        e_ppg = level[m].values * np.exp(reg.predict(X))
        age_h = P.loc[m, "age"].values + h
        w = np.clip((age_h - (ONSET[pos] - 1)) / 2, 0, 1)  # 0 below onset-1, full by onset+1
        ph_m, ppg_m = P.loc[m, f"phealthy{h}_model"].values, P.loc[m, f"ppg{h}_model"].values
        ph = ph_m - w * np.maximum(0, ph_m - e_ph)
        ppg = ppg_m - w * np.maximum(0, ppg_m - e_ppg)
        ratio = np.where(ppg_m > 0, ppg / ppg_m, 1)
        P.loc[m, f"phealthy{h}"] = ph
        P.loc[m, f"ppg{h}"] = ppg
        P.loc[m, f"ppg{h}_p10"] = P.loc[m, f"ppg{h}_p10_model"].values * ratio
        P.loc[m, f"ppg{h}_p90"] = P.loc[m, f"ppg{h}_p90_model"].values * ratio
# Young side: before the cliff onset, later seasons shouldn't fall apart for no age reason.
#  * 2030-2031 healthy chance follows the empirical age curve from 2029 (not a one-year collapse),
#    at every age; the curve itself falls off a cliff for old players.
#  * PPG doesn't drop below last year's level times the empirical one-year change for a player
#    of that age and level who stays healthy (young players hold or grow).
for h in range(2, MAXK + 1):
    for pos in POS:
        m = (P.pos == pos) & P.age.notna()
        age_h = P.loc[m, "age"].values + h
        young = age_h < ONSET[pos] - 1
        clf3, _ = models[(pos, 3)]
        clfh, _ = models[(pos, h)]
        _, reg1 = models[(pos, 1)]
        X = feats(P.loc[m, "age"], level[m], pos)
        if h >= 4:
            s_ratio = clfh.predict_proba(X)[:, 1] / np.maximum(clf3.predict_proba(X)[:, 1], 1e-6)
            floor_ph = P.loc[m, "phealthy3"].values * np.minimum(1, s_ratio)
            ph = P.loc[m, f"phealthy{h}"].values
            P.loc[m, f"phealthy{h}"] = np.maximum(ph, floor_ph)  # any age: the ratio itself carries the cliff
        prev = P.loc[m, f"ppg{h - 1}"].values
        X1 = feats(P.loc[m, "age"].values + h - 1, prev, pos)
        floor_ppg = prev * np.exp(reg1.predict(X1))
        cur = P.loc[m, f"ppg{h}"].values
        new = np.where(young, np.maximum(cur, floor_ppg), cur)
        ratio = np.where(cur > 0, new / cur, 1)
        P.loc[m, f"ppg{h}"] = new
        for c in (f"ppg{h}_p10", f"ppg{h}_p90"):
            P.loc[m, c] = P.loc[m, c].values * ratio
P.to_csv(PROJ, index=False)
show = ["Malik Nabers", "Ashton Jeanty", "Ja'Kobi Lane", "Jeremiyah Love", "Derrick Henry", "George Kittle", "Travis Kelce", "Christian McCaffrey", "Davante Adams", "Josh Jacobs",
        "Aaron Rodgers", "Josh Allen", "Ja'Marr Chase", "Jahmyr Gibbs", "James Cook", "Alvin Kamara"]
pd.set_option("display.width", 250)
for n in show:
    r = P[P.name == n]
    if len(r):
        r = r.iloc[0]
        print(f"{n:22s} {r.age:4.1f}  ph {[round(r[f'phealthy{h}'], 2) for h in range(6)]}  ppg {[round(r[f'ppg{h}'], 1) for h in range(6)]}")
