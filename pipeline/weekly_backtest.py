"""Leave-one-season-out test of the weekly model vs simple baselines (played games, fantasy-relevant players)."""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

R = pd.read_parquet(sys.argv[1])
FEATS = ["std_pts", "l3_pts", "last_pts", "ps_ppg", "games_so_far", "std_tshare", "l3_tshare", "std_cshare", "l3_cshare",
         "std_ashare", "l3_ashare", "std_snap", "l3_snap", "ps_tshare", "ps_cshare", "implied", "opp_implied", "spread",
         "total_line", "home", "dome", "rest", "opp_allow_rel", "q", "prac_dnp", "prac_lim", "prac_full", "hammy",
         "concussion", "missed_last", "gap", "mates_out_usage", "mates_out_pts", "age", "pos_c"]
R["pos_c"] = R.pos.map({"QB": 0, "RB": 1, "WR": 2, "TE": 3})
D = R[(R.played == 1) & R.pts.notna() & (R.season <= 2025)].copy()
# who we care about: players with an established role (5+ PPG this season so far, or last season if early)
base = D.std_pts.where(D.games_so_far >= 3, D.ps_ppg.combine_first(D.std_pts))
D = D[base >= 5].copy()
D["b_std"] = D.std_pts
D["b_l3"] = D.l3_pts
k = D.games_so_far / (D.games_so_far + 4)
D["b_blend"] = (k * D.std_pts.fillna(D.ps_ppg) + (1 - k) * D.ps_ppg.fillna(D.std_pts))
rows, preds = [], []
for s in range(2018, 2026):
    tr, te = D[(D.season != s) & (D.season >= 2016)], D[D.season == s]
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                      l2_regularization=1.0, loss="squared_error", random_state=0)
    m.fit(tr[FEATS], tr.pts)
    p = pd.Series(m.predict(te[FEATS]), index=te.index)
    preds.append(p)
    for name, x in [("model", p), ("season avg", te.b_std), ("last 3", te.b_l3), ("season avg + last year", te.b_blend)]:
        ok = x.notna()
        rows.append({"season": s, "method": name, "mae": (te.pts[ok] - x[ok]).abs().mean(), "r": np.corrcoef(te.pts[ok], x[ok])[0, 1], "n": ok.sum()})
D["pred"] = pd.concat(preds)
T = pd.DataFrame(rows)
print(T.groupby("method")[["mae", "r"]].mean().round(3).sort_values("mae").to_string())
print(T.pivot(index="season", columns="method", values="mae").round(2).to_string())
E = D[D.pred.notna()]
# start/sit test: within each position-week, does the model order players better? (rank correlation)
def rk(g, c):
    return g[[c, "pts"]].corr(method="spearman").iloc[0, 1] if len(g) > 8 else np.nan
for c in ["pred", "b_blend", "b_std"]:
    print("rank corr within position-week", c, round(E.groupby(["season", "week", "pos"]).apply(lambda g: rk(g, c)).mean(), 3))
# Questionable games: how biased is each method?
for nm, m in [("healthy", E.q == 0), ("Questionable", E.q == 1), ("Q + did not practice", (E.q == 1) & (E.prac_dnp == 1))]:
    x = E[m]
    print(f"{nm:22s} n={len(x):5d}  actual {x.pts.mean():5.2f}  model {x.pred.mean():5.2f}  season avg+last yr {x.b_blend.mean():5.2f}")
pd.to_pickle(FEATS, sys.argv[2]) if len(sys.argv) > 2 else None
