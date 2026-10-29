"""Score 2026 players: expected rest-of-season PPG from weeks 1-7 stats.

Two ridge models per position, trained on 2015-2025 (4+ games each half):
  ppg_only  : late PPG ~ early PPG          (plain regression to the mean)
  full      : late PPG ~ all leading stats  (analyze_stats.py feature set)
stat_signal = full - ppg_only: what the underlying stats say beyond ordinary regression.
Positive = usage ahead of scoring (buy); negative = scoring ahead of usage (sell).

Usage: python3 -I score_2026.py FEATURES IDMAP_CSV ROSTERS_JSON USERS_JSON OUT_CSV
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from wrconf import ROOT, SEASON, WEEK, REST, LATE_MIN, BACK_BY, LAST_WEEK, PLAYOFF  # noqa: E402
import json
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ROOT + "/stats")
from feats import FEATS  # noqa: E402

FEAT, IDMAP, ROSTERS, USERS, OUT = sys.argv[1:6]
T = pd.read_parquet(FEAT)
ids = pd.read_csv(IDMAP, low_memory=False).dropna(subset=["sleeper_id", "gsis_id"])
ids["sleeper_id"] = ids.sleeper_id.astype(int).astype(str)
g2s = dict(zip(ids.gsis_id, ids.sleeper_id))
users = {u["user_id"]: u.get("display_name") for u in json.load(open(USERS))}
owner = {}
for r in json.load(open(ROSTERS)):
    for pid in r.get("players") or []:
        owner[pid] = users.get(r["owner_id"])

rows = []
for pos, feats in FEATS.items():
    cols = [f"early_{f}" for f in feats]
    tr = T[(T.pos == pos) & T.season.between(2015, 2025) & (T.early_games >= 4) & (T.late_games >= 4)]
    med = tr[cols].median()
    cur = T[(T.pos == pos) & (T.season == 2026) & (T.early_games >= 4)].copy()

    def fit_predict(c):
        sc = StandardScaler().fit(tr[c].fillna(med[c]))
        m = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(sc.transform(tr[c].fillna(med[c])), tr.late_ppg)
        resid_sd = float(np.std(tr.late_ppg - m.predict(sc.transform(tr[c].fillna(med[c])))))
        return m.predict(sc.transform(cur[c].fillna(med[c]))), resid_sd

    base, _ = fit_predict(["early_ppg"])
    full, sd = fit_predict(cols)
    cur["exp_ppg_base"] = base
    cur["exp_ppg"] = full
    cur["stat_signal"] = full - base
    cur["model_sd"] = sd
    rows.append(cur)

S = pd.concat(rows)
S["sleeper_id"] = S.player_id.map(g2s)
S["owner"] = S.sleeper_id.map(owner).fillna("Free agent")
keep = ["name", "pos", "team", "owner", "early_games", "early_ppg", "exp_ppg_base", "exp_ppg", "stat_signal",
        "model_sd", "early_tgt_share", "early_car_share", "early_snap_pct", "early_td_pg", "early_td_per_opp",
        "early_rz_car_pg", "early_rz_tgt_pg", "sleeper_id", "player_id"]
S = S[keep].round(3).sort_values("stat_signal", ascending=False)
S.to_csv(OUT, index=False)
pd.set_option("display.width", 220)
show = ["name", "pos", "team", "owner", "early_games", "early_ppg", "exp_ppg_base", "exp_ppg", "stat_signal"]
R = S[(S.owner != "Free agent") & (S.exp_ppg >= 8)]
print("BUY (usage ahead of scoring), rostered, exp >= 8 PPG\n", R.head(15)[show].to_string(index=False))
print("\nSELL (scoring ahead of usage)\n", R.tail(15)[show].iloc[::-1].to_string(index=False))
print("\nBRADY\n", S[S.owner == "BraundJB"][show].to_string(index=False))
print("\nFREE AGENTS with exp >= 8\n", S[(S.owner == "Free agent") & (S.exp_ppg >= 8)].head(8)[show].to_string(index=False))
