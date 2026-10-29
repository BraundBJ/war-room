"""Rookie pick values for CTESPN Dynasty (3-round straight rookie drafts), in the page's value units.

1. Slot curves: every straight rookie-draft pick (2019-2023 classes) from the crawl, scored in this
   league's format: season value = max(0, PPG - replacement) x games in each healthy (8+ game)
   season, for career years 1-3. Averaged per slot, then shrunk toward a smooth exponential curve
   in proportion to how few distinct players a slot rests on (1.01 is about four per class), and
   kept from rising with pick number.
2. Pick values: a 2027 pick's rookie plays 2027-2031 (horizons 1-5), a 2028 pick 2028-2031, a 2029
   pick 2029-2031. Career years 4-5 hold the year-3 level (5% off in year 5). Weighted by the same
   contender / rebuilder season weights as players, through 2031.
3. Slots: 2027 picks use a simulated finish (points for so far + projected rest-of-season lineup
   points, with noise; worst team picks first). 2028 and 2029 picks average over all twelve slots.

Usage: python3 -I picks_value.py OUT_JSON
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from wrconf import ROOT, SEASON, WEEK, REST, LATE_MIN, BACK_BY, LAST_WEEK, PLAYOFF  # noqa: E402
import json
import sys
import numpy as np
import pandas as pd

OUT = sys.argv[1]
ROOK = ROOT + "/rookie"
LG = ROOT + "/league/raw"
TEAMS, ROUNDS = 12, 3
st = json.load(open(ROOT + "/stats/proj/actual_values_settings.json"))
REPL = st["replacement_ppg"]
WC = {int(k): v for k, v in st["w_contender"].items()}
WR = {int(k): v for k, v in st["w_rebuilder"].items()}

# ---- historical slot curves ---------------------------------------------------------------------
import os
SAVED = os.environ.get("SLOT_CURVES", ROOT + "/war-room/slot_curves.json")
if not os.path.exists(f"{ROOK}/data/rookie_picks.csv"):
    # no crawl in this workspace: reuse the saved, already-smoothed slot curves
    sv = json.load(open(SAVED))["slots"]
    curve = pd.DataFrame({"players": [x["players"] for x in sv], "s1": [x["y1"] for x in sv],
                          "s2": [x["y2"] for x in sv], "s3": [x["y3"] for x in sv]},
                         index=pd.Index([x["pick"] for x in sv], name="pick"))
    print("using saved slot curves from", SAVED)
else:
    curve = None
if curve is None:
  p = pd.read_csv(f"{ROOK}/data/rookie_picks.csv", dtype={"player_id": str, "draft_id": str})
  o = pd.read_csv(f"{ROOK}/outcomes.csv", dtype={"sleeper_id": str})
  s = pd.read_csv(f"{ROOK}/outcomes_seasons.csv")
  p = p[(p.teams.fillna(TEAMS) == TEAMS) & (p["round"] <= ROUNDS) & (p.type == "linear")]
  p = p.merge(o[["sleeper_id", "gsis_id", "pos", "rookie_class"]], left_on="player_id", right_on="sleeper_id", how="left")
  p["is_rookie"] = p.rookie_class == p.season
  share = p.groupby("draft_id").is_rookie.mean()
  p = p[p.draft_id.isin(share[share >= 0.8].index) & p.is_rookie & p.rookie_class.between(2019, 2023)].copy()
  p["pick"] = p.pick_no
  p["pos"] = p.position.where(p.position.isin(["QB", "RB", "WR", "TE"]), p.pos)
  q = s[s.qual].set_index(["player_id", "season"])
  for k in (1, 2, 3):
      idx = pd.MultiIndex.from_arrays([p.gsis_id, p.rookie_class + k - 1])
      got = q.reindex(idx)
      v = (got.ppg.values - p.pos.map(REPL).values).clip(min=0) * got.games.values
      p[f"v{k}"] = np.nan_to_num(v)


  def pava_dec(y, w):
      y, w = [-v for v in y], list(w)
      blocks = [[y[i], w[i], 1] for i in range(len(y))]
      i = 0
      while i < len(blocks) - 1:
          if blocks[i][0] > blocks[i + 1][0]:
              a, b = blocks[i], blocks[i + 1]
              tw = a[1] + b[1]
              blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / tw, tw, a[2] + b[2]]
              del blocks[i + 1]
              i = max(i - 1, 0)
          else:
              i += 1
      out = []
      for v, _, c in blocks:
          out += [-v] * c
      return np.array(out)


  curve = p.groupby("pick").agg(n=("v1", "size"), players=("player_id", "nunique"),
                                v1=("v1", "mean"), v2=("v2", "mean"), v3=("v3", "mean")).reindex(range(1, TEAMS * ROUNDS + 1))
  # effective distinct players per slot: thousands of leagues taking the same player are one outcome
  w = p.groupby(["pick", "player_id"]).size()
  eff = w.groupby(level=0).apply(lambda x: x.sum() ** 2 / (x ** 2).sum()).reindex(curve.index).fillna(1)
  curve["eff"] = eff
  PRIOR = 10.0  # prior weight, in effective players, of the smooth curve
  x = curve.index.values.astype(float)
  for k in (1, 2, 3):
      y = curve[f"v{k}"].fillna(0).values
      # smooth curve: exponential decay in pick number, fit by weighted least squares on log values
      ok = y > 0
      b = np.polyfit(x[ok], np.log(y[ok]), 1, w=np.sqrt(eff.values[ok]))
      fit = np.exp(np.polyval(b, x))
      shrunk = (eff.values * y + PRIOR * fit) / (eff.values + PRIOR)
      curve[f"s{k}"] = pava_dec(shrunk, eff.clip(lower=1).values)
  print(curve.round(1).head(14).to_string())

# ---- value of a pick at a slot, by draft year -----------------------------------------------------
H = {2027: {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}, 2028: {1: 2, 2: 3, 3: 4, 4: 5}, 2029: {1: 3, 2: 4, 3: 5}}  # career year -> horizon
CAREER_EXT = {4: 1.0, 5: 0.95}  # years 4-5 relative to year 3 (players are 25-26, near peak)


def slot_value(pick, year, W):
    r = curve.loc[pick]
    val = lambda k: r[f"s{k}"] if k <= 3 else r["s3"] * CAREER_EXT[k]
    return float(sum(W.get(h, 0) * val(k) for k, h in H[year].items()))


# ---- simulated 2027 draft order ------------------------------------------------------------------
modes = pd.read_csv(ROOT + "/stats/proj/team_modes.csv", index_col=0)
users = {u["user_id"]: u.get("display_name") for u in json.load(open(f"{LG}/league_1313707182407024640_users.json"))}
rosters = json.load(open(f"{LG}/league_1313707182407024640_rosters.json"))
rid2name = {r["roster_id"]: users.get(r["owner_id"]) for r in rosters}
names = [rid2name[r] for r in sorted(rid2name)]
strength = np.array([modes.at[n, "pf"] + modes.at[n, "proj"] for n in names])
rng = np.random.default_rng(27)
SD = 90.0 * (max(REST, 0.5) / 10) ** 0.5  # rest-of-season lineup points uncertainty (90 at 10 weeks left)
sims = strength[None, :] + rng.normal(0, SD, (20000, len(names)))
ranks = sims.argsort(axis=1).argsort(axis=1) + 1  # 1 = weakest team -> first pick
slot_prob = {n: np.bincount(ranks[:, i], minlength=TEAMS + 1)[1:] / len(sims) for i, n in enumerate(names)}

# ---- all picks and current owners ---------------------------------------------------------------------
owner = {(y, r, rid): rid for y in (2027, 2028, 2029) for r in range(1, ROUNDS + 1) for rid in rid2name}
for t in json.load(open(f"{LG}/league_1313707182407024640_traded_picks.json")):
    y, r = int(t["season"]), int(t["round"])
    if y in H and r <= ROUNDS:
        owner[(y, r, t["roster_id"])] = t["owner_id"]
picks = []
for (y, r, orig), cur in sorted(owner.items()):
    n = rid2name[orig]
    probs = slot_prob[n] if y == 2027 else np.full(TEAMS, 1 / TEAMS)
    picks_at = [(r - 1) * TEAMS + sl for sl in range(1, TEAMS + 1)]
    vc = float(sum(pr * slot_value(pk, y, WC) for pr, pk in zip(probs, picks_at)))
    vr = float(sum(pr * slot_value(pk, y, WR) for pr, pk in zip(probs, picks_at)))
    exp_slot = float(sum(pr * sl for pr, sl in zip(probs, range(1, TEAMS + 1))))
    picks.append({"id": f"pk-{y}-{r}-{orig}", "season": y, "round": r, "orig": n, "owner": rid2name[cur],
                  "expSlot": round(exp_slot, 1), "slotP": [round(float(x), 3) for x in probs],
                  "c": vc, "rb": vr})
slots = [{"pick": int(i), "slot": f"{(i - 1) // TEAMS + 1}.{(i - 1) % TEAMS + 1:02d}",
          "players": int(curve.at[i, "players"] or 0),
          "y1": float(curve.at[i, "s1"]), "y2": float(curve.at[i, "s2"]), "y3": float(curve.at[i, "s3"]),
          "c27": slot_value(i, 2027, WC), "r27": slot_value(i, 2027, WR)} for i in curve.index]
json.dump({"picks": picks, "slots": slots}, open(OUT, "w"))
P = pd.DataFrame(picks)
print(P[P.season == 2027].sort_values("c", ascending=False).head(6)[["id", "orig", "owner", "expSlot", "c", "rb"]].round(1).to_string(index=False))
print(P.groupby(["season", "round"])[["c", "rb"]].mean().round(1).to_string())
