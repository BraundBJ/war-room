"""Player-week table for the weekly projection model: one row per player per regular-season game week,
with features built only from what was known before that game.

Rows: QB/RB/WR/TE who played (or were on the week's injury report) in 2016-2026. 2016 is when the NFL
dropped "Probable", so Questionable means uncertain from then on.

Features
  form      season-to-date PPG, last-3 PPG, last game, prior-season PPG, games so far
  usage     season-to-date and last-3 snap share, target share, carry share, air-yards share, red-zone proxy
  game      Vegas implied team total and spread (nflverse schedules), home, dome, rest days
  opponent  points the opponent allowed to the position so far this season vs league average (prior
            season mixed in early), opponent implied total
  injury    this week's report status (none / Q / D) and final practice status, missed last game, his own
            injury-shortened last game, how many same-position teammates are out this week
Target: Sleeper-default full-PPR points (only games he played).

Usage: python3 -I weekly_data.py RAW_DIR GAMES_CSV OUT_PARQUET
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import sys
import numpy as np
import pandas as pd

RAW, GAMES, OUT = sys.argv[1:4]
POS = ["QB", "RB", "WR", "TE"]
YEARS = range(2015, 2027)  # 2015 only feeds prior-season features


def ppr(d):
    g = lambda c: d[c].fillna(0) if c in d else 0
    return (0.04 * g("passing_yards") + 4 * g("passing_tds") - g("passing_interceptions") + 2 * g("passing_2pt_conversions")
            + 0.1 * g("rushing_yards") + 6 * g("rushing_tds") + 2 * g("rushing_2pt_conversions")
            + g("receptions") + 0.1 * g("receiving_yards") + 6 * g("receiving_tds") + 2 * g("receiving_2pt_conversions")
            + 6 * g("special_teams_tds") - 2 * (g("rushing_fumbles_lost") + g("receiving_fumbles_lost") + g("sack_fumbles_lost")))


ids = pd.read_csv(f"{RAW}/players.csv", low_memory=False)
pfr2gsis = dict(zip(ids.pfr_id, ids.gsis_id))
bd = dict(zip(ids.gsis_id, pd.to_datetime(ids.birth_date, errors="coerce")))

W, INJ = [], []
for y in YEARS:
    w = pd.read_csv(f"{RAW}/week_{y}.csv", low_memory=False)
    w = w[w.season_type == "REG"].copy()
    w["pts"] = ppr(w)
    tm = w.groupby(["team", "week"]).agg(team_tgt=("targets", "sum"), team_car=("carries", "sum"),
                                         team_air=("receiving_air_yards", "sum"), team_pts=("pts", "sum")).reset_index()
    w = w[w.position.isin(POS)].merge(tm, on=["team", "week"], how="left")
    try:
        sn = pd.read_csv(f"{RAW}/snaps_{y}.csv")
        sn = sn[sn.game_type == "REG"].assign(player_id=lambda d: d.pfr_player_id.map(pfr2gsis))
        w = w.merge(sn.groupby(["player_id", "week"]).offense_pct.max().reset_index(), on=["player_id", "week"], how="left")
    except FileNotFoundError:
        w["offense_pct"] = np.nan
    W.append(w[["player_id", "player_display_name", "position", "season", "week", "team", "opponent_team", "pts",
                "targets", "carries", "receiving_air_yards", "attempts", "team_tgt", "team_car", "team_air", "offense_pct"]])
    try:
        i = pd.read_csv(f"{RAW}/injuries_{y}.csv", low_memory=False)
        i = i[(i.game_type == "REG") & i.position.isin(POS)]
        INJ.append(i[["season", "week", "team", "gsis_id", "position", "report_status", "practice_status", "report_primary_injury"]])
    except FileNotFoundError:
        pass
W = pd.concat(W, ignore_index=True).rename(columns={"position": "pos", "player_display_name": "name"})
INJ = pd.concat(INJ, ignore_index=True).rename(columns={"gsis_id": "player_id"}).drop_duplicates(["player_id", "season", "week"])
W["tshare"] = W.targets / W.team_tgt.where(W.team_tgt > 0)
W["cshare"] = W.carries / W.team_car.where(W.team_car > 0)
W["ashare"] = W.receiving_air_yards / W.team_air.where(W.team_air > 0)
W["snap"] = W.offense_pct

# ---- rows: games played plus injury-report weeks he sat (for play-rate), 2016+ ----------------
inj_rows = INJ.merge(W[["player_id", "season", "week"]].assign(played=1), on=["player_id", "season", "week"], how="left")
sat = inj_rows[inj_rows.played.isna() & (inj_rows.season >= 2016)][["player_id", "season", "week", "team", "position"]].rename(columns={"position": "pos"})
R = pd.concat([W.assign(played=1), sat.assign(played=0)], ignore_index=True).sort_values(["player_id", "season", "week"]).reset_index(drop=True)
R = R.merge(INJ[["player_id", "season", "week", "report_status", "practice_status", "report_primary_injury"]], on=["player_id", "season", "week"], how="left")
R["name"] = R.groupby("player_id").name.transform(lambda s: s.ffill().bfill())

# ---- form and usage from strictly earlier games this season ----------------------------------
g = R.groupby(["player_id", "season"])
for c in ["pts", "tshare", "cshare", "ashare", "snap"]:
    played_vals = R[c].where(R.played == 1)
    R[f"std_{c}"] = played_vals.groupby([R.player_id, R.season]).transform(lambda s: s.shift(1).expanding().mean())
    R[f"l3_{c}"] = played_vals.groupby([R.player_id, R.season]).transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
R["last_pts"] = R.pts.where(R.played == 1).groupby([R.player_id, R.season]).transform(lambda s: s.shift(1))
R["games_so_far"] = R.played.groupby([R.player_id, R.season]).transform(lambda s: s.shift(1).fillna(0).cumsum())
R["missed_last"] = (R.played.groupby([R.player_id, R.season]).shift(1) == 0).astype(float)
R["prev_week"] = R.groupby(["player_id", "season"]).week.shift(1)
R["gap"] = (R.week - R.prev_week).fillna(1)
# prior season
full = W.groupby(["player_id", "season"]).agg(ps_ppg=("pts", "mean"), ps_g=("pts", "size"), ps_tshare=("tshare", "mean"), ps_cshare=("cshare", "mean")).reset_index()
full["season"] += 1
R = R.merge(full, on=["player_id", "season"], how="left")
R["age"] = [((pd.Timestamp(f"{s}-09-01") - bd[p]).days / 365.25) if p in bd and pd.notna(bd[p]) else np.nan for p, s in zip(R.player_id, R.season)]

# ---- teammates at the position out this week (weighted by their prior usage) ----------------
use = R[["player_id", "season", "week", "team", "pos", "std_tshare", "std_cshare", "std_pts"]].copy()
use["usage"] = np.where(use.pos == "RB", use.std_cshare, use.std_tshare)
outs = INJ[INJ.report_status.isin(["Out", "Doubtful"])][["player_id", "season", "week"]].merge(
    use.drop_duplicates(["player_id", "season", "week"]), on=["player_id", "season", "week"], how="inner")
o = outs.groupby(["season", "week", "team", "pos"]).agg(mates_out_usage=("usage", "sum"), mates_out_pts=("std_pts", "sum")).reset_index()
R = R.merge(o, on=["season", "week", "team", "pos"], how="left")
R["mates_out_usage"] = R.mates_out_usage.fillna(0)
R["mates_out_pts"] = R.mates_out_pts.fillna(0)
# don't count himself
self_out = R.report_status.isin(["Out", "Doubtful"])
own_use = pd.Series(np.where(R.pos == "RB", R.std_cshare, R.std_tshare), index=R.index).fillna(0)
R.loc[self_out, "mates_out_usage"] = (R.mates_out_usage - own_use)[self_out].clip(lower=0)
R.loc[self_out, "mates_out_pts"] = (R.mates_out_pts - R.std_pts.fillna(0))[self_out].clip(lower=0)

# ---- game context ----------------------------------------------------------------------------
G = pd.read_csv(GAMES)
G = G[(G.game_type == "REG") & G.season.between(2015, 2026)]
home = G.rename(columns={"home_team": "team", "away_team": "opp", "home_rest": "rest"}).assign(home=1)
away = G.rename(columns={"away_team": "team", "home_team": "opp", "away_rest": "rest"}).assign(home=0)
for d, sgn in ((home, 1), (away, -1)):
    d["implied"] = (d.total_line + sgn * d.spread_line) / 2
    d["opp_implied"] = (d.total_line - sgn * d.spread_line) / 2
    d["spread"] = sgn * d.spread_line          # positive = his team favored
GC = pd.concat([home, away])[["season", "week", "team", "opp", "home", "implied", "opp_implied", "spread", "total_line", "rest", "roof"]]
GC["team"] = GC.team.replace({"LA": "LA", "STL": "LA", "SD": "LAC", "OAK": "LV"})
GC["opp"] = GC.opp.replace({"STL": "LA", "SD": "LAC", "OAK": "LV"})
GC["dome"] = GC.roof.isin(["dome", "closed"]).astype(float)
R["team"] = R.team.replace({"STL": "LA", "SD": "LAC", "OAK": "LV"})
R = R.merge(GC.drop(columns="roof"), on=["season", "week", "team"], how="left")

# ---- opponent points allowed to the position, before this week --------------------------------
allow = W.assign(opp=W.opponent_team.replace({"STL": "LA", "SD": "LAC", "OAK": "LV"})).groupby(["season", "week", "opp", "pos"]).pts.sum().reset_index()
allow = allow.sort_values(["opp", "pos", "season", "week"])
allow["opp_allow_std"] = allow.groupby(["opp", "pos", "season"]).pts.transform(lambda s: s.shift(1).expanding().mean())
allow["n"] = allow.groupby(["opp", "pos", "season"]).cumcount()
lg = allow.groupby(["season", "pos"]).pts.mean().rename("lg_allow").reset_index()
prev = allow.groupby(["season", "opp", "pos"]).pts.mean().rename("opp_allow_prev").reset_index().assign(season=lambda d: d.season + 1)
allow = allow.merge(lg, on=["season", "pos"]).merge(prev, on=["season", "opp", "pos"], how="left")
k = allow.n / (allow.n + 4)  # early season leans on last season
allow["opp_allow_rel"] = (k * allow.opp_allow_std.fillna(allow.lg_allow) + (1 - k) * allow.opp_allow_prev.fillna(allow.lg_allow)) / allow.lg_allow
R = R.merge(allow[["season", "week", "opp", "pos", "opp_allow_rel"]], on=["season", "week", "opp", "pos"], how="left")

R["q"] = (R.report_status == "Questionable").astype(float)
R["d"] = (R.report_status == "Doubtful").astype(float)
R["o"] = (R.report_status == "Out").astype(float)
R["prac_dnp"] = R.practice_status.astype(str).str.contains("Did Not", na=False).astype(float)
R["prac_lim"] = R.practice_status.astype(str).str.contains("Limited", na=False).astype(float)
R["prac_full"] = R.practice_status.astype(str).str.contains("Full", na=False).astype(float)
R["hammy"] = R.report_primary_injury.astype(str).str.lower().str.contains("hamstring").astype(float)
R["concussion"] = R.report_primary_injury.astype(str).str.lower().str.contains("concussion").astype(float)
R = R[R.season >= 2016].reset_index(drop=True)
R.to_parquet(OUT)
print(len(R), "player-weeks;", int(R.played.sum()), "played;", R.groupby("season").size().to_dict())
print(R[R.played == 1][["implied", "opp_allow_rel", "std_pts", "l3_snap"]].notna().mean().round(3).to_dict())
