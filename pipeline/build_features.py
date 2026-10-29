"""Player-split feature table for the "which stats lead to points" analysis.

For every season 2015-2026 (regular season), aggregates each skill player's stats over a
set of weeks into per-game rates, shares and efficiency measures. Splits produced:
  early  = weeks 1-7   (matches where 2026 stands)
  late   = weeks 8+    (the target: rest-of-season PPG)
  odd / even weeks     (for stability)
Scoring: Sleeper-default full PPR (pass TD 4, INT -1, fumble lost -2), no TE premium.

Usage: python3 -I build_features.py RAW_DIR OUT_PARQUET
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from wrconf import ROOT, SEASON, WEEK, REST, LATE_MIN, BACK_BY, LAST_WEEK, PLAYOFF  # noqa: E402
import sys
import numpy as np
import pandas as pd

RAW, OUT = sys.argv[1], sys.argv[2]
SEASONS = range(2012, 2027)
EARLY_LAST = WEEK  # first-half / rest-of-season split (wrconf)
POS = ["QB", "RB", "WR", "TE"]

ids = pd.read_csv(f"{RAW}/players.csv", usecols=["gsis_id", "pfr_id"]).dropna()
pfr2gsis = dict(zip(ids.pfr_id, ids.gsis_id))


def sleeper_ppr(w):
    g = lambda c: w[c].fillna(0) if c in w else 0
    return (0.04 * g("passing_yards") + 4 * g("passing_tds") - g("passing_interceptions")
            + 2 * g("passing_2pt_conversions") + 0.1 * g("rushing_yards") + 6 * g("rushing_tds")
            + 2 * g("rushing_2pt_conversions") + g("receptions") + 0.1 * g("receiving_yards")
            + 6 * g("receiving_tds") + 2 * g("receiving_2pt_conversions") + 6 * g("special_teams_tds")
            - 2 * (g("rushing_fumbles_lost") + g("receiving_fumbles_lost") + g("sack_fumbles_lost")))


frames = []
for y in SEASONS:
    w = pd.read_csv(f"{RAW}/week_{y}.csv", low_memory=False)
    w = w[(w.season_type == "REG") & w.position.isin(POS)].copy()
    w["pts"] = sleeper_ppr(w)

    # play-by-play: team volume and red-zone usage per week
    pbp = pd.read_parquet(f"{RAW}/pbp_{y}.parquet", columns=[
        "season_type", "week", "posteam", "yardline_100", "pass_attempt", "rush_attempt",
        "two_point_attempt", "receiver_player_id", "rusher_player_id", "sack"])
    pbp = pbp[(pbp.season_type == "REG") & (pbp.two_point_attempt.fillna(0) == 0)]
    plays = pbp[(pbp.pass_attempt == 1) | (pbp.rush_attempt == 1)]
    team = plays.groupby(["posteam", "week"]).agg(
        team_plays=("week", "size"), team_pass=("pass_attempt", "sum"),
        team_targets=("receiver_player_id", lambda s: s.notna().sum()),
        team_carries=("rush_attempt", "sum")).reset_index().rename(columns={"posteam": "team"})
    rz = plays[plays.yardline_100 <= 20]
    i10 = plays[plays.yardline_100 <= 10]
    rz_t = rz.groupby(["receiver_player_id", "week"]).size().rename("rz_tgt")
    rz_c = rz[rz.rush_attempt == 1].groupby(["rusher_player_id", "week"]).size().rename("rz_car")
    i10_t = i10.groupby(["receiver_player_id", "week"]).size().rename("i10_tgt")
    i10_c = i10[i10.rush_attempt == 1].groupby(["rusher_player_id", "week"]).size().rename("i10_car")
    for s_ in (rz_t, rz_c, i10_t, i10_c):
        s_.index.names = ["player_id", "week"]
        w = w.merge(s_.reset_index(), on=["player_id", "week"], how="left")
    w = w.merge(team, on=["team", "week"], how="left")

    try:
        sn = pd.read_csv(f"{RAW}/snaps_{y}.csv")
        assert "offense_pct" in sn
    except Exception:  # snap counts start in 2013
        sn = pd.DataFrame(columns=["game_type", "pfr_player_id", "week", "offense_pct"])
    sn = sn[sn.game_type == "REG"].assign(player_id=lambda d: d.pfr_player_id.map(pfr2gsis))
    sn = sn.dropna(subset=["player_id"]).groupby(["player_id", "week"]).offense_pct.max().reset_index()
    w = w.merge(sn, on=["player_id", "week"], how="left")
    frames.append(w)

W = pd.concat(frames, ignore_index=True)
for c in ["rz_tgt", "rz_car", "i10_tgt", "i10_car"]:
    W[c] = W[c].fillna(0)

# --- games cut short by injury -----------------------------------------------------------------
# Flag a game when the player's snap share fell below half his season norm AND he then missed his
# team's next game or was on its injury report. Flagged games are dropped from every average.
inj = pd.concat([pd.read_csv(f"{RAW}/injuries_{y}.csv", low_memory=False) for y in SEASONS], ignore_index=True)
inj = inj[(inj.game_type == "REG") & inj.report_status.isin(["Out", "Doubtful", "Questionable"])]
on_report = set(zip(inj.gsis_id, inj.season, inj.week))
team_weeks = W.groupby(["team", "season"]).week.apply(lambda s: sorted(set(s))).to_dict()
played = set(zip(W.player_id, W.season, W.week))
med = W.groupby(["player_id", "season"]).offense_pct.transform("median")


def next_team_week(team, season, week):
    ws = team_weeks.get((team, season), [])
    later = [w for w in ws if w > week]
    return later[0] if later else None


nxt = [next_team_week(t, s_, w_) for t, s_, w_ in zip(W.team, W.season, W.week)]
W["next_wk"] = nxt
missed_next = [(n is not None) and ((pid, s_, n) not in played) for pid, s_, n in zip(W.player_id, W.season, W.next_wk)]
listed_next = [(n is not None) and ((pid, s_, n) in on_report) for pid, s_, n in zip(W.player_id, W.season, W.next_wk)]
W["left_early"] = (W.offense_pct < 0.5 * med) & (med >= 0.4) & (np.array(missed_next) | np.array(listed_next))
print("games flagged as cut short by injury:", int(W.left_early.sum()), "of", len(W))
W_all = W.copy()
W = W[~W.left_early].copy()

# --- depth chart rank around week 7 ------------------------------------------------------------
def depth_rank(y):
    d = pd.read_csv(f"{RAW}/depth_{y}.csv", low_memory=False)
    if "depth_team" in d:  # weekly format through 2024
        d = d[(d.formation == "Offense") & d.depth_position.isin(["QB", "RB", "HB", "WR", "TE"]) & (d.week <= EARLY_LAST)]
        d = d[d.week == d.groupby("club_code").week.transform("max")]
        d["rank"] = pd.to_numeric(d.depth_team, errors="coerce")
    else:  # daily snapshots from 2025
        d = d[d.pos_abb.isin(["QB", "RB", "WR", "TE"])].copy()
        d["day"] = pd.to_datetime(d.dt.str[:10])
        start = pd.Timestamp(f"{y}-09-04") if y == 2025 else pd.Timestamp(f"{y}-09-09")
        d = d[d.day <= start + pd.Timedelta(days=7 * EARLY_LAST + 6)]
        d = d[d.day == d.day.max()]
        d["rank"] = pd.to_numeric(d.pos_rank, errors="coerce")
    return d.groupby("gsis_id")["rank"].min()


DEPTH = {}
for y in SEASONS:
    try:
        DEPTH[y] = depth_rank(y)
    except FileNotFoundError:
        DEPTH[y] = pd.Series(dtype=float)


def aggregate(df, label):
    g = df.groupby(["player_id", "season"])
    s = lambda c: g[c].sum()
    a = pd.DataFrame({
        "name": g.player_display_name.last(), "pos": g.position.last(), "team": g.team.last(),
        "games": g.week.nunique(), "pts": s("pts"),
        "targets": s("targets"), "receptions": s("receptions"), "rec_yds": s("receiving_yards"),
        "rec_td": s("receiving_tds"), "air_yds": s("receiving_air_yards"), "yac": s("receiving_yards_after_catch"),
        "rec_epa": s("receiving_epa"), "carries": s("carries"), "rush_yds": s("rushing_yards"),
        "rush_td": s("rushing_tds"), "rush_epa": s("rushing_epa"),
        "att": s("attempts"), "cmp": s("completions"), "pass_yds": s("passing_yards"),
        "pass_td": s("passing_tds"), "ints": s("passing_interceptions"), "sacks": s("sacks_suffered"),
        "pass_epa": s("passing_epa"), "pass_ay": s("passing_air_yards"), "cpoe": g.passing_cpoe.mean(),
        "rz_tgt": s("rz_tgt"), "rz_car": s("rz_car"), "i10_tgt": s("i10_tgt"), "i10_car": s("i10_car"),
        "team_targets": s("team_targets"), "team_carries": s("team_carries"),
        "team_plays": s("team_plays"), "team_pass": s("team_pass"),
        "air_share_wk": g.air_yards_share.mean(), "wopr_wk": g.wopr.mean(), "snap_pct": g.offense_pct.mean(),
    })
    n = a.games
    safe = lambda x, d: np.where(d > 0, x / d.where(d > 0, 1), np.nan)
    f = pd.DataFrame(index=a.index)
    f["games"] = n
    f["ppg"] = a.pts / n
    # opportunity
    f["tgt_pg"] = a.targets / n
    f["car_pg"] = a.carries / n
    f["opp_pg"] = (a.targets + a.carries) / n
    f["tgt_share"] = safe(a.targets, a.team_targets)
    f["car_share"] = safe(a.carries, a.team_carries)
    f["air_share"] = a.air_share_wk
    f["wopr"] = a.wopr_wk
    f["air_yds_pg"] = a.air_yds / n
    f["rz_tgt_pg"] = a.rz_tgt / n
    f["rz_car_pg"] = a.rz_car / n
    f["i10_opp_pg"] = (a.i10_tgt + a.i10_car) / n
    f["snap_pct"] = a.snap_pct
    # efficiency
    f["catch_rate"] = safe(a.receptions, a.targets)
    f["yds_per_tgt"] = safe(a.rec_yds, a.targets)
    f["adot"] = safe(a.air_yds, a.targets)
    f["yac_per_rec"] = safe(a.yac, a.receptions)
    f["ypc"] = safe(a.rush_yds, a.carries)
    f["rec_epa_per_tgt"] = safe(a.rec_epa, a.targets)
    f["rush_epa_per_car"] = safe(a.rush_epa, a.carries)
    f["td_per_opp"] = safe(a.rec_td + a.rush_td, a.targets + a.carries)
    f["td_pg"] = (a.rec_td + a.rush_td + a.pass_td) / n
    f["yds_pg"] = (a.rec_yds + a.rush_yds) / n
    f["rec_pg"] = a.receptions / n
    # quarterback
    f["att_pg"] = a.att / n
    f["pass_yds_pg"] = a.pass_yds / n
    f["ypa"] = safe(a.pass_yds, a.att)
    f["cmp_pct"] = safe(a.cmp, a.att)
    f["pass_td_rate"] = safe(a.pass_td, a.att)
    f["int_rate"] = safe(a.ints, a.att)
    f["sack_rate"] = safe(a.sacks, a.att + a.sacks)
    f["pass_epa_per_db"] = safe(a.pass_epa, a.att + a.sacks)
    f["cpoe"] = a.cpoe
    f["pass_adot"] = safe(a.pass_ay, a.att)
    f["qb_rush_yds_pg"] = a.rush_yds / n
    # team environment
    f["team_plays_pg"] = a.team_plays / n
    f["team_pass_rate"] = safe(a.team_pass, a.team_plays)
    f = f.add_prefix(f"{label}_")
    meta = a[["name", "pos", "team"]]
    return meta, f


E = W[W.week <= EARLY_LAST]
# each player's last three games inside weeks 1-7: catches mid-season role changes
recent = E[E.groupby(["player_id", "season"]).week.rank(ascending=False, method="first") <= 3]
parts = {
    "early": E, "late": W[W.week > EARLY_LAST], "recent": recent,
    "odd": W[W.week % 2 == 1], "even": W[W.week % 2 == 0], "full": W,
}
meta = None
out = []
for label, df in parts.items():
    m, f = aggregate(df, label)
    meta = m if meta is None else meta.combine_first(m)
    out.append(f)
T = pd.concat([meta] + out, axis=1).copy().reset_index()

# --- role context at week 7: absent teammates, time since last game, depth chart --------------
E_all = W_all[W_all.week <= EARLY_LAST]
last = E_all.groupby(["player_id", "season"]).agg(last_week=("week", "max"), team_last=("team", "last"))
T = T.join(last, on=["player_id", "season"])
tw7 = {k: [w for w in v if w <= EARLY_LAST] for k, v in team_weeks.items()}
T["early_wks_since_last"] = [len([w for w in tw7.get((t, s_), []) if w > lw]) if pd.notna(lw) else np.nan
                             for t, s_, lw in zip(T.team_last, T.season, T.last_week)]
role = np.where(T.pos == "RB", T.early_car_share, np.where(T.pos == "QB", np.nan, T.early_tgt_share))
T["role_share"] = role
absent = (T.early_wks_since_last >= 2) & (T.early_games >= 2)
comp = []
grp = T.groupby(["season", "team_last", "pos"])
abs_sum = (T.role_share.where(absent, 0)).groupby([T.season, T.team_last, T.pos]).transform("sum")
T["early_comp_absent"] = abs_sum - T.role_share.where(absent, 0)
# the same, counting only teammates already out 3+ games (long absences tend to continue)
long_abs = (T.early_wks_since_last >= 3) & (T.early_games >= 2)
long_sum = (T.role_share.where(long_abs, 0)).groupby([T.season, T.team_last, T.pos]).transform("sum")
T["early_comp_absent_long"] = long_sum - T.role_share.where(long_abs, 0)
T["early_depth_rank"] = [DEPTH.get(s_, pd.Series(dtype=float)).get(pid, np.nan) for pid, s_ in zip(T.player_id, T.season)]
T = T.drop(columns=["role_share", "team_last", "last_week"])
T.to_parquet(OUT, index=False)
# weekly usage table for the page's player trend charts
wk = W_all[W_all.season >= 2025][["player_id", "season", "week", "team", "pts", "targets", "carries", "team_targets",
                              "team_carries", "offense_pct", "left_early"]].copy()
wk.to_parquet(OUT.replace(".parquet", "_weekly.parquet"), index=False)
print(T.shape)
print(T.groupby("season").size().to_string())
