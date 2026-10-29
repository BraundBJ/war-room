"""Stat lists shared by analyze_stats.py and score_2026.py."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from wrconf import ROOT, SEASON, WEEK, REST, LATE_MIN, BACK_BY, LAST_WEEK, PLAYOFF  # noqa: E402
KIND = {  # what each stat measures
    "tgt_pg": "opportunity", "car_pg": "opportunity", "opp_pg": "opportunity", "tgt_share": "opportunity",
    "car_share": "opportunity", "air_share": "opportunity", "wopr": "opportunity", "air_yds_pg": "opportunity",
    "rz_tgt_pg": "opportunity", "rz_car_pg": "opportunity", "i10_opp_pg": "opportunity", "snap_pct": "opportunity",
    "att_pg": "opportunity", "rec_pg": "production", "yds_pg": "production", "td_pg": "production",
    "pass_yds_pg": "production", "qb_rush_yds_pg": "production", "ppg": "production",
    "catch_rate": "efficiency", "yds_per_tgt": "efficiency", "adot": "efficiency", "yac_per_rec": "efficiency",
    "ypc": "efficiency", "rec_epa_per_tgt": "efficiency", "rush_epa_per_car": "efficiency",
    "td_per_opp": "efficiency", "ypa": "efficiency", "cmp_pct": "efficiency", "pass_td_rate": "efficiency",
    "int_rate": "efficiency", "sack_rate": "efficiency", "pass_epa_per_db": "efficiency", "cpoe": "efficiency",
    "pass_adot": "efficiency", "team_plays_pg": "team", "team_pass_rate": "team",
}
FEATS = {
    "RB": ["ppg", "opp_pg", "car_pg", "tgt_pg", "car_share", "tgt_share", "rz_car_pg", "rz_tgt_pg", "i10_opp_pg",
           "snap_pct", "rec_pg", "yds_pg", "td_pg", "ypc", "yds_per_tgt", "catch_rate", "rush_epa_per_car",
           "td_per_opp", "team_plays_pg", "team_pass_rate"],
    "WR": ["ppg", "tgt_pg", "tgt_share", "air_share", "wopr", "air_yds_pg", "rz_tgt_pg", "i10_opp_pg", "snap_pct",
           "rec_pg", "yds_pg", "td_pg", "catch_rate", "yds_per_tgt", "adot", "yac_per_rec", "rec_epa_per_tgt",
           "td_per_opp", "car_pg", "team_plays_pg", "team_pass_rate"],
    "QB": ["ppg", "att_pg", "pass_yds_pg", "car_pg", "qb_rush_yds_pg", "rz_car_pg", "td_pg", "ypa", "cmp_pct",
           "pass_td_rate", "int_rate", "sack_rate", "pass_epa_per_db", "cpoe", "pass_adot", "team_plays_pg",
           "team_pass_rate"],
}
FEATS["TE"] = FEATS["WR"]
