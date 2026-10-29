#!/usr/bin/env bash
# Weekly War Room build: download data, rerun the model, write site/index.html.
# Usage: WR_ROOT=/path/to/work ./run.sh      (WR_WEEK=N to force a week; SKIP_DOWNLOAD=1 to reuse data)
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
export WR_ROOT="${WR_ROOT:-$REPO/work}"
L=1313707182407024640
S="$WR_ROOT/stats"; R="$WR_ROOT/league/raw"; IDS="$WR_ROOT/rookie/raw/db_playerids.csv"
mkdir -p "$S/raw" "$S/proj" "$S/results" "$R" "$WR_ROOT/rookie/raw" "$WR_ROOT/rookie/results" "$WR_ROOT/war-room" "$WR_ROOT/site"
cp "$REPO"/pipeline/*.py "$S/"
cp "$REPO/data/slot_curves.json" "$WR_ROOT/war-room/"
cp "$REPO/data/rookie_by_slot.csv" "$WR_ROOT/rookie/results/by_slot.csv"
cp "$REPO/data/site_study.json" "$WR_ROOT/rookie/results/"

get() { [ -s "$2" ] && [ "${3:-}" != fresh ] && return 0; curl -sSfL --retry 3 -o "$2.tmp" "$1" && mv "$2.tmp" "$2" || { rm -f "$2.tmp"; echo "skip $1"; }; }
if [ "${SKIP_DOWNLOAD:-0}" != 1 ]; then
  B=https://github.com/nflverse/nflverse-data/releases/download
  cd "$S/raw"
  for y in $(seq 1999 2026); do f=""; [ "$y" = 2026 ] && f=fresh
    get "$B/stats_player/stats_player_week_$y.csv" "week_$y.csv" $f; done
  for y in $(seq 2012 2026); do f=""; [ "$y" = 2026 ] && f=fresh
    [ "$y" -ge 2013 ] && get "$B/snap_counts/snap_counts_$y.csv" "snaps_$y.csv" $f
    get "$B/pbp/play_by_play_$y.parquet" "pbp_$y.parquet" $f
    get "$B/injuries/injuries_$y.csv" "injuries_$y.csv" $f
    get "$B/depth_charts/depth_charts_$y.csv" "depth_$y.csv" $f
  done
  get "$B/players/players.csv" players.csv fresh
  get https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv "$IDS" fresh
  cd "$R"; A=https://api.sleeper.app/v1
  for p in league/$L league/$L/users league/$L/rosters league/$L/traded_picks league/$L/drafts state/nfl; do
    curl -sSf -A war-room -o "$(echo $p | tr / _).json" "$A/$p"; done
  curl -sSf -A war-room -o players_nfl.json "$A/players/nfl"
  for w in $(seq 0 18); do curl -sSf -A war-room -o "tx_$w.json" "$A/league/$L/transactions/$w"; done
fi

# last completed regular-season week of 2026
cd "$S"
python3 - <<'PY'
import json, os, pandas as pd
S = os.environ["WR_ROOT"] + "/stats"
d = pd.read_csv(f"{S}/raw/week_2026.csv", usecols=["season_type", "week"])
w = int(os.environ.get("WR_WEEK") or d[d.season_type == "REG"].week.max())
json.dump({"week": w}, open(f"{S}/week.json", "w")); print("week", w)
PY

python3 build_features.py raw features.parquet
python3 score_2026.py features.parquet "$IDS" "$R/league_${L}_rosters.json" "$R/league_${L}_users.json" results/scores_2026.csv
python3 injury_returns.py raw features.parquet "$R/players_nfl.json" "$IDS" proj/injury_outlook.json
python3 project.py features.parquet raw/players.csv proj
python3 extend_horizons.py features.parquet raw/players.csv proj/projections_2026.csv
python3 -W ignore age_cliffs.py raw proj/projections_2026.csv
python3 value.py proj/projections_2026.csv features.parquet "$IDS" "$R/league_${L}_rosters.json" "$R/league_${L}_users.json" proj/actual_values.csv
python3 team_value.py proj/actual_values.csv proj/actual_values_settings.json "$R/league_${L}_rosters.json" "$R/league_${L}_users.json" proj
python3 picks_value.py proj/picks.json
python3 build_site_data.py "$WR_ROOT/site/data.json"
REPO_DIR="$REPO" python3 - <<'PY'
import os
W, REPO = os.environ["WR_ROOT"], os.environ.get("REPO_DIR", ".")
s = open(f"{REPO}/site/page.src.html").read()
d = open(f"{W}/site/data.json").read().replace("</", "<\\/")
os.makedirs(f"{REPO}/public", exist_ok=True)
html = '<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">\n' + s.replace("__DATA__", d) + "\n</html>\n"
open(f"{REPO}/public/index.html", "w").write(html)
print("wrote public/index.html", round(len(html) / 1024), "KB")
PY
