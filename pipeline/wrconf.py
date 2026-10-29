"""Shared settings for the weekly War Room pipeline.

ROOT  working folder (env WR_ROOT); scripts read and write under ROOT/stats, ROOT/league/raw, etc.
WEEK  last completed regular-season week of SEASON: env WR_WEEK, else ROOT/stats/week.json
      (written by run.sh from the nflverse 2026 file), else 7.
Everything week-dependent derives from WEEK.
"""
import json
import os

ROOT = os.environ.get("WR_ROOT", "/home/claude")
SEASON = 2026
LAST_WEEK = 17      # last fantasy week (championship)
PLAYOFF = 15        # first fantasy playoff week


def _week():
    if os.environ.get("WR_WEEK"):
        return int(os.environ["WR_WEEK"])
    try:
        return int(json.load(open(f"{ROOT}/stats/week.json"))["week"])
    except (FileNotFoundError, KeyError, ValueError):
        return 7


WEEK = max(4, min(16, _week()))          # model needs 4+ weeks in; nothing left to project after 16
REST = LAST_WEEK - WEEK                  # fantasy weeks left
LATE_MIN = max(1, min(4, round(0.4 * REST)))   # games that make a "healthy rest of season"
BACK_BY = LAST_WEEK - LATE_MIN + 1       # must be back by this week to get LATE_MIN games
