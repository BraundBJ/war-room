"""Pack the daily value snapshots (history branch, values/*.json) into one file the page fetches.

Keeps every day of the last 60 days and one day a week before that. Players are kept if any of their
values ever reached 100.

Writes {"dates": [...], "p": {sleeper id: [[contender...], [rebuilder...], [market...]]}, "k": {pick id: same}}

Usage: python3 -I history_pack.py VALUES_DIR OUT_JSON
"""
import datetime
import glob
import json
import os
import sys

SRC, OUT = sys.argv[1:3]
files = sorted(glob.glob(os.path.join(SRC, "*.json")))
snaps = [json.load(open(f)) for f in files]
if snaps:
    last = datetime.date.fromisoformat(snaps[-1]["date"])
    keep = []
    for i, s in enumerate(snaps):
        d = datetime.date.fromisoformat(s["date"])
        if (last - d).days <= 60 or d.weekday() == 1 or i == 0:  # Tuesdays before that (full rebuild day)
            keep.append(s)
    snaps = keep
dates = [s["date"] for s in snaps]


def pack(key):
    ids = {i for s in snaps for i in s[key]}
    out = {}
    for i in ids:
        rows = [s[key].get(i) or [None] * 3 for s in snaps]
        cols = [[None if r[j] is None else int(round(r[j])) for r in rows] for j in range(3)]
        if max((v for c in cols for v in c if v is not None), default=0) >= 100:
            out[i] = cols
    return out


H = {"dates": dates, "p": pack("players"), "k": pack("picks")}
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
json.dump(H, open(OUT, "w"), separators=(",", ":"))
print(OUT, len(dates), "days", len(H["p"]), "players", round(os.path.getsize(OUT) / 1024), "KB")
