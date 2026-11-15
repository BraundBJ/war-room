"""Daily value snapshot for the value-history charts.

Writes OUT_DIR/YYYY-MM-DD.json (UTC date; a later refresh the same day overwrites it):
{"date", "week", "players": {sleeper id: [contender, rebuilder, market, rest-of-season PPG, next-season PPG]},
 "picks": {pick id: [contender, rebuilder, market]}}
Market is FantasyCalc's dynasty value that day (null when it has none).

Usage: python3 -I snapshot.py SITE_DATA_JSON OUT_DIR
"""
import datetime
import json
import os
import sys

SRC, OUT = sys.argv[1:3]
D = json.load(open(SRC))
mk = D.get("market") or {}
r1 = lambda x: None if x is None else round(x, 1)
snap = {"date": datetime.datetime.now(datetime.timezone.utc).date().isoformat(), "week": D["meta"].get("throughWeek"),
        "players": {p["id"]: [p.get("c"), p.get("rb"), mk.get(p["id"]), r1((p.get("proj") or [None])[0]),
                              r1((p.get("proj") or [None, None])[1])] for p in D["players"]},
        "picks": {k["id"]: [k.get("c"), k.get("rb"), mk.get(k["n"].split(" (")[0])] for k in D.get("picks", [])}}
os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, snap["date"] + ".json")
json.dump(snap, open(path, "w"), separators=(",", ":"))
print(path, len(snap["players"]), "players", round(os.path.getsize(path) / 1024), "KB")
