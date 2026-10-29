"""Team value for every roster in the league, plus a two-sided trade checker.

Each simulated week, a player plays with probability P(healthy season) x games-if-healthy /
weeks. The best legal lineup is set (1QB, 2RB, 2WR, 1TE, 3FLEX of RB/WR/TE); an empty slot is
filled from waivers at replacement PPG. A player's value to a team = expected lineup points with
him minus without (for his own team) or with minus current (for anyone else). Trades are scored
on whole rosters, so 2-for-1s are judged by the lineup they produce. Common random numbers keep
comparisons stable across scenarios.

Horizons: rest of 2026 (10 weeks; taxi players can't start), then 2027-2031 (17 weeks each).
Each team is labeled contender or rebuilder (current points for + projected rest-of-2026 lineup),
and its own value uses that mode's weights (value.py): contender 1.0/0.5/0.25/0.1,
rebuilder 0.1/0.7/1.0/0.8.

Usage:  python3 -I team_value.py VALUES_CSV SETTINGS_JSON ROSTERS_JSON USERS_JSON OUT_DIR
Import: from team_value import League; League(...).trade("BraundJB", [...], "ryanswee", [...])
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from wrconf import ROOT, SEASON, WEEK, REST, LATE_MIN, BACK_BY, LAST_WEEK, PLAYOFF  # noqa: E402
import json
import sys
import zlib
import numpy as np
import pandas as pd

SIMS = 3000
WEEKS = {0: REST, 1: 17, 2: 17, 3: 17, 4: 17, 5: 17}
SLOTS = [("QB", 1), ("RB", 2), ("WR", 2), ("TE", 1)]
FLEX_N, FLEX_POS = 3, ("RB", "WR", "TE")


class League:
    def __init__(self, values_csv, settings_json, rosters_json, users_json):
        st = json.load(open(settings_json))
        self.repl = st["replacement_ppg"]
        self.w = {"contender": {int(k): v for k, v in st["w_contender"].items()},
                  "rebuilder": {int(k): v for k, v in st["w_rebuilder"].items()}}
        V = pd.read_csv(values_csv, dtype={"sleeper_id": str})
        self.V = V[V.sleeper_id.notna()].drop_duplicates("sleeper_id").set_index("sleeper_id")
        users = {u["user_id"]: u.get("display_name") for u in json.load(open(users_json))}
        self.teams, self.taxi, self.pf = {}, set(), {}
        for r in json.load(open(rosters_json)):
            name = users.get(r["owner_id"], f"roster{r['roster_id']}")
            self.teams[name] = [p for p in (r.get("players") or []) if p in self.V.index]
            self.taxi |= set(r.get("taxi") or [])
            s = r.get("settings") or {}
            self.pf[name] = s.get("fpts", 0) + s.get("fpts_decimal", 0) / 100
        self.byname = {self.V.at[p, "name"]: p for p in self.V.index}
        self._draws = {}
        self._cache = {}
        self.base = {t: self.season_points(ps) for t, ps in self.teams.items()}
        self.mode = self._modes()

    # --- simulation -------------------------------------------------------------------------
    def _u(self, pid, h):
        k = (pid, h)
        if k not in self._draws:
            self._draws[k] = np.random.default_rng(zlib.crc32(f"{pid}-{h}".encode())).random(SIMS)
        return self._draws[k]

    def _z(self, pid, h):
        k = (pid, h, "z")
        if k not in self._draws:
            self._draws[k] = np.random.default_rng(zlib.crc32(f"{pid}-{h}-z".encode())).standard_normal(SIMS)
        return self._draws[k]

    def lineup_points(self, pids, h):
        key = (frozenset(pids), h)
        if key in self._cache:
            return self._cache[key]
        V, repl = self.V, self.repl
        if h == 0:
            pids = [p for p in pids if p not in self.taxi]
        n = len(pids)
        mu = np.array([V.at[p, f"ppg{h}"] for p in pids], float)
        sd = np.array([V.at[p, f"sd{h}"] for p in pids], float)
        # each simulated season draws the player's PPG from his projected range (common random numbers)
        Z = np.column_stack([self._z(p, h) for p in pids]) if n else np.zeros((SIMS, 0))
        ppg = np.maximum(mu + sd * Z, 0)
        pos = np.array([V.at[p, "pos"] for p in pids], dtype=object)
        p_play = np.array([min(1.0, V.at[p, f"phealthy{h}"] * V.at[p, f"games{h}"] / WEEKS[h]) for p in pids])
        avail = (np.column_stack([self._u(p, h) for p in pids]) < p_play) if n else np.zeros((SIMS, 0), bool)
        score = np.where(avail, ppg, -np.inf)
        total = np.zeros(SIMS)
        used = np.zeros((SIMS, n), bool)
        rows = np.arange(SIMS)
        for P_, k_need in SLOTS:
            cols = np.where(pos == P_)[0]
            if len(cols):
                order = np.argsort(-score[:, cols], axis=1)
            for k in range(k_need):
                if len(cols) > k:
                    c = cols[order[:, k]]
                    v = score[rows, c]
                    ok = np.isfinite(v)
                    total += np.where(ok, np.maximum(v, repl[P_]), repl[P_])
                    used[rows[ok], c[ok]] = True
                else:
                    total += repl[P_]
        fcols = np.where(np.isin(pos, FLEX_POS))[0]
        flex_repl = max(repl[p] for p in FLEX_POS)
        if len(fcols):
            s = np.sort(np.where(used[:, fcols], -np.inf, score[:, fcols]), axis=1)[:, ::-1]
        for k in range(FLEX_N):
            v = s[:, k] if len(fcols) > k else np.full(SIMS, -np.inf)
            total += np.where(np.isfinite(v), np.maximum(v, flex_repl), flex_repl)
        out = float(total.mean())
        self._cache[key] = out
        return out

    def season_points(self, pids):
        return {h: self.lineup_points(list(pids), h) * WEEKS[h] for h in WEEKS}

    def weigh(self, d, mode):
        return sum(self.w[mode][h] * d[h] for h in WEEKS)

    # --- team modes ---------------------------------------------------------------------------
    def _modes(self):
        df = pd.DataFrame({"pf": self.pf, "proj": {t: b[0] for t, b in self.base.items()}})
        z = lambda s: (s - s.mean()) / s.std(ddof=0)
        df["score"] = z(df.pf) + z(df.proj)
        df["mode"] = np.where(df.score.rank(ascending=False) <= 6, "contender", "rebuilder")
        self.mode_table = df.sort_values("score", ascending=False)
        return df["mode"].to_dict()

    # --- values -------------------------------------------------------------------------------
    def value_to(self, team, pid):
        roster = self.teams[team]
        if pid in roster:
            alt = self.season_points([p for p in roster if p != pid])
            d = {h: self.base[team][h] - alt[h] for h in WEEKS}
        else:
            alt = self.season_points(roster + [pid])
            d = {h: alt[h] - self.base[team][h] for h in WEEKS}
        return d

    def trade(self, team_a, give_a, team_b, give_b):
        """give_a: players team_a sends (names or ids); give_b: players team_b sends."""
        ids = lambda xs: [self.byname.get(x, x) for x in xs]
        ga, gb = ids(give_a), ids(give_b)
        res = {}
        for t, out, inn in ((team_a, ga, gb), (team_b, gb, ga)):
            after = [p for p in self.teams[t] if p not in out] + inn
            a = self.season_points(after)
            d = {h: a[h] - self.base[t][h] for h in WEEKS}
            res[t] = {"mode": self.mode[t], "by_season": {h: round(d[h], 1) for h in WEEKS},
                      "contender": round(self.weigh(d, "contender"), 1),
                      "rebuilder": round(self.weigh(d, "rebuilder"), 1),
                      "for_their_mode": round(self.weigh(d, self.mode[t]), 1)}
        return res


if __name__ == "__main__":
    VALS, SETTINGS, ROSTERS, USERS, OUT = sys.argv[1:6]
    L = League(VALS, SETTINGS, ROSTERS, USERS)
    pd.set_option("display.width", 220)
    print(L.mode_table.round(1).to_string())
    rows = []
    for team, roster in L.teams.items():
        for pid in L.V.index:
            d = L.value_to(team, pid)
            rows.append({"team": team, "team_mode": L.mode[team], "sleeper_id": pid, "name": L.V.at[pid, "name"],
                         "pos": L.V.at[pid, "pos"], "owner": L.V.at[pid, "owner"],
                         "on_team": pid in roster, "rest2026": d[0], "y2027": d[1], "y2028": d[2], "y2029": d[3], "y2030": d[4], "y2031": d[5],
                         "team_contender": L.weigh(d, "contender"), "team_rebuilder": L.weigh(d, "rebuilder"),
                         "team_their_mode": L.weigh(d, L.mode[team]),
                         "actual_contender": L.V.at[pid, "contender"], "actual_rebuilder": L.V.at[pid, "rebuilder"]})
    R = pd.DataFrame(rows).round(1)
    R.to_csv(f"{OUT}/team_values_all.csv", index=False)
    L.mode_table.round(2).to_csv(f"{OUT}/team_modes.csv")
    me = R[R.team == "BraundJB"]
    c = ["name", "pos", "owner", "rest2026", "y2027", "team_contender", "actual_contender"]
    print("\nBRADY: value to keep\n", me[me.on_team].sort_values("team_contender", ascending=False)[c].to_string(index=False))
    print("\nBRADY: best adds\n", me[~me.on_team].sort_values("team_contender", ascending=False).head(12)[c].to_string(index=False))
    # proportionality check: one stud vs two mid starters, added to Brady's roster
    allv = L.V[~L.V.index.isin(L.teams["BraundJB"])].sort_values("contender", ascending=False)
    mids = list(allv.index[20:22])
    for stud in ["Jaxon Smith-Njigba", "Jahmyr Gibbs"]:
        t1 = L.trade("BraundJB", [], L.V.at[L.byname[stud], "owner"], [stud])["BraundJB"]["contender"]
        a = L.season_points(L.teams["BraundJB"] + mids)
        t2 = L.weigh({h: a[h] - L.base["BraundJB"][h] for h in WEEKS}, "contender")
        print(f"Add {stud}: {t1:.1f} vs add {L.V.at[mids[0], 'name']} + {L.V.at[mids[1], 'name']}: {t2:.1f}  "
              f"(actual: {L.V.at[L.byname[stud], 'contender']:.0f} vs {L.V.loc[mids, 'contender'].sum():.0f})")
