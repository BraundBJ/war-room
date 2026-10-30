# CTESPN Dynasty War Room

Player values, team values, trade calculator, pick values and the rookie draft slot study for the
CTESPN Dynasty league (Sleeper `1313707182407024640`). Rebuilt automatically every Tuesday.

Anyone can load their own Sleeper league from the **League** button (username or league ID), and
`?league=<id>` in the URL opens the page straight to that league. League data loads live from Sleeper
in the browser; player projections come from the weekly build.

## How it updates

`.github/workflows/weekly.yml` runs every Tuesday at 10:17 am Eastern:

1. Downloads public NFL data (nflverse) and the league from the Sleeper API.
2. Finds the latest completed week and retrains the projection models from that point in the season.
3. Recomputes player values, team values, pick values and trade inputs.
4. Publishes `public/index.html` to GitHub Pages.

Run it any time from the **Actions** tab: pick "Weekly War Room update", then **Run workflow**.
A run takes about 30 to 45 minutes.

## Layout

| Path | What |
|---|---|
| `run.sh` | The whole build, in order |
| `pipeline/` | Model scripts (`wrconf.py` holds the week and paths) |
| `site/page.src.html` | Page template; data goes in at `__DATA__` (`assemble.py`) |
| `data/` | Fixed inputs: rookie slot curves, slot study, hit rates |

Run locally: `pip install -r requirements.txt && ./run.sh` (writes `public/index.html`).
