# WNBA Line Origination Tool

A Streamlit app for projecting WNBA game spreads and totals using ridge-regression RAPM, built from play-by-play stint data.

---

## What's in the App

### Game Tab
The main tool. Pick a home and away team, adjust rotations, get a projected line.

- **Rotation editor** — drag-and-drop stint chart for each team. Each player row shows their stint blocks across a 0–40 minute timeline. You can:
  - Drag stints left/right or resize from either edge
  - Add/remove stints using the `+` / `−` count column on the left
  - Mark a player **OUT** to zero their minutes (they drop to the bottom automatically)
  - Override minutes manually with the number input
  - Edit a player's RAPM inline via the `?` button
- **Position labels** — top 5 players by minutes are auto-labeled PG/SG/SF/PF/C
- **Net RAPM** shown next to each player name (green = positive, orange = negative)
- **Output** — projected spread, total, home/away points, win probability, and money lines
- **Last 2 games reference** — expandable heatmap per player showing minute-by-minute court presence

### Season Tab
Win total projections for all 15 teams based on current rosters and RAPM ratings. Adjustable games-played assumption.

### Roster Tab
Persistent overrides that survive app restarts:
- Reassign a player to a different team (trades, free agency)
- Override a player's RAPM manually
- Upload a rotation CSV

### Performance Tab
Model projections vs. actual results. Tracks spread and total accuracy over the season.

---

## Where RAPM Comes From

RAPM (Regularized Adjusted Plus/Minus) is computed from a separate pipeline in `wnba_rapm.ipynb`. Here's the short version:

1. **Play-by-play data** is pulled from the NBA Stats API (league_id `"10"` = WNBA) for every regular season game going back to 2017.
2. Each game is broken into **stints** — continuous possessions where both 5-player lineups stay the same.
3. A **ridge regression** is run on all stints (weighted by possessions), solving for each player's marginal offensive and defensive contribution. Ridge penalty = 2000; minimum 300 possessions to be included.
4. The final model uses a **3-year rolling window** (e.g., 2023–2025 combined), which improves stability for players with limited single-year samples.

The outputs are two numbers per player:
- `orapm` — offensive RAPM (points added per 100 possessions on offense)
- `drapm` — defensive RAPM (points added per 100 possessions on defense; positive = good defender)

**Priority order for what ends up in the app:**
1. 2026 single-year RAPM (once enough possessions accumulate mid-season)
2. 2025 single-year RAPM (for players with 300+ 2025 possessions)
3. 2025 3-year RAPM (fallback for players missing from the 1yr file — e.g., traded players or those with limited 2025 data)
4. `0.0` — league average (true rookies / expansion players with no history)

**Line projection math:**
```
home_pts = (adj_ortg_home + adj_drtg_away - league_avg) * pace / 100
away_pts = (adj_ortg_away + adj_drtg_home - league_avg) * pace / 100
spread   = home_pts - away_pts + HCA
total    = home_pts + away_pts
```
Where `league_avg ≈ 95`, `HCA ≈ 2.2 pts`, and pace is team-specific from the stints data.

---

## How to Update After New Games

New game data flows through two steps: the RAPM notebook, then the app caches.

### Step 1 — Fetch new games (run this after each game day)

Open `wnba_rapm.ipynb` in Jupyter and set these flags at the top of **Cell 0**:

```python
RUN_SEASON         = 2025   # or whatever the current year is
RUN_REBUILD_STINTS = True
```

Run Cell 0. The pipeline will:
- Skip games already processed (PBP JSON cached in `wnba_data/raw_pbp/`)
- Fetch new game PBP from the NBA Stats API (~2 sec/game)
- Rebuild stints and recompute RAPM
- Output updated `wnba_data/rapm_2025_RS.csv` and `wnba_data/stints/stints_2025_RS.csv`

> **Note:** Keep `RUN_REBUILD_STINTS = False` in other cells/runs unless you actually want to reprocess. The notebook has resume logic so it's safe to re-run — it won't re-fetch cached games.

### Step 2 — Refresh app caches

From the `wnba_origination/` folder:

```bash
python refresh.py
```

This rebuilds `data/player_store.csv` (picks up new RAPM values) and `data/game_log.csv` (picks up new box scores for the last-games heatmaps).

### Step 3 — Restart the app

```bash
streamlit run app.py
```

Then hard refresh the browser (`Ctrl+Shift+R`).

---

## Roster Updates (Trades / Free Agency)

The roster file is **hardcoded in `_load_rosters.py`**. Run this script once after any trade to update team assignments:

```bash
python _load_rosters.py
```

It matches abbreviated names (`S. Ionescu`) to player IDs via last-name lookup and saves to `data/rotation_overrides.csv`. Edit the `RAW` string at the top to reflect new rosters.

For one-off changes (single player), use the **Roster tab** in the app — no code required.

---

## Running Locally

### App only (pre-built RAPM CSVs included)

```bash
cd wnba_origination
pip install streamlit pandas numpy scipy scikit-learn
streamlit run app.py
```

The app reads from:
- `data/player_store.csv` — player RAPM + team assignments
- `data/game_log.csv` — per-game box scores and four factors
- `data/pace_cache.csv` — team pace estimates
- `wnba_data/` — raw RAPM outputs and stints

The `wnba_data/` folder with pre-built CSVs is included — **you don't need to run the notebook to use the app.** The RAPM data is already computed through the 2025 season.

---

## Running the RAPM Model Yourself

If you want to rebuild RAPM from scratch or update it with new games, you'll need the notebook.

### Dependencies

```bash
pip install numpy pandas scipy scikit-learn tqdm nba_api jupyter
```

### Setup

The notebook (`wnba_rapm.ipynb`) lives one level above `wnba_origination/`, next to the `wnba_data/` folder. It uses `DATA_DIR = Path("wnba_data")` — a relative path — so it works on any machine as long as you run it from that directory.

```
WNBA_RAPM/
├── wnba_rapm.ipynb      ← notebook lives here
├── wnba_data/           ← outputs land here
│   ├── raw_pbp/         ← cached PBP JSON per game
│   ├── stints/          ← stint CSVs per season
│   ├── stints_rich/     ← 8-factor input CSVs
│   ├── rapm_2025_RS.csv ← single-year RAPM
│   └── rapm_2025_RS_3yr.csv ← 3yr rolling RAPM
└── wnba_origination/    ← app lives here
```

### First-time full build

Open `wnba_rapm.ipynb` in Jupyter. In Cell 0, set:

```python
SEASONS            = [2022, 2023, 2024, 2025]   # years to process
RUN_SEASON         = None        # None = loop all SEASONS
RUN_REBUILD_STINTS = True
RUN_MULTI_YEAR     = True        # builds 3yr rolling RAPM
```

Run Cell 0. It will:
1. Fetch game IDs for each season from the NBA Stats API
2. Download PBP JSON for every game into `wnba_data/raw_pbp/` (~2 sec/game, ~572 games for 2025)
3. Parse stints (5v5 lineup segments) from each game
4. Run ridge regression per season → `rapm_{year}_RS.csv`
5. Run 3-year rolling regression → `rapm_{year}_RS_3yr.csv`

**Resume logic is built in** — if the run is interrupted, re-running Cell 0 skips any game whose PBP is already cached. You won't re-download anything.

### API rate limiting

The NBA Stats API is unauthenticated but will block you if you hammer it. The notebook has a `SLEEP_SEC = 2.0` delay between requests. Don't lower it. On a first full run expect ~20 minutes for a single season.

### After rebuilding

Run `python wnba_origination/refresh.py` to rebuild the app's caches from the new RAPM outputs.

---

## Key Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — all tabs |
| `matchup.py` | Spread/total/win prob math |
| `player_store.py` | Builds player RAPM table from CSV inputs |
| `game_log.py` | Builds per-game box score cache from raw PBP |
| `roster.py` | Roster override layer (trades, manual RAPM edits) |
| `pace.py` | Team pace estimates |
| `_load_rosters.py` | One-time script: load 2026 rosters from hardcoded list |
| `refresh.py` | Run after notebook update to rebuild app caches |
| `components/rotation_chart/` | React drag-and-drop component (built with Vite) |
| `wnba_rapm.ipynb` | RAPM pipeline — PBP fetch, stints, ridge regression |
