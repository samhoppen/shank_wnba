# `wnba_origination` — Full Setup Guide

End-to-end setup for running the WNBA line-origination Streamlit app on a fresh machine (Linux / macOS / Windows). This covers the Python environment, the external `WNBA_RAPM` data dependency, optional inputs (BigDataBall, EC scraper), and the order of scripts to run for a first-time build and for daily refreshes.

> TL;DR: the app *can* boot off the pre-computed CSVs already committed to `wnba_origination/data/`, but rebuilding caches, drawing rotation grids, or projecting a fresh slate of games requires the external `WNBA_RAPM/` directory (raw PBP JSON + stints + RAPM coefficients).

---

## 1. Prerequisites

| Tool | Min version | Notes |
|---|---|---|
| Python | 3.9+ | `zoneinfo` stdlib is required; 3.11 recommended |
| pip / venv | latest | use a virtualenv to isolate deps |
| git | any | repo is cloned; data lives outside the repo |
| Node + npm | 18+ | **only if rebuilding the React rotation chart**; pre-built assets are committed |
| Jupyter | latest | only if rebuilding RAPM from scratch via the notebook |

---

## 2. Repository Layout

The app expects this on-disk layout. `WNBA_RAPM/` is a **sibling** of `wnba_origination/`, not inside the repo:

```
<workspace>/
├── shank_wnba/                       ← this repo
│   └── wnba_origination/
│       ├── app.py, matchup.py, …
│       ├── data/                     ← mirrored CSVs (the app reads from here)
│       ├── paths.py                  ← path config (env-var overridable)
│       └── components/rotation_chart/frontend/build/   ← pre-built React asset
│
└── WNBA_RAPM/                        ← external, NOT in this repo
    ├── wnba_rapm.ipynb               ← RAPM pipeline notebook
    ├── refresh_2026.py               ← (optional) used by app's Refresh button
    ├── wnba_data/
    │   ├── raw_pbp/                  ← {game_id}_pbp.json, {game_id}_starters.json
    │   ├── stints/                   ← stints_{year}_RS.csv  (e.g. 2021..2026)
    │   ├── stints_rich/              ← stints_rich_{year}_RS.csv (8-factor inputs)
    │   ├── games_{year}_Regular_Season.csv
    │   ├── rapm_{year}_RS.csv        ← single-year RAPM outputs
    │   ├── rapm_2025_RS_3yr.csv      ← 3-year rolling RAPM
    │   ├── player_minutes.csv
    │   └── player_names.csv
    └── analysis/
        ├── pace_stats.csv
        ├── ft_decomp.csv
        ├── bonus_by_quarter.csv
        ├── foul_violation_rates.csv
        ├── player_stats.csv
        └── ft_shooting_foul_locs.csv
```

`paths.py` defaults `RAPM_DIR` to a Windows-only path. **On Linux/macOS you must override it** — either via the `WNBA_RAPM_DIR` env var or by editing `paths.py:9-12`.

---

## 3. One-time setup

### 3.1 Clone and create the Python env

```bash
git clone <repo-url> shank_wnba
cd shank_wnba

python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

cd wnba_origination
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins: `nba_api 1.11.4`, `numpy 2.1.3`, `pandas 2.2.3`, `playwright 1.57.0`, `plotly 6.5.2`, `scipy 1.17.1`, `scikit-learn 1.8.0`, `streamlit 1.54.0`.

### 3.2 Install Playwright browser (only if running the EC scraper)

`ingest.py:run_ec_scraper()` launches a headless browser via Playwright. Skip if you'll always run with `--skip-ec`.

```bash
playwright install chromium
```

### 3.3 Stand up the external `WNBA_RAPM/` directory

Two options:

**Option A — copy a populated `WNBA_RAPM/` from another machine.** Fastest. Put it next to `shank_wnba/`. Confirm it contains `wnba_data/raw_pbp/`, `wnba_data/stints/`, `wnba_data/games_*_Regular_Season.csv`, `rapm_*_RS.csv`, `player_minutes.csv`, `player_names.csv`, plus an `analysis/` subfolder.

**Option B — rebuild from scratch via the notebook.** Slow (~20 min per season; rate-limited at 2 s/request). You need the `wnba_rapm.ipynb` notebook itself (it lives outside this repo — get it from the upstream source). Then:

```bash
mkdir -p ../WNBA_RAPM/wnba_data
cp /path/to/wnba_rapm.ipynb ../WNBA_RAPM/
cd ../WNBA_RAPM
pip install jupyter tqdm
jupyter notebook wnba_rapm.ipynb
```

In Cell 0 set:
```python
SEASONS            = [2022, 2023, 2024, 2025]
RUN_SEASON         = None     # None = loop all SEASONS
RUN_REBUILD_STINTS = True
RUN_MULTI_YEAR     = True     # builds 3-year rolling RAPM
```
Run Cell 0. Outputs land in `WNBA_RAPM/wnba_data/`. The notebook has resume logic — interrupted runs pick back up without re-downloading.

### 3.4 Point the app at `WNBA_RAPM/`

Set the env var so `paths.py:9-12` resolves correctly:

```bash
export WNBA_RAPM_DIR="$HOME/path/to/WNBA_RAPM/wnba_data"
```

Persist it in `~/.bashrc` / `~/.zshrc` (or `.env` + a loader of your choice). Confirm:

```bash
python -c "from paths import RAPM_DIR; print(RAPM_DIR, RAPM_DIR.exists())"
```

> **Two other Windows paths in `paths.py:14-16`** (`EC_ALL_SEASONS`, `EC_SCRAPER`, `PYTHON`) only matter if you're running `ingest.py`'s EC scraper. Edit them inline if you need them; otherwise leave them and skip EC steps.

### 3.5 (Optional) Drop BigDataBall CSVs in place

Needed only for the **Performance** tab.

```bash
mkdir -p data/bigdataball
# Manually download from BigDataBall and drop in:
#   data/bigdataball/2024.csv
#   data/bigdataball/2025.csv
#   data/bigdataball/2026.csv
```

### 3.6 (Optional) Rebuild the React rotation chart

The build artifact in `components/rotation_chart/frontend/build/` is already committed. Only do this if you've modified TypeScript sources.

```bash
cd components/rotation_chart/frontend
npm install
npm run build
cd -
```

---

## 4. First-time data build

Once `WNBA_RAPM/` is in place and `WNBA_RAPM_DIR` is set, build all caches the app reads.

Run **from `wnba_origination/`**:

```bash
# 1. Mirror analysis CSVs from WNBA_RAPM/{analysis,wnba_data}/ into data/
python sync_data.py

# 2. Build/refresh the unified player table (RAPM + minutes + team + EC)
#    Writes data/player_store.csv
python player_store.py

# 3. Build the per-game four-factors cache from raw PBP JSON
#    Writes data/game_log.csv
python game_log.py               # current year (2026)
python game_log.py --year 2025   # add historical years one by one if desired

# 4. Build the pace cache (only if data/pace_cache.csv is missing/stale)
#    Uses 2021–2025 stints with year-weighted shrinkage to league mean
python pace.py
```

Sanity-check that the four cache files exist:

```bash
ls -lh data/player_store.csv data/game_log.csv data/pace_cache.csv data/pace_stats.csv
```

### 4.1 (Optional) Bootstrap 2026 rotations

If `data/rotation_overrides.csv` is empty or out of date, edit the hardcoded `RAW` roster string at the top of `_load_rosters.py` and run once:

```bash
python _load_rosters.py
```

This fuzzy-matches names → player IDs and writes default 28-min starters / 12-min backups.

---

## 5. Launch the app

```bash
cd wnba_origination
streamlit run app.py
```

Default URL: <http://localhost:8501>. Tabs:

- **Game** — pick teams, edit rotations, see projected spread/total/ML.
- **Season** — 44-game win totals from current RAPM.
- **Roster** — persistent player and rotation overrides.
- **Performance** — backtest vs BigDataBall (requires Section 3.5).
- **League Dashboard** — pace, four factors, foul/bonus rates.

The sidebar **Refresh data** button kicks off the full data pipeline (see `app.py:2214` → `_refresh_all_data()`). It expects `WNBA_RAPM/refresh_2026.py` to exist; without it, step 1 is skipped with a warning and the remaining steps (sync_data → game_log → player_store) still run.

---

## 6. Daily / in-season refresh

After each game day, do one of the following.

### 6.1 In-app (recommended)

Click **Refresh data** in the sidebar. It runs `WNBA_RAPM/refresh_2026.py` → `sync_data.sync()` → `game_log.build()` → `player_store.build()`, then clears Streamlit's cache.

### 6.2 From the command line

Two paths depending on where new data lives:

**A. The notebook already produced new RAPM CSVs.** Just rebuild the app-side caches:

```bash
python refresh.py                # rebuilds player_store + game_log for current year
python refresh.py --year 2025    # explicit year
python refresh.py --all          # rebuild game_log across all years
```

**B. You want the app to fetch new PBP itself** (instead of running the notebook). Use the daily ETL:

```bash
python ingest.py                 # full refresh: games → PBP → stints → minutes → EC → maybe refit RAPM → player_store
python ingest.py --skip-ec       # skip Playwright EC scraper
python ingest.py --skip-rapm     # skip RAPM refit even past MIN_NEW_POSS=500
```

`ingest.py` writes to `RAPM_DIR` directly (stints, player_minutes), refits RAPM when ≥500 new possessions have accumulated, then rebuilds `player_store.csv`.

### 6.3 (Optional) Cross-check vs `helpthehelper.vercel.app`

```bash
python fetch_hth.py              # writes data/hth_players_2026.csv
```

### 6.4 (Optional) Validate PBP-derived minutes vs ESPN box scores

```bash
python verify_minutes.py
```

Requires `data/espn_box_2026.tsv` and a populated `RAPM_DIR/raw_pbp/`.

---

## 7. Script reference (run order)

| Stage | Script | When to run | Reads | Writes |
|---|---|---|---|---|
| Setup | `sync_data.py` | After upstream notebook updates `WNBA_RAPM/analysis` or `wnba_data` | `WNBA_RAPM/{analysis,wnba_data}/*` | `data/*.csv` (12 files mirrored) |
| Setup | `pace.py` | Once, or when stints 2021–2025 change | `RAPM_DIR/stints/stints_{2021..2025}_RS.csv` | `data/pace_cache.csv` |
| Setup | `player_store.py` | After any RAPM or roster update | RAPM CSVs, `player_minutes`, `player_names`, `analysis_player_stats`, EC CSVs | `data/player_store.csv` |
| Setup | `game_log.py` | After new games | `RAPM_DIR/raw_pbp/*.json`, `games_*_Regular_Season.csv`, stints | `data/game_log.csv` |
| Setup | `_load_rosters.py` | Once at season start (edit hardcoded list first) | embedded `RAW` string | `data/rotation_overrides.csv` |
| Daily | `refresh.py` | After notebook produces new RAPM | calls `player_store.py` + `game_log.py` | same as above |
| Daily | `ingest.py` | Alternative to the notebook — fetches PBP itself | nba_api, existing CSVs | `RAPM_DIR/stints`, `player_minutes`, `data/ec_2026.csv`, `data/rapm_2026_RS.csv`, `data/player_store.csv` |
| Daily | `rapm.py` | Implicit via `ingest.py`; manual if rebuilding RAPM from custom stints | `stints_{year}_RS.csv` | `data/rapm_{year}_RS.csv` |
| App | `streamlit run app.py` | Always last | all of `data/*.csv` + `RAPM_DIR/raw_pbp/` for rotation grids | Streamlit UI |
| Optional | `fetch_hth.py` | Cross-validation | `helpthehelper.vercel.app` | `data/hth_players_2026.csv` |
| Optional | `verify_minutes.py` | Sanity check | `data/espn_box_2026.tsv`, raw PBP | console report |

---

## 8. Troubleshooting

**`FileNotFoundError` pointing at `C:/Users/shank...`** — you skipped 3.4. Export `WNBA_RAPM_DIR` and restart your shell / Streamlit process.

**`refresh_2026.py not found`** in the in-app Refresh log — the notebook side of the upstream repo doesn't ship that helper here. Either author it (it should regenerate the analysis CSVs and 2026 PBP) or run the notebook + `python refresh.py` manually.

**Rotation chart shows empty / "no PBP for game X"** — `RAPM_DIR/raw_pbp/{game_id}_pbp.json` is missing. Re-run the notebook (or `ingest.py`) to pull it.

**`player_store.csv` has 0 rows for 2026** — RAPM priority falls back to 2025 / 8-factor / 0.0 (`player_store.py:RAPM priority order`). Expected early-season; will populate once 2026 RAPM CSV crosses 300 possessions per player.

**`nba_api` rate-limited** — there's a built-in 2 s sleep in the notebook and `ingest.py`; don't lower it. Backoff and retry.

**Performance tab empty** — drop BigDataBall CSVs into `data/bigdataball/` (see 3.5).

**Win totals don't sum to 308** — the zero-sum rescale in `win_totals.py` enforces this; a violation usually means a team is missing from `player_store.csv`. Check `team_abbr` coverage for the 15 WNBA tricodes.

---

## 9. Minimum viable run (no `WNBA_RAPM/` available)

If you only want to *boot* the app to inspect the UI using the CSVs already committed to `data/`:

```bash
cd wnba_origination
pip install -r requirements.txt
export WNBA_RAPM_DIR=/tmp/empty_rapm     # any existing path; rotation grids will be blank
mkdir -p /tmp/empty_rapm/raw_pbp
streamlit run app.py
```

Expect:
- ✓ Game / Season / Roster / League tabs render from the cached CSVs.
- ✗ Rotation last-N-games heatmaps blank (no raw PBP).
- ✗ Performance tab empty (no BigDataBall).
- ✗ `refresh.py`, `game_log.py`, `pace.py` will all fail because they need real `RAPM_DIR` data.

For anything beyond inspection, complete Section 3.3.
