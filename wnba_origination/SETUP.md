# `wnba_origination` — Operations Runbook

End-to-end guide for running the WNBA line-origination Streamlit app: setup on a fresh machine, daily in-season refresh, and how to actually use the app. Linux / macOS / Windows.

> TL;DR — clone with `--recurse-submodules`, `pip install -r wnba_origination/requirements.txt`, bootstrap the raw-PBP cache with `python scripts/fetch_pbp.py`, regenerate the analysis CSVs, then `streamlit run app.py`.

---

## 1. Architecture

The repo is split into two pieces:

```
shank_wnba/
├── wnba_origination/             ← the streamlit app + orchestration
│   ├── app.py                    ← Streamlit entry point
│   ├── paths.py                  ← single source of truth for all file paths
│   ├── refresh.py                ← CLI orchestrator
│   ├── sync_data.py              ← mirror submodule CSVs → data/
│   ├── scripts/
│   │   ├── fetch_pbp.py          ← download raw PBP JSON via nba_api
│   │   └── regen_analysis.py     ← rebuild pace_stats / bonus / ft / fouls
│   └── data/                     ← local cache the app actually reads
│
├── wnba_rapm/                    ← git submodule: shankapotomus/wnba-rapm
│   ├── update_stints.py          ← fetch new games, append to stints CSVs
│   ├── run_rapm.py               ← ridge-regression RAPM fit (script)
│   ├── rapm_reproducible.ipynb   ← same fit as a notebook (optional)
│   ├── pbp_shares.py             ← lineup/share analytics helpers
│   └── wnba_data/                ← stints, games, RAPM coefficients
│
└── wnba_rapm_cache/              ← .gitignored, populated on demand
    └── raw_pbp/                  ← {game_id}_pbp.json + {game_id}_starters.json
```

**Why three locations?**

- `wnba_rapm/` is a clean public repo (shankapotomus/wnba-rapm). It owns the RAPM math and the stints data. Pull updates with `git submodule update --remote`.
- `wnba_rapm_cache/` holds the ~400 MB of raw PBP JSON the submodule intentionally excludes. The app reads it for rotation heatmaps, box scores, and to regenerate analysis CSVs.
- `wnba_origination/data/` is the local cache the streamlit app reads from. Everything in it is either copied from the submodule (via `sync_data.py`) or computed locally (via `regen_analysis.py`, `player_store.py`, `game_log.py`).

`paths.py` is the single source of truth. `RAPM_DIR` defaults to `wnba_rapm/wnba_data/` and `RAW_PBP_DIR` defaults to `wnba_rapm_cache/raw_pbp/`. Override either with the `WNBA_RAPM_DIR` and `WNBA_RAW_PBP_DIR` env vars.

---

## 2. Prerequisites

| Tool | Min version | Notes |
|---|---|---|
| Python | 3.10+ | `from __future__ import annotations` is used; 3.11 recommended |
| pip / venv | latest | a virtualenv is strongly recommended |
| git | 2.13+ | submodule support |
| Jupyter | latest | **optional** — only needed if you prefer the notebook path; `run_rapm.py` covers the same fit without it |
| Node + npm | 18+ | **only if rebuilding the React rotation chart** — pre-built assets are committed |

Network access to `stats.nba.com` is required for any data fetch (stints updates, raw PBP downloads).

---

## 3. One-Time Setup

### 3.1 Clone with submodules

```bash
git clone --recurse-submodules https://github.com/samhoppen/shank_wnba.git
cd shank_wnba
```

If you forgot `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

Verify the submodule populated:

```bash
ls wnba_rapm/wnba_data/stints/stints_2026_RS.csv   # should exist
```

### 3.2 Create a Python env and install deps

```bash
python3 -m venv .venv
source .venv/bin/activate                     # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r wnba_origination/requirements.txt
pip install jupyter nbconvert                 # only if you want the notebook RAPM refit path
```

### 3.3 Verify paths resolve

```bash
cd wnba_origination
python -c "from paths import RAPM_DIR, RAW_PBP_DIR; print('RAPM_DIR:', RAPM_DIR); print('RAW_PBP_DIR:', RAW_PBP_DIR)"
```

Expected output:

```
RAPM_DIR:    /path/to/shank_wnba/wnba_rapm/wnba_data
RAW_PBP_DIR: /path/to/shank_wnba/wnba_rapm_cache/raw_pbp
```

If you keep `wnba_data` elsewhere, point `WNBA_RAPM_DIR` at it:

```bash
export WNBA_RAPM_DIR=/some/other/path/wnba_data
```

### 3.4 Bootstrap the raw PBP cache

The submodule excludes raw PBP JSON. Download it (~400 MB, slow — be patient, rate-limited at 0.6 s/request):

```bash
# All seasons (~1 hour wall time)
python scripts/fetch_pbp.py --years 2017-2026

# Or just the current year (~5 min)
python scripts/fetch_pbp.py --year 2026
```

Files land in `wnba_rapm_cache/raw_pbp/` as `{game_id}_pbp.json` and `{game_id}_starters.json`. Re-running skips files that already exist; pass `--force` to redownload.

### 3.5 Build the local caches the app reads

```bash
python sync_data.py             # mirror games + rapm + stints_rich from submodule into data/
python scripts/regen_analysis.py --all   # build pace_stats, bonus, ft_decomp, foul rates from PBP
python pace.py                  # build pace_cache.csv (uses 2021-2025 stints)
python player_store.py          # build unified player table
python game_log.py              # build per-game four/five factors
```

After this, `wnba_origination/data/` should contain:

```
player_store.csv      pace_cache.csv      pace_stats.csv
game_log.csv          bonus_by_quarter.csv
rapm_2025_RS.csv      ft_decomp.csv       foul_violation_rates.csv
games_2025_RS.csv     games_2026_RS.csv
stints_rich_2025.csv  stints_rich_2026.csv
```

### 3.6 (Optional) Bootstrap 2026 rotations

If `data/rotation_overrides.csv` is empty or out of date, edit the hardcoded `RAW` roster string at the top of `_load_rosters.py` and run once:

```bash
python _load_rosters.py
```

This fuzzy-matches names → player IDs and writes default 28-min starters / 12-min backups.

### 3.7 (Optional) BigDataBall CSVs for the Performance tab

```bash
mkdir -p data/bigdataball
# Drop BigDataBall season CSVs here:
#   data/bigdataball/2025.csv
#   data/bigdataball/2026.csv
```

### 3.8 (Optional) Rebuild the React rotation chart

The build artifact in `components/rotation_chart/frontend/build/` is already committed. Only do this if you've modified TypeScript sources.

```bash
cd components/rotation_chart/frontend
npm install
npm run build
cd -
```

---

## 4. Launch the App

```bash
cd wnba_origination
streamlit run app.py
```

Default URL: <http://localhost:8501>.

---

## 5. Using the App

### Game Tab — line origination
The main tool. Pick home and away teams, adjust rotations, get a projected line.

- **Rotation editor** — drag-and-drop stint chart per team. Each player row shows stint blocks across a 0–40 min timeline. You can:
  - Drag stints left/right or resize from either edge
  - Add/remove stints with the `+` / `−` column on the left
  - Mark a player **OUT** to zero their minutes
  - Override minutes via the number input
  - Edit a player's RAPM inline with the `?` button
- **Position labels** — top 5 by minutes auto-labeled PG/SG/SF/PF/C
- **Net RAPM** color-coded per player (green = positive, orange = negative)
- **Output** — projected spread, total, home/away points, win probability, money lines
- **Last 2 games reference** — expandable heatmap per player showing minute-by-minute court presence (requires raw PBP cache populated)

### Roster Tab — persistent overrides
- Reassign a player to a different team (trades, free agency)
- Override a player's RAPM manually
- Upload a rotation CSV
- All edits survive restarts via `data/rotation_overrides.csv`

### Performance Tab — backtest vs market
Spread and total accuracy vs BigDataBall lines (requires §3.7).

### Game Log Tab — five factors per game
eFG%, TOV%, OREB%, FT Rate, Pace for every game. Drives the rolling 30-game baselines used in the League Dashboard.

### League Dashboard Tab
- **Season Context** — pace / ORTG / DRTG / PTS for 2025 vs 2026 vs last 30 games
- **Team Profiles** — per-team pace, ORTG, DRTG, Net RTG, four factors
- **Bonus & FTs** — bonus reach % by quarter, foul/violation rates

### EC Historical Tab
External calibration metrics (no live data dependency).

### Sidebar
- **🔄 Refresh all data** — runs the full 7-step pipeline (see §6.1)
- **Advanced rebuilds** expander:
  - *Rebuild player store* — re-derive `data/player_store.csv` only
  - *Rebuild pace cache* — re-derive `data/pace_cache.csv` only
  - *Run daily ingest (fast)* — stints + PBP + analysis, no RAPM refit
  - *Backfill raw PBP cache (slow)* — one-shot 2017-2026 PBP download

---

## 6. Daily / In-Season Refresh

### 6.1 In-app (recommended)

Click **🔄 Refresh all data** in the sidebar. Runs:

1. `wnba_rapm/update_stints.py --season 2026` — fetches new games, appends to stints
2. `jupyter nbconvert --execute rapm_reproducible.ipynb` — refits RAPM coefficients
3. `scripts/fetch_pbp.py --year 2026` — downloads raw PBP for new games
4. `scripts/regen_analysis.py --year 2026 --append-2025` — rebuilds analysis CSVs
5. `game_log.build()` — per-game four/five factors
6. `fetch_hth.py` + `sync_data.sync()` — HTH cross-check + mirror submodule CSVs
7. `player_store.build()` — unified player table

Then clears Streamlit's cache and reruns. Total: ~5-15 min depending on how many new games.

### 6.2 From the command line

Full pipeline (equivalent to the sidebar button):

```bash
cd wnba_origination
python refresh.py                   # current year (2026)
python refresh.py --year 2025       # explicit year
python refresh.py --skip-notebook   # skip RAPM refit (fast path)
python refresh.py --skip-fetch      # skip stints + PBP downloads
```

### 6.3 Fast incremental path

If you just need new PBP and analysis without refitting RAPM:

```bash
cd wnba_rapm && python update_stints.py --season 2026 && cd -
python scripts/fetch_pbp.py --year 2026
python scripts/regen_analysis.py --year 2026 --append-2025
python game_log.py --year 2026
python player_store.py
```

### 6.4 Refit RAPM only

Script path (no Jupyter required):

```bash
cd wnba_rapm && python run_rapm.py && cd -
python sync_data.py        # copy fresh rapm_*.csv into wnba_origination/data/
python player_store.py
```

Notebook path (equivalent):

```bash
cd wnba_rapm
jupyter nbconvert --to notebook --execute --inplace rapm_reproducible.ipynb
cd -
python sync_data.py
python player_store.py
```

### 6.5 Pull submodule updates from upstream

```bash
cd wnba_rapm && git pull origin main && cd -
git add wnba_rapm && git commit -m "Bump wnba_rapm submodule"
```

### 6.6 Cross-check vs helpthehelper.vercel.app

```bash
python fetch_hth.py        # writes data/hth_players_2026.csv
```

### 6.7 Validate PBP-derived minutes vs ESPN box scores

```bash
python verify_minutes.py   # requires data/espn_box_2026.tsv + populated raw PBP
```

---

## 7. Script Reference

| Script | When to run | Reads | Writes |
|---|---|---|---|
| `wnba_rapm/update_stints.py` | After new game(s) played | nba_api, existing stints CSV | `wnba_rapm/wnba_data/stints/stints_{year}_RS.csv` (+`stints_rich`) |
| `wnba_rapm/run_rapm.py` | After stints update (script path) | stints CSVs | `wnba_rapm/wnba_data/rapm_and_4f_output.csv`, prints top-25 net RAPM |
| `wnba_rapm/rapm_reproducible.ipynb` | After stints update (notebook path) | stints CSVs | `wnba_rapm/wnba_data/rapm_{year}_RS.csv` (and 3yr/8factor variants) |
| `scripts/fetch_pbp.py` | After new game(s); first-time backfill | `wnba_rapm/wnba_data/games_*.csv`, nba_api | `wnba_rapm_cache/raw_pbp/{game_id}_pbp.json` + `_starters.json` |
| `scripts/regen_analysis.py` | After raw PBP cache changes | raw PBP, `stints_rich_*` | `data/{pace_stats,bonus_by_quarter,ft_decomp,foul_violation_rates}.csv` |
| `sync_data.py` | After submodule RAPM/games update | `wnba_rapm/wnba_data/` | `data/{games,stints_rich,rapm}_*.csv` |
| `pace.py` | Once, or when stints 2021-2025 change | `stints_{2021..2025}_RS.csv` | `data/pace_cache.csv` |
| `player_store.py` | After RAPM or roster update | RAPM CSVs, `player_minutes`, `player_names`, EC CSVs | `data/player_store.csv` |
| `game_log.py` | After raw PBP cache changes | `raw_pbp/*.json`, `games_*.csv`, stints | `data/game_log.csv` |
| `_load_rosters.py` | Once at season start (edit hardcoded list first) | embedded `RAW` string | `data/rotation_overrides.csv` |
| `refresh.py` | Daily orchestrator (CLI equivalent of sidebar button) | drives all of the above | same as above |
| `fetch_hth.py` | Cross-validation | `helpthehelper.vercel.app` | `data/hth_players_2026.csv` |
| `verify_minutes.py` | Sanity check | `data/espn_box_2026.tsv`, raw PBP | console report |
| `streamlit run app.py` | Always last | all of `data/*.csv` + `RAW_PBP_DIR/` | Streamlit UI |

---

## 8. Troubleshooting

**`FileNotFoundError` for files in `wnba_rapm/wnba_data/`** — the submodule isn't initialized. Run `git submodule update --init --recursive`.

**Rotation chart shows empty / "no PBP for game X"** — the raw PBP for that game isn't in `wnba_rapm_cache/raw_pbp/`. Run `python scripts/fetch_pbp.py --year 2026` (or `--game-ids <gid>`).

**League Dashboard `Bonus & FTs` panel empty** — analysis CSVs haven't been regenerated. Run `python scripts/regen_analysis.py --year 2026 --append-2025`. Requires raw PBP to be cached first.

**`update_stints.py` rate-limited** — there's a built-in 0.6 s sleep between requests. If you hit a rate limit anyway, wait a few minutes and rerun; it skips games already processed.

**`player_store.csv` has 0 rows for 2026 RAPM** — fallback chain hits 2025 1yr → 2025 3yr → 0.0 (`player_store.py`). Expected early-season; populates once 2026 RAPM CSV crosses 300 possessions per player.

**Refresh button hangs on Step 2 (notebook refit)** — `jupyter nbconvert --execute` runs for 5-15 min. Watch the streaming log; if it actually stalls, the timeout is 30 min. Use **Run daily ingest (fast)** in Advanced rebuilds to skip the refit, or refit from the CLI with `python wnba_rapm/run_rapm.py` (no Jupyter required).

**Performance tab empty** — drop BigDataBall CSVs into `data/bigdataball/` (see §3.7).

**Win totals don't sum to 308** — the zero-sum rescale in `win_totals.py` enforces this; a violation usually means a team is missing from `player_store.csv`. Check `team_abbr` coverage for the 15 WNBA tricodes.

**Submodule shows untracked files in `git status`** — you wrote into it (e.g. let the notebook regenerate CSVs in-place). Either commit those upstream in the wnba-rapm repo or `git -C wnba_rapm restore .` to discard.

---

## 9. Minimum Viable Run (no network, no PBP cache)

If you just want to *boot* the app to inspect the UI using the CSVs already committed to `data/`:

```bash
git clone --recurse-submodules https://github.com/samhoppen/shank_wnba.git
cd shank_wnba
pip install -r wnba_origination/requirements.txt
cd wnba_origination
streamlit run app.py
```

Expect:
- ✓ Game / Roster / Game Log / League Dashboard tabs render from the cached CSVs.
- ✗ Rotation last-N-games heatmaps blank (no raw PBP cached).
- ✗ Performance tab empty (no BigDataBall).
- ✗ Refresh button steps 1-3 fail without `stats.nba.com` access.

For anything beyond inspection, complete §3.4-3.5.
