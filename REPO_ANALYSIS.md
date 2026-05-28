# Repository Analysis: `shank_wnba`

A comprehensive reference for the WNBA / college basketball analytics codebase. The repo contains three loosely coupled sub-projects: a **WNBA line origination Streamlit app** (`wnba_origination/`), a **college basketball play-by-play pipeline + shots dashboard** (`cbbd_data/`, `shots_dashboard/`), and a **tournament simulation toolkit** (root level).

---

## Table of Contents

1. [Script Descriptions](#1-script-descriptions)
   - [1.1 Root Level](#11-root-level)
   - [1.2 `cbbd_data/`](#12-cbbd_data)
   - [1.3 `shots_dashboard/`](#13-shots_dashboard)
   - [1.4 `wnba_origination/`](#14-wnba_origination)
2. [Package Dependencies](#2-package-dependencies)
3. [Potential Improvements](#3-potential-improvements)
4. [Primary Data Sources](#4-primary-data-sources)

---

## 1. Script Descriptions

### 1.1 Root Level

The root level holds the college-basketball ETL pipeline, two tournament-simulation tools, and exploratory notebooks.

| Script | Description | Inputs | Outputs |
|---|---|---|---|
| `daily_fetch.py` | Automated daily ETL for college basketball PBP. Fetches one day of games, parses substitutions, tracks lineups (5-man units), computes possession state machine, classifies dead-ball rebounds, and computes four factors. Designed to run from cron. | `.env` (`CBBD_API_KEY`); CBBD API (games, plays, game_players); optional `--date YYYY-MM-DD` (default: yesterday ET) | 9 daily CSVs in `cbbd_data/{games,plays,possessions,possessions_enriched,shots,lineup_stints,players,pbp_flat,four_factors}/YYYYMMDD_SEASON.csv`. Exit code `2` on any game failure |
| `run_daily_fetch.sh` | Bash wrapper for `daily_fetch.py`. Loads `.env`, creates `logs/`, redirects all output to `logs/daily_fetch_YYYYMMDD.log`. Intended for crontab (`0 6 * * *` with `TZ=America/New_York`) | `.env`, CLI args passed through | `logs/daily_fetch_YYYYMMDD.log`; propagates exit code |
| `audit_season.py` | Season-wide completeness auditor. Chunks the season into monthly blocks (avoids CBBD's ~3000-record cap), compares expected games vs the union of `games/` and `plays/` CSVs, and emits backfill commands | `.env`; local `cbbd_data/games/*.csv` and `cbbd_data/plays/*.csv`; optional `--season` and `--through` flags | `audit_missing_games.csv`, `audit_backfill_dates.csv` (with suggested `daily_fetch.py` invocations); console summary |
| `run_march_madness.py` | Launcher that boots `march_madness_sim_app.py` as a Streamlit server on port 8507 and opens a browser via a daemon thread. Compatible with PyInstaller bundling (`resource_path()` shim) | None | Browser window pointing at `localhost:8507` |
| `march_madness_sim_app.py` | 68-team NCAA bracket Monte Carlo simulator. Win probability uses possession-scaled spreads: `spread = (net_A - net_B) * poss / 100`, `P(A) = norm.cdf(spread / 11.0)`. Auto-fills ratings from cache, supports First Four and regional bracket editing, runs 1K–25K sims | `team_ratings_cache.csv` (adj_ortg, adj_drtg, tempo, adj_net); user-entered bracket via Streamlit sidebar | Streamlit UI with per-round advancement probabilities (R32%, S16%, E8%, F4%, Final%, Champ%); CSV/TSV export |
| `race_to_10_analysis.py` | Tests "race to 10 points" as a game-outcome predictor. Fits a logistic model `P(race_win) = sigmoid(k*spread + b)`, computes per-team season stats, then projects 2026 R64 matchups with Bayesian shrinkage (`M=20`) and seed bonus EV picks | `.env`; CBBD API (lines); `cbbd_data/pbp_flat/*.csv`; `cbbd_data/games/*.csv`; optional `team_ratings_cache.csv`; hardcoded `TOURNAMENT_MATCHUPS` (32 R64 pairs) + `TEAM_ALIASES` | `race_to_10_team_stats.csv`, `race_to_10_tournament.csv`, `race_to_10_actual_vs_expected.csv`, `race_to_10_bracket.html`; console output of logistic coefficients, calibration, top EV picks |
| `cbbd_playbyplay_2025_26.ipynb` | Batch (date-range) version of `daily_fetch.py` as a notebook. User enters API key via `getpass`, picks date range, fetches and processes all games in one pass | `.env` / `getpass`; CBBD API | 7 combined CSVs (one per data type for the entire range) |
| `cbbd_batch_pipeline.ipynb` | Interactive variant of the batch pipeline with per-cell control and verbose logging. Same logic as `daily_fetch.py` but step-by-step | `.env`; CBBD API; user-supplied date range | 7 CSVs (with logging) |
| `race_to_10.ipynb` | Interactive notebook driving the same race-to-10 logic as the script. Useful for tuning the logistic model and inspecting calibration | Same as `race_to_10_analysis.py` | CSVs + HTML viz |
| `poss_starter_analysis.ipynb` | Analyzes points-per-possession conditioned on how the possession started (`prev_poss_ender`). 15+ buckets: steal, dead-ball TO, made FG, DREB by shot zone, OOB by trigger type, etc. | `cbbd_data/{possessions_enriched, possessions, plays, shots}/*.csv` via `load_data.py` | Stacked-bar visualizations + PPP summary table |
| `team_possessions.ipynb` | Compares **tracked possessions** (rows in `possessions_enriched`) vs the KenPom estimate (`FGA - ORB + TOV + 0.475*FTA`) per game for a chosen team | Same inputs via `load_data.py` | Time-series plots + summary stats (mean/median/std/min/max difference) |

---

### 1.2 `cbbd_data/`

Helper scripts that operate on the daily CSVs produced by `daily_fetch.py`. The directory also serves as the storage location for those CSVs (`games/`, `plays/`, etc.).

| Script | Description | Inputs | Outputs |
|---|---|---|---|
| `load_data.py` | Single source of truth for loading the 8 CSV types into deduplicated DataFrames. Globs each subdirectory, concatenates, drops dupes on type-specific keys (e.g., `(gameId, possession_id, possession_team)` for `possessions_enriched`), and attaches `game_date` from `games_df` | `cbbd_data/{games,plays,possessions,possessions_enriched,shots,lineup_stints,players,pbp_flat,four_factors}/*.csv` | 8 module-level DataFrames: `games_df`, `plays_df`, `poss_df`, `poss_enriched_df`, `shots_df`, `lineup_stints_df`, `players_df`, `pbp_flat_df`, `ff_df` |
| `consolidate_csvs.py` | One-time migration script: converts the legacy flat-file layout (`games_*.csv` at root) into the daily-organized structure (`cbbd_data/games/YYYYMMDD_SEASON.csv`) that `daily_fetch.py` expects | Old flat CSVs at the root | Organized subdirectories; deletes flat files after migration |
| `fix_possessions.py` | Offline batch regenerator. Re-runs `track_possessions_v2()` + `classify_possessions()` on existing `plays/*.csv` to fix v7 bugs: M-of-N regex for last-FT, id tiebreaker for stable sorting, context-aware DBR categories, technical-FT possession retention, retroactive steal assignment, and DBR-between-FT detection in the old format | `cbbd_data/plays/*.csv` | Overwrites `cbbd_data/possessions/*.csv` and `cbbd_data/possessions_enriched/*.csv` |
| `diag_dbr_trigger.py` | Diagnostic: identifies what play triggered each dead-ball rebound in the preceding possession (made FG, missed jump shot, etc.) | `cbbd_data/{possessions_enriched, possessions, plays, shots}/*.csv` | Console output / summary tables |
| `diag_dbr_dead_ball_rebound.py` | Deep-dive on 1-point and FT-only DBR possessions. Samples 30 suspicious cases and shows the previous possession's plays. Checks whether prev possession ended with a missed FT | Same as above | Console samples + summary |
| `diag_dbr_shot_types.py` | Analyzes the shot types that triggered DBRs (rim vs mid-range vs 3pt, corner vs above-break 3, blocked vs not) | Same | Console output |
| `diag_dbr_lastplay.py` | For DBR-tagged possessions, prints the actual last play of the preceding possession. Also samples non-DBR-ending cases for classification debugging | Same | Console output |
| `diag_dbr_samples.py` | PBP samples of DBR possessions bucketed by their first meaningful play (PersonalFoul, MadeFreeThrow, JumpShot, Turnover, LayUpShot). Shows prev + curr possession plays side-by-side | Same | Console output |

---

### 1.3 `shots_dashboard/`

A Streamlit shooting-efficiency dashboard built on top of `cbbd_data/` outputs.

| Script | Description | Inputs | Outputs |
|---|---|---|---|
| `app.py` | Streamlit UI. Per-team offensive/defensive shooting profiles: four factors with percentile ranks vs D1, zone efficiency (rim, paint, non-paint 2, corner 3, non-corner 3), possession-bucket efficiency, shot-clock × zone heatmaps, 3-point context (corner share, shooter grades, clock distribution), and last-5-game breakdowns | `cbbd_data/{shots, possessions_enriched, four_factors, games}/*.csv` (via `data_loader.load_all_data`) | Interactive Streamlit dashboard (tables + Plotly charts) |
| `data_loader.py` | Feature-engineering pipeline. Loads the 4 raw tables, classifies shot zones, possession buckets (transition / first_hc / putback / second_hc / third_plus_hc), shot-clock buckets, links shots to possessions via a vectorized range-join, computes cumulative FGA within a trip, and assigns shooter 3pt grades (Red ≥37% on 3+ 3PA/G, Yellow 32–37%, Green <32% or low volume) | `cbbd_data/{shots, possessions_enriched, four_factors, games}/*.csv` | Returns `shots_enriched`, `possessions`, `four_factors`, `games` DataFrames |
| `metrics.py` | ~450 lines of aggregation helpers: `compute_zone_metrics`, `compute_bucket_metrics`, `compute_clock_zone_metrics`, `compute_assisted_metrics`, `compute_threept_context`, `compute_four_factors`, `compute_team_record`, `compute_league_stats` (D1 baseline of teams with ≥10 games), `percentile_rank`, `compute_poss_type_summary`, `compute_poss_ppp_league`, `compute_last5_game_breakdown`, `poss_percentile_rank`, `_ordinal` | DataFrames from `data_loader.py` | Aggregated metric DataFrames + percentile ranks |

---

### 1.4 `wnba_origination/`

The flagship sub-project: a Streamlit application that generates WNBA spread / total / moneyline lines from a Regularized Adjusted Plus-Minus (RAPM) model. The app surfaces game projections, season win totals, roster management, and model-vs-market performance tracking.

#### Data Flow

```
WNBA_RAPM notebook (external)              nba_api / playwright scraper
    │ raw_pbp/*.json                              │
    │ stints_*.csv, games_*.csv                   │
    │ rapm_*.csv, player_minutes.csv              │
    ▼                                             ▼
┌───────────────────┐         ┌──────────────────────────┐
│  sync_data.py     │◄────────│  ingest.py (daily ETL)   │
│  (mirror CSVs)    │         │   • fetch_games/pbp/box  │
└─────────┬─────────┘         │   • update_stints        │
          │                   │   • run_ec_scraper       │
          ▼                   │   • refit RAPM if needed │
data/*.csv (analysis CSVs)    └────────────┬─────────────┘
          │                                │
          ▼                                ▼
┌──────────────────────┐         ┌─────────────────────┐
│  player_store.py     │         │  game_log.py        │
│  (RAPM + minutes +   │         │  (four factors per  │
│   team + EC merge)   │         │   team per game)    │
└─────────┬────────────┘         └─────────┬───────────┘
          │ player_store.csv               │ game_log.csv
          │                                │
          ▼                                │
┌──────────────────────┐                   │
│  roster.py           │                   │
│  (apply overrides:   │                   │
│   trades, RAPM, mins)│                   │
└─────────┬────────────┘                   │
          ▼                                │
┌──────────────────────┐                   │
│  matchup.py          │                   │
│  (spread/total/ML)   │                   │
│   + pace.py          │                   │
│   + win_totals.py    │                   │
│   + lineup.py        │                   │
└─────────┬────────────┘                   │
          ▼                                ▼
┌──────────────────────────────────────────────────────┐
│  app.py — Streamlit UI                               │
│   • Game tab        (matchup engine)                 │
│   • Season tab      (win totals)                     │
│   • Roster tab      (roster.py)                      │
│   • Performance tab (performance.py vs BigDataBall)  │
│   • League tab      (league_dashboard / stats)       │
└──────────────────────────────────────────────────────┘
```

#### Core Application Files

**`app.py`** — Streamlit UI orchestrator (~main entry point).
- Tabs: **Game**, **Season**, **Roster**, **Performance**, **League Dashboard**.
- Builds an interactive drag-and-drop rotation chart per team, with minute sliders and per-player RAPM overrides.
- `_game_presence_from_pbp()` walks raw PBP JSON to reconstruct minute-by-minute on-court presence (handles substitutions and PBP data gaps).
- `_recent_game_presence()` extracts presence grids for the last N games; `_team_rotation_grid_html()` renders them as HTML heatmaps.
- `_html_table()` injects sortable JavaScript tables for game logs.
- Cached loaders: `get_store`, `get_pace_cache`, `get_win_totals`, `get_performance`, `get_game_log`.
- Constants: `TEAM_COLORS`, `TEAMS` (15 WNBA tricodes), `_PCT_COLS`, `_INT_COLS`.
- **Inputs:** `data/player_store.csv`, `data/pace_cache.csv`, `data/game_log.csv`, performance CSVs, game/season CSVs in `RAPM_DIR`, raw PBP JSON in `RAPM_DIR/raw_pbp/`.
- **Outputs:** Streamlit UI only (session state changes; roster module persists user edits).

**`matchup.py`** — Core line-origination engine.
- `project_matchup(home_lineup, away_lineup, ...)`: takes `{player_id: minutes}` dicts, computes minutes-weighted oRAPM/dRAPM per team, applies HCA, returns `{spread, total, home_pts, away_pts, win_prob_home, pace, ortg, drtg, ...}`.
- `predict_game_pace_v2()`: Bayesian regression of team pace from `pace_stats.csv`.
- `ml_from_prob()`: win-probability → American odds.
- Constants: `PACE_DEFAULT=80`, `HCA=2.2`, `LEAGUE_ORTG=107.3`, `BASE_PTS=102.8`, `SIGMA_MARGIN=12.5`.
- Formulas: `home_pts = home_adj_ortg * pace / 100 + HCA/2`; `win_prob = norm.cdf(spread / SIGMA_MARGIN)`.
- **Inputs:** `data/pace_stats.csv`, player_store DataFrame, lineup dicts.
- **Outputs:** projection dict.

**`win_totals.py`** — Season win-total projections.
- Pythagorean win % with exponent `PYTH_EXP=10.80`, scaled to a 44-game schedule.
- Zero-sum rescaling (total league wins = 308) to ensure projections sum correctly.
- Threshold probabilities (`≥30W`, `≥32W`, `<15W`, etc.) using `SIGMA_WINS=8.5`.
- `fair_ml_win_total()` prices over/under for a posted line.
- **Inputs:** `player_store` DataFrame.
- **Outputs:** DataFrame with `team, orapm, drapm, net_rapm, proj_wins, p_ge_30, p_ge_32, p_lt_15, ...`.

#### Data Pipeline

**`ingest.py`** — Daily WNBA ETL.
- `fetch_games`, `fetch_pbp`, `fetch_box_score`: thin wrappers around `nba_api` with `league_id="10"` for WNBA.
- `update_stints()` parses new PBPs and appends to `stints_2026_RS.csv`.
- `update_player_minutes()` replaces the 2026 slice of `player_minutes.csv` from box scores.
- `run_ec_scraper()` launches a Playwright scraper for the EC ratings page.
- Conditionally refits RAPM when ≥`MIN_NEW_POSS=500` possessions have been added.
- **Inputs:** existing stints, player_minutes, RAPM CSVs; nba_api.
- **Outputs:** updated stints, player_minutes, `data/ec_2026.csv`, optionally `data/rapm_2026_RS.csv`, then rebuilds `player_store.csv`.
- Constants: `WNBA_LEAGUE_ID="10"`, `CURRENT_YEAR=2026`, `MIN_NEW_POSS=500`.

**`sync_data.py`** — Mirrors the latest analysis CSVs from the external `WNBA_RAPM/analysis/` directory into `wnba_origination/data/`. Triggered by the "Refresh data" button in the app sidebar. Synced files include `pace_stats`, `ft_decomp`, `bonus_by_quarter`, `foul_violation_rates`, `analysis_player_stats`, season game CSVs, `stints_rich`, and 1-yr + 3-yr RAPM CSVs.

**`refresh.py`** — Rebuild app-side caches after the upstream RAPM notebook produces new data. CLI: `python refresh.py [--year 2025] [--all]`. Rebuilds `player_store.csv`, `game_log.csv`, optionally `pace_cache.csv`.

**`paths.py`** — Central path config. Defines `HERE`, `DATA`, `RAPM_DIR` (default Windows path, overridable via `$WNBA_RAPM_DIR`), `PLAYER_STORE`, `EC_2026`, `RAPM_2026`, `EC_ALL_SEASONS`, plus helpers `rapm_season()`, `stints()`, `rapm_8factor()`.

**`fetch_hth.py`** — Scrapes RAPM data from `helpthehelper.vercel.app`. `fetch_page()` downloads the HTML for a season; `extract_players()` parses the embedded `window.PLAYERS = [...]` JavaScript array; saves to `data/hth_players_2026.csv` for validation/comparison.

**`_load_rosters.py`** — One-off bootstrapping script. Hard-codes a 2026 roster CSV (player, team, position, depth), fuzzy-matches to `player_store.csv` by last name, assigns 28 min to starters / 12 min to backups, and writes to `data/rotation_overrides.csv` via `roster.set_rotation()`. Not meant to be re-run.

#### Models & Computation

**`player_store.py`** — Builds the unified player table.
- `build()` merges RAPM from multiple sources, team assignments, and EC stats.
- RAPM priority: 2026 (if ≥300 possessions) → 2025 → 8-factor reconstructed → 0.0.
- Team priority: 2026 `analysis_player_stats` → legacy `player_minutes`.
- `_derive_2026_minutes()` re-derives 2026 minutes from possession-based calculations.
- `_load_team_map_2026()` fixes legacy seconds-scale bugs in minutes columns.
- **Inputs:** RAPM CSVs (2026 RS, 2025 1yr + 3yr), 8-factor RAPM, `player_minutes.csv`, `player_names.csv`, `analysis_player_stats.csv`, `ec_2026.csv` / `ec_all_seasons.csv`.
- **Outputs:** `data/player_store.csv` with `player_id, player_name, team_abbr, minutes, minutes_per_game, orapm, drapm, net_rapm_reconstructed, oec, dec, ec`.

**`rapm.py`** — Ridge-regression RAPM fitter.
- Two separate `Ridge(alpha=2000)` regressions: offensive (X=1 if player on offense, y=points scored) and defensive (X=1 if on defense, y=points allowed; coefficients negated).
- Uses a sparse design matrix (`scipy.sparse`) for memory efficiency.
- Filters output to players with ≥`MIN_POSS_OUTPUT=100` possessions.
- **Inputs:** `stints_*.csv` (columns: `off_team, def_team, off_p1-5, def_p1-5, points`).
- **Outputs:** `data/rapm_{year}_RS.csv` with `player_id, orapm, drapm, net_rapm, poss, season, season_type`.
- Constants: `LAMBDA=2000` (ridge penalty on possession scale).

**`pace.py`** — Dynamic pace prediction.
- Model: `team_pace = median(poss/game) regressed 25% toward league mean`; `player_residual = avg pace when player played − team's median pace`; `lineup_pace = team_pace + minutes-weighted sum of residuals`.
- `build_pace_cache()` uses 2021–2025 stints with year weights `{2021:0.5, …, 2025:1.2}`.
- `predict_lineup_pace()` / `predict_game_pace()` are the runtime entry points.
- **Inputs:** stints files for 2021–2025.
- **Outputs:** `data/pace_cache.csv` (`player_id, team_id, team_pace, player_residual`).
- Constants: `LEAGUE_MEAN_REGRESSION=0.25`, `STINTS_YEARS=[2021..2025]`, `YEAR_WEIGHTS`.

**`game_log.py`** — Per-team-per-game four-factors + pace.
- `_compute_factors_from_pbp()` extracts FGA, FGM, FTA, FTM, 3PA, 3PM, OREB, DREB, TOV from raw PBP JSON.
- `_four_factors()` computes eFG%, TOV%, OREB%, FT Rate.
- `_pace_from_stints()` uses 1 row = 1 possession (so `poss = rows / 2`).
- **Inputs:** raw PBP JSON in `RAPM_DIR/raw_pbp/`, season `games_*.csv`, stints files.
- **Outputs:** `data/game_log.csv` (2 rows per game, prefixed `h_` and `a_`): `game_id, season, game_date, home_team_id, away_team_id, pts, margin, pace, poss, ot_periods, efg, tov, oreb, ftr, fga, fgm, tpa, tpm, fta, ftm, oreb_n, dreb_n, tov_n`.

**`lineup.py`** — 5-man unit net ratings.
- `recent_game_rotations()` returns per-player minutes over the last N games.
- `lineup_net_rating()` computes ORtg/DRtg for an exact 5-man unit from stints.
- `rotation_for_app()` returns top-12 players by possession share for the UI.
- Caveat: stints only track ~83% of possessions due to upstream lineup-tracking gaps, so lineup-level projections are less reliable than player-level.
- **Inputs:** `stints_*.csv`, `player_minutes`, `player_names`.
- **Outputs:** DataFrame with `player_id, player_name, games, total_poss, poss_per_game, min_share`.

**`league_stats.py`** — Aggregated league/team statistics.
- `league_baselines()`: per-team-game means for pace, ORTG, PTS, FGA, FTA, TOV, OREB, FT/FGA, transition %, bonus %.
- `team_profiles()`: per-team season aggregates including opponent-faced DRTG.
- `recent_games()`, `bonus_summary()` (% team-quarters reaching 5-foul bonus), `foul_rates()`.
- `regress()`: Bayesian shrinkage `posterior = (n*observed + k*league_mean) / (n + k)`.
- Normalizes legacy team tricodes (`PDX→POR`, `PHO→PHX`).
- Season keys: `{"2025":"2025_full","2026":"2026_first8"}`.
- **Inputs:** `pace_stats.csv`, `ft_decomp.csv`, `bonus_by_quarter.csv`, `foul_violation_rates.csv`, `games_*.csv`, `stints_rich_*.csv`.
- **Outputs:** DataFrames cached for 10 min in Streamlit.

**`league_dashboard.py`** — Streamlit tab views (mostly caching wrappers around `league_stats`):
- `_view_season_context()` — 2025 vs 2026 baselines with delta arrows.
- `_view_game_log()` — sortable recent games table.
- `_view_team_profiles()` — per-team season stats with team badges.
- `_view_bonus_fts()` — bonus-reach % by quarter, foul rates.
- `render()` — entry point from `app.py`'s sidebar.

**`performance.py`** — Backtests model spread/total against actuals + market.
- `load_bdb()` loads BigDataBall CSVs (2024-2026).
- `_reshape_to_game_level()` pivots BDB's 2-row format to 1 row per game.
- `compute_model_projections()` re-projects each historical game with the current RAPM and computes errors.
- `summary_stats()` returns RMSE, ATS%, O/U% for both model and market.
- **Inputs:** `data/bigdataball/*.csv`.
- **Outputs:** game-level DataFrame with `model_spread, model_total, model_spread_err, model_total_err, model_ats, model_ou`.

#### Utilities & Persistence

**`roster.py`** — Persistent override layer (survives across sessions).
- `apply()` merges overrides onto `player_store`.
- `upsert_player()` / `remove_player()`: CRUD for player edits.
- `set_rotation()` / `get_rotation()` / `clear_rotation()`: team rotation overrides.
- `parse_rotation_csv()` accepts uploaded CSVs with flexible column naming.
- **Schemas:**
  - `data/roster_overrides.csv`: `player_id, player_name, team_abbr, orapm_override, drapm_override, notes`.
  - `data/rotation_overrides.csv`: `team_abbr, player_id, player_name, projected_minutes`.

**`verify_minutes.py`** — Sanity check that compares PBP-walker minute counts (from `app.py:_game_presence_from_pbp()`) against ESPN box-score minutes. Prints per-player deltas, distribution buckets, and the top 20 worst mismatches.
- **Inputs:** `data/espn_box_2026.tsv`, `RAPM_DIR/games_2026_Regular_Season.csv`, raw PBP JSONs.
- **Outputs:** console report.

#### React Component (`components/rotation_chart/`)

A drag-and-drop minute editor embedded in the Streamlit app. Built with React 18 + TypeScript + Vite, using `streamlit-component-lib` for bidirectional state. Pre-built static assets live in `frontend/build/`.

---

## 2. Package Dependencies

### Python (from `wnba_origination/requirements.txt`)

| Package | Pinned Version | Used For |
|---|---|---|
| `nba_api` | 1.11.4 | WNBA game logs, PBP, box scores (league_id=10) |
| `numpy` | 2.1.3 | Array math across all numeric code |
| `pandas` | 2.2.3 | DataFrame manipulation everywhere |
| `playwright` | 1.57.0 | Headless browser for EC ratings scraper (`ingest.py`) |
| `plotly` | 6.5.2 | Charts in Streamlit dashboards |
| `scipy` | 1.17.1 | Sparse matrices for RAPM, `norm.cdf` for win probability |
| `scikit-learn` | 1.8.0 | `Ridge` regression for RAPM fitting |
| `streamlit` | 1.54.0 | Web UI for the line-origination app and shots dashboard |

### Additional Python (imported but not pinned in `requirements.txt`)

| Package | Used By | Used For |
|---|---|---|
| `cbbd` | `daily_fetch.py`, `audit_season.py`, `race_to_10_analysis.py` | CollegeBasketballData.com API SDK |
| `python-dotenv` (implicit via `os.environ`) | `daily_fetch.py`, `audit_season.py`, `race_to_10_analysis.py` | Load `CBBD_API_KEY` from `.env` |
| `zoneinfo` (stdlib, Py ≥3.9) | `daily_fetch.py` | Eastern-time conversions for "yesterday" detection |

### JavaScript / Node (from `wnba_origination/components/rotation_chart/frontend/package.json`)

| Package | Version | Role |
|---|---|---|
| `react` | ^18.2.0 | UI framework for the rotation chart component |
| `react-dom` | ^18.2.0 | DOM bindings |
| `streamlit-component-lib` | ^2.0.0 | Streamlit ↔ React communication |
| `typescript` | ^5.0.0 | Type-safe component code (dev) |
| `vite` | ^5.0.0 | Build tool (dev) |
| `@vitejs/plugin-react` | ^4.2.0 | React plugin for Vite (dev) |
| `@types/react`, `@types/react-dom` | ^18.2.0 | TypeScript types (dev) |

---

## 3. Potential Improvements

### Performance

- **Switch CSVs to Parquet.** The `cbbd_data/` and `wnba_origination/data/` directories accumulate dozens of daily CSV files that get re-globbed and deduplicated on every dashboard load. Parquet would cut load times substantially and shrink disk footprint, especially for `pbp_flat/` and `possessions/`.
- **Parallelize `daily_fetch.py`.** Each game is processed sequentially. Using `concurrent.futures.ThreadPoolExecutor` for the API calls (I/O-bound) and `ProcessPoolExecutor` for the possession tracker (CPU-bound) would speed up backfills materially.
- **Cache the deduplicated tables.** `load_data.py` re-reads and re-dedups all CSVs on every import. Persist a single deduplicated Parquet snapshot keyed off the most recent file's mtime.
- **Vectorize the PBP walker.** `app.py:_game_presence_from_pbp()` iterates play-by-play in Python. A pandas-vectorized version using cumulative `subbing_in/out` events would scale far better when computing presence grids across many games.

### Code Quality

- **Consolidate duplicated possession logic.** The possession state machine appears in three places: `daily_fetch.py`, `fix_possessions.py`, and the two batch notebooks. Extract into a single `cbbd_data/possessions.py` module so bug fixes only have to land once.
- **Decouple hardcoded paths.** `wnba_origination/paths.py` hardcodes a Windows path (`C:/...../WNBA_RAPM/...`) as the default. Move all paths into a `config.yaml` / `.env` and require explicit overrides on non-Windows systems.
- **Add type hints throughout.** Most modules lack annotations; adding them would let mypy/pyright catch shape errors in DataFrame columns and dict-based payloads.
- **Extract magic numbers.** Constants like `HCA=2.2`, `SIGMA_MARGIN=12.5`, `PYTH_EXP=10.80`, and `LAMBDA=2000` live in module-level scope. A single `constants.py` (or `config.py`) would make tuning easier.
- **Replace the embedded roster CSV** in `_load_rosters.py` with an external `rosters/2026.csv` file. The current setup couples roster data to a Python source file.

### Testing

- **No test suite exists.** Highest-value additions:
  - Unit tests for `rapm.fit_rapm()` against synthetic stints with known ground-truth coefficients.
  - Unit tests for `matchup.project_matchup()` to lock in spread / total math at known inputs (catches accidental constant changes).
  - Unit tests for the possession state machine on a small fixture PBP file (catches the kinds of bugs `fix_possessions.py` exists to patch).
  - Property-based tests for `win_totals.project_all_teams()` enforcing the zero-sum invariant.

### Data Pipeline

- **Automate `sync_data.py`.** Currently triggered by a manual sidebar button; running it on a cron alongside `daily_fetch.py` would eliminate the "did you remember to sync?" failure mode.
- **Add data validation gates.** Each pipeline stage could assert minimum row counts, non-null primary keys, and schema conformance (e.g., via `pandera` or a lightweight homegrown validator). Failures in `daily_fetch.py` currently surface only as missing rows downstream.
- **Cache CBBD API responses.** A simple disk cache keyed by `(endpoint, params)` would make backfills idempotent and dramatically faster on retry.
- **Capture upstream version.** Record the git SHA of the `WNBA_RAPM` notebook that produced each set of analysis CSVs so projection drift can be attributed.

### Architecture

- **Containerize.** Provide a `Dockerfile` that pins the Python version, installs requirements, and launches the Streamlit app — eliminates the Windows-path coupling and makes the app deployable to any host.
- **Add CI/CD.** GitHub Actions could run the test suite plus a smoke-test of `daily_fetch.py` against a recorded fixture.
- **Promote the React component build artifact.** Pre-built assets live in `frontend/build/`; commit a `Makefile` / `npm script` documentation so future contributors know how to rebuild.
- **Documentation refresh.** The root `README.md` describes the shots dashboard but does not mention `wnba_origination/` or `daily_fetch.py`. A top-level overview pointing readers at the three sub-projects would help newcomers.

---

## 4. Primary Data Sources

| Source | Type | Used By | Data Provided |
|---|---|---|---|
| **NBA API** (`nba_api` SDK, `league_id="10"`) | REST API | `wnba_origination/ingest.py`, `wnba_origination/app.py` (via cached PBP JSONs) | WNBA game logs, play-by-play, box scores |
| **CollegeBasketballData.com (CBBD)** | REST API + `cbbd` SDK | `daily_fetch.py`, `audit_season.py`, `race_to_10_analysis.py`, `cbbd_*` notebooks | NCAA D1 games, plays, rosters (`game_players`), lines (spread / OU / ML), team metadata |
| **helpthehelper.vercel.app** | Public web page (HTML scrape) | `wnba_origination/fetch_hth.py` | Third-party WNBA RAPM and efficiency stats — used to cross-check the internal model |
| **BigDataBall** (manually downloaded CSVs) | Historical CSV files | `wnba_origination/performance.py` | Historical WNBA spreads, totals, moneylines, actual scores — drives the model backtest |
| **ESPN box scores** (TSV) | Web-export TSV | `wnba_origination/verify_minutes.py` | Player minutes per game — ground truth for validating the PBP walker |
| **WNBA_RAPM notebook** (external sibling repo) | Local CSV outputs | `wnba_origination/sync_data.py`, `player_store.py`, `game_log.py`, `pace.py`, `rapm.py`, `league_stats.py` | Raw PBP JSON (`raw_pbp/*.json`), stints, season game CSVs, RAPM coefficients (1yr + 3yr), 8-factor RAPM, EC ratings, pace stats, free-throw decomposition, foul/violation rates, bonus-by-quarter |
| **Internal Playwright scraper** (`ingest.py:run_ec_scraper`) | Headless browser | `wnba_origination/ingest.py` | Expected Contribution (EC) ratings, parsed from a JavaScript-rendered page and persisted to `data/ec_2026.csv` |
| **User input** (Streamlit UI, hardcoded rosters) | Manual | `wnba_origination/app.py`, `_load_rosters.py`, `roster.py` | Trade overrides, RAPM edits, projected minutes, depth charts — persisted in `roster_overrides.csv` and `rotation_overrides.csv` |
