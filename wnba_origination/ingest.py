"""
Daily data refresh pipeline.

Steps:
  1. Fetch 2026 WNBA games + PBP from nba_api (league_id="10")
  2. Build stints from PBP, append to stints_2026_RS.csv
  3. Update player_minutes for 2026
  4. Run EC scraper -> ec_2026.csv
  5. Refit RAPM if new data crosses MIN_NEW_POSS threshold
  6. Rebuild player_store

Run:
  python ingest.py                  # full refresh
  python ingest.py --skip-ec        # skip Playwright EC scraper (faster)
  python ingest.py --skip-rapm      # skip RAPM refit
"""
import argparse
import subprocess
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta

from paths import (
    RAPM_DIR, DATA, EC_SCRAPER, PYTHON, PLAYER_MINUTES,
    stints, rapm_season, EC_2026,
)

WNBA_LEAGUE_ID = "10"
CURRENT_YEAR = 2026
MIN_NEW_POSS = 500   # refit RAPM only when this many new possessions accumulated

# ── nba_api helpers ─────────────────────────────────────────────────────────

def _nba_api_available() -> bool:
    try:
        import nba_api  # noqa: F401
        return True
    except ImportError:
        return False


def fetch_games(season: str = "2026") -> pd.DataFrame:
    """Fetch WNBA game log for the season."""
    from nba_api.stats.endpoints import leaguegamelog
    gl = leaguegamelog.LeagueGameLog(
        league_id=WNBA_LEAGUE_ID,
        season=season,
        season_type_all_star="Regular Season",
    )
    return gl.get_data_frames()[0]


def fetch_pbp(game_id: str) -> pd.DataFrame:
    """Fetch play-by-play for a single game."""
    from nba_api.stats.endpoints import playbyplayv3
    pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
    return pbp.get_data_frames()[0]


def fetch_box_score(game_id: str) -> pd.DataFrame:
    """Fetch player-level box score for minutes tracking."""
    from nba_api.stats.endpoints import boxscoretraditionalv3
    box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
    return box.get_data_frames()[0]


def _parse_stints_from_pbp(pbp: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """
    Convert raw PBP into possession-level stints (same schema as existing stints files).
    Returns DataFrame with columns: game_id, poss_id, off_team, def_team, points,
                                    off_p1..off_p5, def_p1..def_p5
    This is a simplified version — mirrors logic from the existing RAPM pipeline.
    """
    # Columns vary between WNBA PBP versions; try both naming schemes
    home_team_col = next((c for c in pbp.columns if "hometeamid" in c.lower()), None)
    away_team_col = next((c for c in pbp.columns if "awayteamid" in c.lower()), None)

    if home_team_col is None:
        # Can't parse without team IDs in PBP — skip this game
        return pd.DataFrame()

    # Delegate to existing RAPM pipeline parser if available
    try:
        sys.path.insert(0, str(RAPM_DIR.parent))
        from _combined import parse_game_stints  # existing pipeline helper
        return parse_game_stints(pbp, game_id)
    except (ImportError, AttributeError):
        pass

    # Minimal fallback: return empty (tell caller to use box score only for minutes)
    return pd.DataFrame()


def update_stints(season: str = "2026") -> int:
    """
    Fetch new games, parse stints, append to stints_2026_RS.csv.
    Returns number of new possessions added.
    """
    if not _nba_api_available():
        print("  [WARN] nba_api not installed — skipping stints update")
        return 0

    stints_path = stints(CURRENT_YEAR)
    stints_path.parent.mkdir(parents=True, exist_ok=True)

    existing_games: set[str] = set()
    if stints_path.exists():
        existing = pd.read_csv(stints_path, usecols=["game_id"])
        existing_games = set(existing["game_id"].astype(str).unique())

    games = fetch_games(season)
    new_games = games[~games["GAME_ID"].astype(str).isin(existing_games)]
    print(f"  {len(new_games)} new games to process")

    all_new_stints = []
    for game_id in new_games["GAME_ID"].astype(str):
        try:
            pbp = fetch_pbp(game_id)
            game_stints = _parse_stints_from_pbp(pbp, game_id)
            if not game_stints.empty:
                all_new_stints.append(game_stints)
        except Exception as e:
            print(f"    [WARN] game {game_id}: {e}")

    if not all_new_stints:
        return 0

    combined = pd.concat(all_new_stints, ignore_index=True)
    if stints_path.exists():
        combined.to_csv(stints_path, mode="a", header=False, index=False)
    else:
        combined.to_csv(stints_path, index=False)

    n_new = len(combined)
    print(f"  Added {n_new:,} new possessions to {stints_path}")
    return n_new


def update_player_minutes(season: str = "2026") -> None:
    """Update player_minutes.csv with 2026 box score data."""
    if not _nba_api_available():
        print("  [WARN] nba_api not installed — skipping minutes update")
        return

    games = fetch_games(season)
    rows = []
    for game_id in games["GAME_ID"].astype(str).unique():
        try:
            box = fetch_box_score(game_id)
            rows.append(box)
        except Exception as e:
            print(f"    [WARN] box score {game_id}: {e}")

    if not rows:
        return

    box_all = pd.concat(rows, ignore_index=True)
    # Normalize column names (v3 uses camelCase)
    box_all.columns = [c.upper() for c in box_all.columns]

    pm = pd.read_csv(PLAYER_MINUTES)
    # Remove existing 2026 rows and replace
    pm = pm[pm["season"] != CURRENT_YEAR]

    # Aggregate box scores to season totals
    def _safe_min(m):
        try:
            parts = str(m).split(":")
            return int(parts[0]) + int(parts[1]) / 60 if len(parts) == 2 else float(m)
        except Exception:
            return 0.0

    if "MINUTES" in box_all.columns:
        box_all["minutes_float"] = box_all["MINUTES"].apply(_safe_min)
        agg = (box_all.groupby(["PLAYERID", "PLAYERNAME", "TEAMID", "TEAMABBREVIATION"])
                       .agg(games=("GAMEID", "nunique"),
                            minutes=("minutes_float", "sum"))
                       .reset_index()
                       .rename(columns={
                           "PLAYERID": "player_id",
                           "PLAYERNAME": "player_name",
                           "TEAMID": "team_id",
                           "TEAMABBREVIATION": "team_abbr",
                       }))
        agg["minutes_per_game"] = agg["minutes"] / agg["games"]
        agg["season"] = CURRENT_YEAR
        agg["season_type"] = "Regular Season"
        pm = pd.concat([pm, agg], ignore_index=True)
        pm.to_csv(PLAYER_MINUTES, index=False)
        print(f"  Updated player_minutes: {len(agg)} players in {CURRENT_YEAR}")


# ── EC scraper ───────────────────────────────────────────────────────────────

def run_ec_scraper() -> bool:
    """Run the positiveresidual EC scraper for 2026 season."""
    # We use a modified version that only scrapes 2026 and outputs to DATA/ec_2026.csv
    ec_2026_script = DATA.parent / "ingest_ec_2026.py"
    _write_ec_2026_script(ec_2026_script)

    print("  Running EC scraper (Playwright)...")
    result = subprocess.run(
        [PYTHON, str(ec_2026_script)],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"  [WARN] EC scraper failed:\n{result.stderr[-500:]}")
        return False
    print("  EC scraper done")
    return True


def _write_ec_2026_script(path: Path) -> None:
    """Write a minimal EC scraper that fetches only 2026 data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    script = f'''
import json, csv, time
from playwright.sync_api import sync_playwright

URL = "https://www.positiveresidual.com/shiny/wnba/"
OUTPUT = r"{str(EC_2026)}"
YEAR = 2026

def get_ec_table(page):
    return page.evaluate("""
        () => {{
            let vals = window.Shiny && window.Shiny.shinyapp && window.Shiny.shinyapp.$values;
            if (!vals || !vals.ec_table) return null;
            try {{
                let obj = typeof vals.ec_table === 'string' ? JSON.parse(vals.ec_table) : vals.ec_table;
                return obj.x.tag.attribs.data;
            }} catch(e) {{ return {{error: e.message}}; }}
        }}
    """)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(URL, timeout=30000)
    page.wait_for_load_state("networkidle", timeout=20000)
    page.wait_for_timeout(4000)

    for tab in page.query_selector_all(".nav-link, [role=\\'tab\\']"):
        if "estimated" in tab.inner_text().lower() or "contribution" in tab.inner_text().lower():
            tab.click()
            page.wait_for_timeout(2000)
            break

    page.evaluate(f"""
        () => {{
            let slider = $('#ec_season').data('ionRangeSlider');
            if (slider) slider.update({{from: {YEAR}, to: {YEAR}}});
        }}
    """)
    page.wait_for_timeout(300)
    page.click('#ec_min_mp', click_count=3)
    page.keyboard.type('0')
    page.keyboard.press('Tab')
    page.wait_for_timeout(200)

    page.evaluate("""
        () => {{
            let btns = Array.from(document.querySelectorAll('button, input[type=submit]'));
            let btn = btns.find(b => b.innerText &&
                (b.innerText.toLowerCase().includes('submit') ||
                 b.innerText.toLowerCase().includes('update')));
            if (btn) btn.click();
        }}
    """)
    page.wait_for_timeout(4000)

    data = get_ec_table(page)
    browser.close()

if data and "error" not in data:
    cols = list(data.keys())
    rows = list(zip(*[data[c] for c in cols]))
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(cols)
        csv.writer(f).writerows(rows)
    print(f"Saved {{len(rows)}} players -> {{OUTPUT}}")
else:
    print(f"ERROR: {{data}}")
'''
    path.write_text(script, encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────────

def run(skip_ec: bool = False, skip_rapm: bool = False) -> None:
    import player_store
    import rapm as rapm_module

    print("=== WNBA Daily Ingest ===")

    print("\n[1/4] Updating stints + player minutes...")
    n_new_poss = update_stints()
    update_player_minutes()

    if not skip_ec:
        print("\n[2/4] Refreshing EC ratings...")
        run_ec_scraper()
    else:
        print("\n[2/4] EC scraper skipped")

    if not skip_rapm and n_new_poss >= MIN_NEW_POSS:
        print(f"\n[3/4] Refitting RAPM ({n_new_poss:,} new possessions)...")
        rapm_module.run(year=CURRENT_YEAR)
    else:
        reason = "skipped" if skip_rapm else f"only {n_new_poss} new poss (< {MIN_NEW_POSS})"
        print(f"\n[3/4] RAPM refit {reason}")

    print("\n[4/4] Rebuilding player store...")
    player_store.build()

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ec", action="store_true")
    parser.add_argument("--skip-rapm", action="store_true")
    args = parser.parse_args()
    run(skip_ec=args.skip_ec, skip_rapm=args.skip_rapm)
