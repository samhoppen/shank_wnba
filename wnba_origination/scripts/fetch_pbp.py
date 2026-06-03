"""
fetch_pbp.py — download raw WNBA play-by-play JSON for the streamlit app.

Raw PBP files (~400 MB) are gitignored. The app's rotation grid heatmaps,
per-game box scores, and analysis regenerator need them, so this script
populates the local cache (default location: <repo>/wnba_rapm_cache/raw_pbp/).

Writes two files per game:
    {game_id}_pbp.json       — list[dict] events normalised to snake_case
    {game_id}_starters.json  — {str(team_id): [player_id, ...]}

Usage:
    python -m wnba_origination.scripts.fetch_pbp --year 2026
    python -m wnba_origination.scripts.fetch_pbp --years 2017-2026
    python -m wnba_origination.scripts.fetch_pbp --game-ids 1022500001 1022500002
    python -m wnba_origination.scripts.fetch_pbp --year 2026 --force
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Allow `python scripts/fetch_pbp.py` and `python -m wnba_origination.scripts.fetch_pbp`
HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from paths import RAPM_DIR, RAW_PBP_DIR  # noqa: E402

WNBA_LEAGUE_ID = "10"
SLEEP_SEC = 0.6


# ── nba_api helpers ──────────────────────────────────────────────────────────

def _fetch_game_ids(year: int) -> list[str]:
    """Prefer the games CSV shipped in wnba_rapm/; fall back to nba_api."""
    games_csv = RAPM_DIR / f"games_{year}_Regular_Season.csv"
    if games_csv.exists():
        df = pd.read_csv(games_csv, usecols=["game_id"], dtype={"game_id": str})
        return sorted(df["game_id"].astype(str).str.zfill(10).unique().tolist())

    from nba_api.stats.endpoints import leaguegamelog
    gl = leaguegamelog.LeagueGameLog(
        league_id=WNBA_LEAGUE_ID,
        season=str(year),
        season_type_all_star="Regular Season",
    )
    df = gl.get_data_frames()[0]
    return sorted(df["GAME_ID"].astype(str).str.zfill(10).unique().tolist())


_COL_RENAME = {
    "actionType": "action_type",
    "subType": "sub_type",
    "teamId": "team_id",
    "personId": "person_id",
    "playerName": "playerName",
    "playerNameI": "playerNameI",
    "teamTricode": "teamTricode",
    "scoreHome": "score_home",
    "scoreAway": "score_away",
    "pointsTotal": "pointsTotal",
    "shotValue": "shotValue",
    "shotResult": "shotResult",
    "isFieldGoal": "isFieldGoal",
    "description": "description",
    "period": "period",
    "clock": "clock",
}


def _normalise_event(ev: dict) -> dict:
    """Convert PlayByPlayV3 camelCase to the snake_case schema the app reads."""
    out: dict = {}
    for k, v in ev.items():
        out[_COL_RENAME.get(k, k)] = v
    return out


def _fetch_pbp(game_id: str) -> list[dict]:
    from nba_api.stats.endpoints import playbyplayv3
    pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
    df = pbp.get_data_frames()[0]
    return [_normalise_event(r) for r in df.to_dict(orient="records")]


def _starters_from_events(events: list[dict]) -> dict[str, list[int]]:
    """Derive starters as the first 5 distinct non-substitution player_ids per team."""
    seen: dict[int, list[int]] = {}
    for ev in events:
        try:
            tid = int(ev.get("team_id", 0) or 0)
        except Exception:
            tid = 0
        if not tid:
            continue
        if ev.get("action_type") == "Substitution":
            continue
        pid_raw = ev.get("person_id", 0)
        try:
            pid = int(pid_raw or 0)
        except Exception:
            pid = 0
        if not pid:
            continue
        bucket = seen.setdefault(tid, [])
        if pid not in bucket and len(bucket) < 5:
            bucket.append(pid)
    return {str(tid): pids for tid, pids in seen.items()}


# ── Main ─────────────────────────────────────────────────────────────────────

def _parse_year_range(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(spec)]


def run(
    game_ids: list[str],
    out_dir: Path = RAW_PBP_DIR,
    force: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_new = n_skipped = n_failed = 0

    for i, gid in enumerate(game_ids, 1):
        gid = str(gid).zfill(10)
        pbp_path = out_dir / f"{gid}_pbp.json"
        sp_path = out_dir / f"{gid}_starters.json"
        if pbp_path.exists() and sp_path.exists() and not force:
            n_skipped += 1
            continue

        print(f"  [{i}/{len(game_ids)}] {gid}", end=" ... ", flush=True)
        try:
            time.sleep(SLEEP_SEC)
            events = _fetch_pbp(gid)
            if not events:
                print("empty PBP — skipped")
                n_failed += 1
                continue
            with open(pbp_path, "w", encoding="utf-8") as f:
                json.dump(events, f)
            with open(sp_path, "w", encoding="utf-8") as f:
                json.dump(_starters_from_events(events), f)
            n_new += 1
            print(f"{len(events)} events")
        except Exception as exc:
            n_failed += 1
            print(f"ERROR: {exc}")

    return {"new": n_new, "skipped": n_skipped, "failed": n_failed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch raw WNBA PBP JSON.")
    parser.add_argument("--year", type=int)
    parser.add_argument("--years", help="Inclusive range, e.g. 2017-2026")
    parser.add_argument("--game-ids", nargs="+", help="Explicit game IDs")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if cached")
    parser.add_argument("--out-dir", type=Path, default=RAW_PBP_DIR,
                        help=f"Output directory (default: {RAW_PBP_DIR})")
    args = parser.parse_args()

    if args.game_ids:
        game_ids = args.game_ids
    else:
        years: list[int] = []
        if args.year:
            years.append(args.year)
        if args.years:
            years.extend(_parse_year_range(args.years))
        if not years:
            parser.error("provide --year, --years, or --game-ids")

        game_ids = []
        for y in years:
            ids = _fetch_game_ids(y)
            print(f"{y}: {len(ids)} games")
            game_ids.extend(ids)

    print(f"Writing PBP cache to {args.out_dir}")
    summary = run(game_ids, out_dir=args.out_dir, force=args.force)
    print(
        f"\nDone — new: {summary['new']}, skipped: {summary['skipped']}, "
        f"failed: {summary['failed']}"
    )


if __name__ == "__main__":
    main()
