"""
Model vs. market performance tracking.

Loads BigDataBall CSVs (2024, 2025, 2026) and joins with model projections
to evaluate spread and total accuracy vs. actuals and vs. opening/closing lines.

BDB CSV format (one row per team per game):
  GAME-ID, DATE, TEAMS, VENUE (Home/Road), F (final score),
  POSS, PACE, OEFF, DEFF, TEAM REST DAYS,
  OPENING SPREAD, CLOSING SPREAD, OPENING O/U, CLOSING O/U,
  OPENING MONEYLINE, CLOSING MONEYLINE

Usage:
  python performance.py           # print summary stats
  df = load_results()             # load joined DataFrame
"""
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from paths import BDB_DIR

# Column aliases — BDB headers can have leading spaces
_RENAME = {
    "GAME-ID": "game_id",
    "DATE": "date",
    "TEAMS": "teams",
    "VENUE": "venue",
    "F": "score",
    "POSS": "poss",
    "PACE": "pace",
    "OEFF": "oeff",
    "DEFF": "deff",
    "TEAM REST DAYS": "rest",
    "OPENING SPREAD": "open_spread",
    "CLOSING SPREAD": "close_spread",
    "OPENING O/U": "open_ou",
    "CLOSING O/U": "close_ou",
    "OPENING MONEYLINE": "open_ml",
    "CLOSING MONEYLINE": "close_ml",
}


def load_bdb(years: list[int] = [2024, 2025, 2026]) -> pd.DataFrame:
    """Load and concatenate all available BigDataBall CSVs."""
    frames = []
    for yr in years:
        path = BDB_DIR / f"{yr}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={k: v for k, v in _RENAME.items() if k in df.columns})
        df["season"] = yr
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    bdb = pd.concat(frames, ignore_index=True)

    # Parse numeric columns
    for col in ["open_spread", "close_spread", "open_ou", "close_ou", "pace", "oeff", "deff"]:
        if col in bdb.columns:
            bdb[col] = pd.to_numeric(bdb[col], errors="coerce")

    return bdb


def _reshape_to_game_level(bdb: pd.DataFrame) -> pd.DataFrame:
    """
    BDB has two rows per game (one per team). Pivot to one row per game:
    home_team, away_team, home_score, away_score, home_spread (opening), total_ou, etc.
    """
    home = bdb[bdb["venue"].str.strip().str.lower() == "home"].copy()
    away = bdb[bdb["venue"].str.strip().str.lower() == "road"].copy()

    def parse_score(s):
        try:
            parts = str(s).split("-")
            return int(parts[0]), int(parts[1])
        except Exception:
            return None, None

    home[["home_pts", "away_pts"]] = home["score"].apply(
        lambda s: pd.Series(parse_score(s))
    )
    home["actual_margin"] = home["home_pts"] - home["away_pts"]
    home["actual_total"] = home["home_pts"] + home["away_pts"]

    game = home[["game_id", "date", "season", "teams",
                 "home_pts", "away_pts", "actual_margin", "actual_total",
                 "poss", "pace", "oeff", "deff",
                 "open_spread", "close_spread", "open_ou", "close_ou"]].copy()

    # Home spread convention: negative = home favored
    game["home_covered"] = (
        game["actual_margin"] + game["close_spread"] > 0
    ).astype(int)
    game["over_hit"] = (game["actual_total"] > game["close_ou"]).astype(int)

    return game


def compute_model_projections(game_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add model spread + total projections to the game-level DataFrame.
    Requires team name parsing from the 'teams' column.

    This is a retrospective backtest — for each game, re-project using
    historical RAPM ratings for that season. Currently uses 2025 ratings
    for all historical games (best available baseline).
    """
    import player_store as ps
    import matchup as mx

    store = ps.load()

    proj_spreads, proj_totals = [], []
    for _, row in game_df.iterrows():
        # Parse team abbrs from 'teams' column (format: "NYL vs LVA")
        teams_str = str(row.get("teams", ""))
        parts = [p.strip() for p in teams_str.replace(" vs ", "/").replace("@", "/").split("/")]
        if len(parts) < 2:
            proj_spreads.append(np.nan)
            proj_totals.append(np.nan)
            continue

        home_abbr, away_abbr = parts[0], parts[1]
        try:
            pace = float(row["pace"]) if pd.notna(row.get("pace")) else mx.PACE_DEFAULT
            result = mx.project_by_team(home_abbr, away_abbr, pace=pace, store=store)
            proj_spreads.append(result["spread"])
            proj_totals.append(result["total"])
        except Exception:
            proj_spreads.append(np.nan)
            proj_totals.append(np.nan)

    game_df = game_df.copy()
    game_df["model_spread"] = proj_spreads
    game_df["model_total"] = proj_totals
    game_df["model_spread_err"] = game_df["model_spread"] - game_df["actual_margin"]
    game_df["model_total_err"] = game_df["model_total"] - game_df["actual_total"]
    game_df["model_ats"] = (
        (game_df["actual_margin"] + game_df["model_spread"] > 0)
        .astype(int)
    )
    game_df["model_ou"] = (
        (game_df["actual_total"] > game_df["model_total"])
        .astype(int)
    )
    return game_df


def summary_stats(results: pd.DataFrame) -> dict:
    """Compute RMSE, ATS%, O/U% for model and market."""
    valid = results.dropna(subset=["model_spread", "actual_margin"])
    if valid.empty:
        return {}

    return {
        "n_games": len(valid),
        "model_spread_rmse": round(np.sqrt(np.mean(valid["model_spread_err"] ** 2)), 2),
        "model_total_rmse": round(np.sqrt(np.mean(valid["model_total_err"].dropna() ** 2)), 2),
        "model_ats_pct": round(valid["model_ats"].mean() * 100, 1),
        "market_ats_push_pct": round(valid["home_covered"].mean() * 100, 1),
        "model_ou_pct": round(valid["model_ou"].mean() * 100, 1),
        "over_hit_pct": round(valid["over_hit"].mean() * 100, 1),
    }


def load_results() -> pd.DataFrame:
    bdb = load_bdb()
    if bdb.empty:
        return pd.DataFrame()
    games = _reshape_to_game_level(bdb)
    return compute_model_projections(games)


if __name__ == "__main__":
    results = load_results()
    if results.empty:
        print("No BigDataBall data found. Drop CSVs into wnba_origination/data/bigdataball/")
    else:
        stats = summary_stats(results)
        print("\n=== Model Performance ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print(f"\n{len(results)} games loaded")
        print(results[["date", "teams", "actual_margin", "model_spread",
                        "actual_total", "model_total"]].head(10).to_string())
