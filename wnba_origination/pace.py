"""
Dynamic pace prediction: team median + per-player residuals.

Model:
  team_pace        = median possessions/game for that team (regressed 25% to league mean)
  player_residual  = avg pace in games the player appeared in - their team's median
                     (captures "this player speeds up / slows down the game" independent of team)
  lineup_pace      = team_median + minutes-weighted sum of player residuals for the 5 on floor
  game_pace        = average of home_lineup_pace and away_lineup_pace

All computed from existing stints files (2021-2025 by default, weighted toward recent).
Results cached to data/pace_cache.csv — rebuilt by calling build_pace_cache().
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from paths import RAPM_DIR, DATA, stints

PACE_CACHE = DATA / "pace_cache.csv"
LEAGUE_MEAN_REGRESSION = 0.25   # regress team median 25% toward league mean
STINTS_YEARS = [2021, 2022, 2023, 2024, 2025]
YEAR_WEIGHTS = {2021: 0.5, 2022: 0.6, 2023: 0.8, 2024: 1.0, 2025: 1.2}


def _load_all_stints(years: list[int] = STINTS_YEARS) -> pd.DataFrame:
    frames = []
    for yr in years:
        p = stints(yr)
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["year"] = yr
        df["weight"] = YEAR_WEIGHTS.get(yr, 1.0)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No stints files found")
    return pd.concat(frames, ignore_index=True)


def _poss_per_game(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weighted possessions per game for each team.
    Each row in stints is one possession, so count rows per (game, team) on offense.
    """
    off_counts = (df.groupby(["game_id", "off_team", "year", "weight"])
                    .size().reset_index(name="poss"))
    # Weighted average poss/game per team
    off_counts["wposs"] = off_counts["poss"] * off_counts["weight"]
    team_pace = (off_counts.groupby("off_team")
                           .agg(wposs=("wposs", "sum"), weight=("weight", "sum"))
                           .assign(pace=lambda x: x["wposs"] / x["weight"])
                           .reset_index()
                           .rename(columns={"off_team": "team_id"}))
    return team_pace[["team_id", "pace"]]


def _player_game_pace(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (player, game) combination, return the pace of that game.
    Pace of a game = total offensive possessions for the team in that game.
    """
    game_team_pace = (df.groupby(["game_id", "off_team"])
                        .size().reset_index(name="game_poss"))

    off_cols = [f"off_p{i}" for i in range(1, 6)]
    def_cols = [f"def_p{i}" for i in range(1, 6)]

    # Build player -> (game_id, off_team) mapping
    rows = []
    for col in off_cols:
        tmp = df[["game_id", "off_team", "year", "weight", col]].rename(columns={col: "player_id"})
        rows.append(tmp)
    for col in def_cols:
        tmp = df[["game_id", "def_team", "year", "weight", col]].rename(
            columns={col: "player_id", "def_team": "off_team"})
        rows.append(tmp)

    player_games = pd.concat(rows, ignore_index=True).drop_duplicates(
        ["game_id", "off_team", "player_id"]
    )
    player_games["player_id"] = player_games["player_id"].astype(int)

    # Join game pace
    player_games = player_games.merge(game_team_pace, on=["game_id", "off_team"], how="left")

    # Weighted average game pace per (player, team)
    player_games["wposs"] = player_games["game_poss"] * player_games["weight"]
    agg = (player_games.groupby(["player_id", "off_team"])
                       .agg(wposs=("wposs", "sum"), w=("weight", "sum"))
                       .assign(player_pace_avg=lambda x: x["wposs"] / x["w"])
                       .reset_index()
                       .rename(columns={"off_team": "team_id"}))
    return agg[["player_id", "team_id", "player_pace_avg"]]


def build_pace_cache(years: list[int] = STINTS_YEARS) -> pd.DataFrame:
    """
    Build and save pace cache.
    Returns DataFrame with columns: player_id, team_id, team_pace, player_residual
    """
    print("Building pace cache...")
    df = _load_all_stints(years)

    team_pace = _poss_per_game(df)
    league_mean = team_pace["pace"].mean()
    team_pace["pace_regressed"] = (
        team_pace["pace"] * (1 - LEAGUE_MEAN_REGRESSION)
        + league_mean * LEAGUE_MEAN_REGRESSION
    )
    print(f"  League mean pace: {league_mean:.1f} poss/game")

    player_pace = _player_game_pace(df)
    merged = player_pace.merge(
        team_pace[["team_id", "pace_regressed"]].rename(columns={"pace_regressed": "team_pace"}),
        on="team_id", how="left"
    )
    merged["player_residual"] = merged["player_pace_avg"] - merged["team_pace"]

    PACE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(PACE_CACHE, index=False)
    print(f"  Saved {len(merged)} player-team records -> {PACE_CACHE}")
    return merged


def load_pace_cache() -> pd.DataFrame:
    if not PACE_CACHE.exists():
        return build_pace_cache()
    return pd.read_csv(PACE_CACHE)


def team_median_pace(team_id: int, cache: Optional[pd.DataFrame] = None) -> float:
    if cache is None:
        cache = load_pace_cache()
    rows = cache[cache["team_id"] == team_id]
    if rows.empty:
        return load_league_mean(cache)
    return float(rows["team_pace"].iloc[0])


def load_league_mean(cache: Optional[pd.DataFrame] = None) -> float:
    if cache is None:
        cache = load_pace_cache()
    return float(cache["team_pace"].mean())


def predict_lineup_pace(
    lineup: dict[int, float],
    team_id: int,
    cache: Optional[pd.DataFrame] = None,
) -> float:
    """
    Predict pace for a specific lineup.
    lineup: {player_id: minutes}  (raw minutes, will normalize internally)
    team_id: integer team ID from stints
    """
    if cache is None:
        cache = load_pace_cache()

    base = team_median_pace(team_id, cache)

    # Player residuals indexed by (player_id, team_id) — prefer same-team, fall back to any
    pid_rows = cache[cache["player_id"].isin(lineup.keys())]
    same_team = pid_rows[pid_rows["team_id"] == team_id].set_index("player_id")["player_residual"]
    any_team = pid_rows.groupby("player_id")["player_residual"].mean()
    residuals = same_team.combine_first(any_team)

    total_mins = sum(lineup.values())
    if total_mins == 0:
        return base

    weighted_residual = sum(
        residuals.get(pid, 0.0) * mins / total_mins
        for pid, mins in lineup.items()
    )
    return round(base + weighted_residual, 1)


def predict_game_pace(
    home_lineup: dict[int, float],
    away_lineup: dict[int, float],
    home_team_id: int,
    away_team_id: int,
    cache: Optional[pd.DataFrame] = None,
) -> float:
    """Average of home and away lineup pace predictions."""
    if cache is None:
        cache = load_pace_cache()
    home_pace = predict_lineup_pace(home_lineup, home_team_id, cache)
    away_pace = predict_lineup_pace(away_lineup, away_team_id, cache)
    return round((home_pace + away_pace) / 2, 1)


def team_id_from_abbr(abbr: str) -> Optional[int]:
    """Look up team_id from team abbreviation via player_minutes."""
    from paths import PLAYER_MINUTES
    pm = pd.read_csv(PLAYER_MINUTES)
    row = pm[pm["team_abbr"] == abbr]
    if row.empty:
        return None
    return int(row["team_id"].mode().iloc[0])


if __name__ == "__main__":
    cache = build_pace_cache()
    print("\nSample team paces:")
    print(cache.groupby("team_id")["team_pace"].first().sort_values(ascending=False).head(10))
    print("\nTop pace-boosting players (residual):")
    top = cache.sort_values("player_residual", ascending=False).head(10)
    print(top[["player_id", "team_id", "team_pace", "player_residual"]])
