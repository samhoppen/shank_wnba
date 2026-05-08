"""
5-man unit net ratings from stints data.

Primary uses:
  1. Seed the rotation chart in app.py (recent game lineups + minutes)
  2. Compute lineup-level net rating for a specific 5v5 matchup

Known caveat: stints track ~83% of possessions (lineup tracking gap in RAPM pipeline).
Player-level projections in matchup.py are more reliable than lineup-level outputs here.
"""
import pandas as pd
import numpy as np
from typing import Optional
from paths import stints, PLAYER_MINUTES, PLAYER_NAMES, RAPM_DIR


def _player_name_map() -> dict:
    pn = PLAYER_NAMES if PLAYER_NAMES.exists() else None
    if pn and pn.exists():
        df = pd.read_csv(pn)
        return dict(zip(df["player_id"], df["player_name"]))
    pm = pd.read_csv(PLAYER_MINUTES)
    return dict(zip(pm["player_id"], pm["player_name"]))


def load_stints(year: int = 2025) -> pd.DataFrame:
    path = stints(year)
    if not path.exists():
        raise FileNotFoundError(f"Stints not found: {path}")
    return pd.read_csv(path)


def team_id_for_abbr(abbr: str, year: int = 2025) -> Optional[int]:
    pm = pd.read_csv(PLAYER_MINUTES)
    row = pm[(pm["team_abbr"] == abbr) & (pm["season"] == year)]
    if row.empty:
        return None
    return int(row["team_id"].iloc[0])


def recent_game_rotations(
    team_abbr: str,
    year: int = 2025,
    n_games: int = 10,
) -> pd.DataFrame:
    """
    Return per-player minutes totals over last N games for a team.

    Output columns: player_id, player_name, games, total_poss, poss_per_game, min_share
    """
    df = load_stints(year)
    team_id = team_id_for_abbr(team_abbr, year)
    if team_id is None:
        raise ValueError(f"Team '{team_abbr}' not found for {year}")

    off_cols = [f"off_p{i}" for i in range(1, 6)]
    def_cols = [f"def_p{i}" for i in range(1, 6)]

    # Stints where this team is on offense OR defense
    team_stints = df[(df["off_team"] == team_id) | (df["def_team"] == team_id)].copy()

    # Last N games
    games = team_stints["game_id"].unique()
    if len(games) > n_games:
        games = games[-n_games:]
    team_stints = team_stints[team_stints["game_id"].isin(games)]

    # Count possessions played per player on this team
    poss_map: dict[int, int] = {}
    for _, row in team_stints.iterrows():
        if row["off_team"] == team_id:
            player_cols = off_cols
        else:
            player_cols = def_cols
        for col in player_cols:
            pid = int(row[col])
            poss_map[pid] = poss_map.get(pid, 0) + 1

    name_map = _player_name_map()
    rows = []
    total_poss = sum(poss_map.values())
    for pid, poss in sorted(poss_map.items(), key=lambda x: -x[1]):
        rows.append({
            "player_id": pid,
            "player_name": name_map.get(pid, str(pid)),
            "games": len(games),
            "total_poss": poss,
            "poss_per_game": round(poss / len(games), 1),
            "min_share": round(poss / (total_poss / 5), 4),  # per-player fraction of team poss
        })

    return pd.DataFrame(rows)


def lineup_net_rating(
    lineup: list[int],
    stints_df: pd.DataFrame,
    min_poss: int = 20,
) -> Optional[float]:
    """
    Compute net rating for an exact 5-man lineup from stints.
    Returns None if fewer than min_poss possessions found.
    """
    off_cols = [f"off_p{i}" for i in range(1, 6)]
    def_cols = [f"def_p{i}" for i in range(1, 6)]
    lineup_set = frozenset(lineup)

    # Find possessions where all 5 are on offense
    mask_off = stints_df[off_cols].apply(
        lambda row: frozenset(row.values) == lineup_set, axis=1
    )
    # Find possessions where all 5 are on defense
    mask_def = stints_df[def_cols].apply(
        lambda row: frozenset(row.values) == lineup_set, axis=1
    )

    off_stints = stints_df[mask_off]
    def_stints = stints_df[mask_def]

    if len(off_stints) + len(def_stints) < min_poss:
        return None

    n_off = len(off_stints)
    n_def = len(def_stints)
    pts_scored = off_stints["points"].sum()
    pts_allowed = def_stints["points"].sum()

    ortg = (pts_scored / n_off * 100) if n_off > 0 else None
    drtg = (pts_allowed / n_def * 100) if n_def > 0 else None

    if ortg is not None and drtg is not None:
        return round(ortg - drtg, 1)
    return None


def rotation_for_app(team_abbr: str, year: int = 2025, n_games: int = 10) -> pd.DataFrame:
    """
    Rotation data for Streamlit UI.
    Returns player-level minutes share + auto-computed default rotation.
    """
    rotations = recent_game_rotations(team_abbr, year=year, n_games=n_games)
    # Scale min_share so top 8 players sum to reasonable rotation
    rotations = rotations.sort_values("total_poss", ascending=False).head(12)
    return rotations.reset_index(drop=True)


if __name__ == "__main__":
    import sys
    team = sys.argv[1] if len(sys.argv) > 1 else "LVA"
    df = rotation_for_app(team)
    print(f"\n{team} rotation (last 10 games):")
    print(df[["player_name", "poss_per_game", "min_share"]].to_string())
