"""
Persistent roster override layer.

Sits between player_store (daily refreshed source of truth) and the app.
Overrides can:
  - Reassign a player to a different team (trades, free agency, expansion)
  - Override orapm / drapm (manual analyst adjustment)
  - Set projected minutes for a team's rotation

Schema: data/roster_overrides.csv
  player_id, player_name, team_abbr, orapm_override, drapm_override, notes

Schema: data/rotation_overrides.csv (projected minutes per team)
  team_abbr, player_id, player_name, projected_minutes

Usage:
  store = player_store.load()
  store = roster.apply(store)   # merged store ready for matchup engine
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from paths import DATA

ROSTER_OVERRIDES = DATA / "roster_overrides.csv"
ROTATION_OVERRIDES = DATA / "rotation_overrides.csv"

_ROSTER_COLS = ["player_id", "player_name", "team_abbr", "orapm_override", "drapm_override", "notes"]
_ROTATION_COLS = ["team_abbr", "player_id", "player_name", "projected_minutes"]


# ── Load / Save ──────────────────────────────────────────────────────────────

def load_roster_overrides() -> pd.DataFrame:
    if not ROSTER_OVERRIDES.exists():
        return pd.DataFrame(columns=_ROSTER_COLS)
    df = pd.read_csv(ROSTER_OVERRIDES, dtype={"player_id": "Int64"})
    for col in ["orapm_override", "drapm_override"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def save_roster_overrides(df: pd.DataFrame) -> None:
    ROSTER_OVERRIDES.parent.mkdir(parents=True, exist_ok=True)
    df[_ROSTER_COLS].to_csv(ROSTER_OVERRIDES, index=False)


def load_rotation_overrides() -> pd.DataFrame:
    if not ROTATION_OVERRIDES.exists():
        return pd.DataFrame(columns=_ROTATION_COLS)
    return pd.read_csv(ROTATION_OVERRIDES, dtype={"player_id": "Int64"})


def save_rotation_overrides(df: pd.DataFrame) -> None:
    ROTATION_OVERRIDES.parent.mkdir(parents=True, exist_ok=True)
    df[_ROTATION_COLS].to_csv(ROTATION_OVERRIDES, index=False)


# ── Apply overrides ──────────────────────────────────────────────────────────

def apply(store: pd.DataFrame) -> pd.DataFrame:
    """
    Merge roster overrides onto player_store.
    Returns a new DataFrame — does not mutate input.
    """
    store = store.copy()
    overrides = load_roster_overrides()
    if overrides.empty:
        return store

    for _, row in overrides.iterrows():
        pid = int(row["player_id"])
        mask = store["player_id"] == pid

        if not mask.any():
            # Player not in store — add them
            new_row = {col: np.nan for col in store.columns}
            new_row["player_id"] = pid
            new_row["player_name"] = row["player_name"]
            new_row["orapm"] = float(row["orapm_override"]) if pd.notna(row["orapm_override"]) else 0.0
            new_row["drapm"] = float(row["drapm_override"]) if pd.notna(row["drapm_override"]) else 0.0
            new_row["team_abbr"] = row["team_abbr"] if pd.notna(row.get("team_abbr")) else "UNK"
            new_row["minutes"] = 0.0
            store = pd.concat([store, pd.DataFrame([new_row])], ignore_index=True)
            continue

        if pd.notna(row.get("team_abbr")):
            store.loc[mask, "team_abbr"] = row["team_abbr"]
        if pd.notna(row.get("orapm_override")):
            store.loc[mask, "orapm"] = float(row["orapm_override"])
        if pd.notna(row.get("drapm_override")):
            store.loc[mask, "drapm"] = float(row["drapm_override"])

    return store


# ── Roster override CRUD ─────────────────────────────────────────────────────

def upsert_player(
    player_id: int,
    player_name: str,
    team_abbr: Optional[str] = None,
    orapm_override: Optional[float] = None,
    drapm_override: Optional[float] = None,
    notes: str = "",
) -> None:
    """Add or update a single player override."""
    df = load_roster_overrides()
    mask = df["player_id"] == player_id
    new_row = {
        "player_id": player_id,
        "player_name": player_name,
        "team_abbr": team_abbr,
        "orapm_override": orapm_override,
        "drapm_override": drapm_override,
        "notes": notes,
    }
    if mask.any():
        for k, v in new_row.items():
            if v is not None:
                df.loc[mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_roster_overrides(df)


def remove_player(player_id: int) -> None:
    df = load_roster_overrides()
    df = df[df["player_id"] != player_id]
    save_roster_overrides(df)


# ── Rotation overrides ───────────────────────────────────────────────────────

def set_rotation(team_abbr: str, rotation: pd.DataFrame) -> None:
    """
    Save a projected rotation for a team.
    rotation: DataFrame with columns player_id, player_name, projected_minutes
    """
    df = load_rotation_overrides()
    df = df[df["team_abbr"] != team_abbr]   # remove existing for this team
    rotation = rotation.copy()
    rotation["team_abbr"] = team_abbr
    df = pd.concat([df, rotation[_ROTATION_COLS]], ignore_index=True)
    save_rotation_overrides(df)


def get_rotation(team_abbr: str) -> Optional[pd.DataFrame]:
    """Return saved rotation for a team, or None if not set."""
    df = load_rotation_overrides()
    team = df[df["team_abbr"] == team_abbr]
    return team if not team.empty else None


def clear_rotation(team_abbr: str) -> None:
    df = load_rotation_overrides()
    df = df[df["team_abbr"] != team_abbr]
    save_rotation_overrides(df)


# ── CSV upload parser ────────────────────────────────────────────────────────

def parse_rotation_csv(uploaded_bytes: bytes, team_abbr: str) -> pd.DataFrame:
    """
    Parse an uploaded rotation CSV.
    Expected columns (flexible): player_name or name, minutes or projected_minutes, player_id (optional)
    Returns normalized DataFrame ready for set_rotation().
    """
    import io
    df = pd.read_csv(io.BytesIO(uploaded_bytes))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Normalize column names
    if "name" in df.columns and "player_name" not in df.columns:
        df = df.rename(columns={"name": "player_name"})
    if "minutes" in df.columns and "projected_minutes" not in df.columns:
        df = df.rename(columns={"minutes": "projected_minutes"})
    if "mins" in df.columns and "projected_minutes" not in df.columns:
        df = df.rename(columns={"mins": "projected_minutes"})

    if "player_name" not in df.columns:
        raise ValueError("CSV must have a 'player_name' or 'name' column")
    if "projected_minutes" not in df.columns:
        raise ValueError("CSV must have a 'minutes' or 'projected_minutes' column")

    if "player_id" not in df.columns:
        df["player_id"] = np.nan

    df["team_abbr"] = team_abbr
    df["projected_minutes"] = pd.to_numeric(df["projected_minutes"], errors="coerce").fillna(0.0)
    return df[_ROTATION_COLS]
