"""
Build unified player table.

Priority for RAPM:
  1. rapm_2026_RS.csv (current season, rows with poss >= 300)
  2. rapm_8factor_2025_RS_5yr.csv  net_rapm_reconstructed (fallback)
  3. 0.0 (league average — rookies / expansion players)

Run directly to rebuild:
  python player_store.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from paths import (
    RAPM_DIR, PLAYER_MINUTES, PLAYER_NAMES,
    PLAYER_STORE, EC_ALL_SEASONS, EC_2026, RAPM_2026,
    rapm_season, rapm_8factor,
)

MIN_POSS_2026 = 300   # minimum 2026 possessions to use current-season RAPM


def _load_player_names() -> pd.DataFrame:
    if PLAYER_NAMES.exists():
        return pd.read_csv(PLAYER_NAMES)[["player_id", "player_name"]]
    # Fall back to extracting names from player_minutes
    pm = pd.read_csv(PLAYER_MINUTES)
    return pm[["player_id", "player_name"]].drop_duplicates("player_id")


def _load_rapm_2025() -> pd.DataFrame:
    """Load 2025 RAPM — prefer 1yr file, fall back to 3yr for missing players."""
    path_1yr = rapm_season(2025)
    path_3yr = RAPM_DIR / "rapm_2025_RS_3yr.csv"

    df1 = pd.read_csv(path_1yr)[["player_id", "orapm", "drapm", "net_rapm"]] if path_1yr.exists() else pd.DataFrame(columns=["player_id", "orapm", "drapm", "net_rapm"])

    if path_3yr.exists():
        df3 = pd.read_csv(path_3yr)[["player_id", "orapm", "drapm", "net_rapm"]]
        # Add players in 3yr that are missing from 1yr
        missing = df3[~df3["player_id"].isin(df1["player_id"])]
        df1 = pd.concat([df1, missing], ignore_index=True)

    return df1.rename(columns={"net_rapm": "net_rapm_2025"})


def _load_rapm_8factor() -> pd.DataFrame:
    path = rapm_8factor(2025)
    df = pd.read_csv(path)
    return df[["player_id", "player_name", "poss", "net_rapm_reconstructed"]]


def _load_rapm_2026() -> pd.DataFrame:
    """Returns current-season RAPM for players with >= MIN_POSS_2026 possessions."""
    if not RAPM_2026.exists():
        return pd.DataFrame(columns=["player_id", "orapm_2026", "drapm_2026", "poss_2026"])
    df = pd.read_csv(RAPM_2026)
    df = df[df["poss"] >= MIN_POSS_2026].copy()
    return df[["player_id", "orapm", "drapm", "poss"]].rename(columns={
        "orapm": "orapm_2026", "drapm": "drapm_2026", "poss": "poss_2026"
    })


def _load_ec_2026() -> pd.DataFrame:
    """Load 2026 EC from daily scrape if available, else fall back to 2025 from EC_ALL_SEASONS."""
    if EC_2026.exists():
        df = pd.read_csv(EC_2026)
        df["player_name_ec"] = df["Player"]
        return df[["Player", "oec", "dec", "ec"]].rename(columns={"Player": "player_name_ec"})

    # Fall back to 2025 from historical file
    if EC_ALL_SEASONS.exists():
        df = pd.read_csv(EC_ALL_SEASONS)
        df2025 = df[df["Season"] == 2025].copy()
        df2025["player_name_ec"] = df2025["Player"]
        return df2025[["player_name_ec", "oec", "dec", "ec"]]

    return pd.DataFrame(columns=["player_name_ec", "oec", "dec", "ec"])


def _derive_2026_minutes() -> pd.DataFrame:
    """Derive per-player 2026 minutes from analysis_player_stats.csv.

    The raw player_minutes.csv has scrambled values for 2026 (e.g. 58 min/game).
    We re-derive from the clean possession data:

        minutes_per_game ≈ (on_court_off_poss / team_off_poss) * 40
        total_minutes    = Σ minutes_per_game across games

    Returns DataFrame: player_id | player_name | team_abbr | minutes | minutes_per_game
    """
    from pathlib import Path
    p = Path(__file__).parent / "data" / "analysis_player_stats.csv"
    if not p.exists():
        return pd.DataFrame(columns=["player_id", "player_name", "team_abbr",
                                      "minutes", "minutes_per_game"])
    ps = pd.read_csv(p)
    ps = ps[ps["season"] == "2026_first8"].copy()
    if ps.empty:
        return pd.DataFrame(columns=["player_id", "player_name", "team_abbr",
                                      "minutes", "minutes_per_game"])

    # poss column = team's total offensive possessions for that game (denominator)
    # on_court_poss = how many of those the player was on the floor for
    ps["min_in_game"] = (ps["on_court_poss"] / ps["poss"].replace(0, np.nan)) * 40.0

    # Only count games where the player actually has on-floor data
    played = ps.dropna(subset=["on_court_poss", "poss", "min_in_game"]).copy()
    played = played[played["on_court_poss"] > 0]

    out = (played.groupby(["person_id", "player_name", "team"])
                  .agg(games=("game_id", "nunique"),
                       minutes=("min_in_game", "sum"))
                  .reset_index()
                  .rename(columns={"person_id": "player_id", "team": "team_abbr"}))
    out["minutes_per_game"] = out["minutes"] / out["games"].replace(0, np.nan)
    out = out[["player_id", "player_name", "team_abbr", "minutes", "minutes_per_game"]]
    return out


def _load_team_map_2026() -> pd.DataFrame:
    """Return player_id -> team_abbr mapping + minutes.

    Strategy:
      1. Get 2026 minutes from analysis_player_stats (clean per-poss derivation)
      2. For players without 2026 analysis data, fall back to 2025 player_minutes
         (the legacy file is in seconds-scale, divide by 60)
      3. Skip the legacy player_minutes.csv 2026 rows entirely — they have a
         different 2x scaling bug we can't reliably correct.
    """
    pm = pd.read_csv(PLAYER_MINUTES)
    pm = pm[pm["season"] < 2026].copy()  # skip buggy 2026 legacy rows
    # Legacy 2017–2025 rows are seconds-scale: divide by 60 if min/g > 40 (impossible).
    bad = pm["minutes_per_game"] > 40
    pm.loc[bad, "minutes_per_game"] /= 60.0
    pm.loc[bad, "minutes"] /= 60.0
    # Defensive cap — if still >40 after /60, the row is corrupt; null it out.
    still_bad = pm["minutes_per_game"] > 40
    pm.loc[still_bad, ["minutes", "minutes_per_game"]] = float("nan")
    legacy_fallback = (
        pm.sort_values(["season", "minutes"], ascending=[False, False])
          .drop_duplicates("player_id")
          [["player_id", "team_abbr", "minutes", "minutes_per_game"]]
    )

    # 2026 derived minutes (preferred — clean possession-based derivation)
    m26 = _derive_2026_minutes()
    if m26.empty:
        return legacy_fallback

    m26_pids = set(m26["player_id"])
    fb_only = legacy_fallback[~legacy_fallback["player_id"].isin(m26_pids)]
    return pd.concat(
        [m26[["player_id", "team_abbr", "minutes", "minutes_per_game"]], fb_only],
        ignore_index=True,
    )


def _fuzzy_join_ec(store: pd.DataFrame, ec: pd.DataFrame) -> pd.DataFrame:
    """Join EC by exact name match first, then fall back to None (no EC data)."""
    if ec.empty:
        store["oec"] = np.nan
        store["dec"] = np.nan
        store["ec"] = np.nan
        return store

    ec_indexed = ec.set_index("player_name_ec")
    store["oec"] = store["player_name"].map(ec_indexed["oec"])
    store["dec"] = store["player_name"].map(ec_indexed["dec"])
    store["ec"] = store["player_name"].map(ec_indexed["ec"])
    return store


def build(save: bool = True) -> pd.DataFrame:
    names = _load_player_names()
    rapm_8f = _load_rapm_8factor()
    rapm_25 = _load_rapm_2025()
    rapm_26 = _load_rapm_2026()
    ec = _load_ec_2026()
    teams = _load_team_map_2026()

    # Base: 8-factor RAPM has most players with multi-year history
    base = rapm_8f.merge(names, on="player_id", how="outer", suffixes=("_8f", "_nm"))
    base["player_name"] = base["player_name_8f"].fillna(base["player_name_nm"])
    base = base.drop(columns=["player_name_8f", "player_name_nm"], errors="ignore")

    # Bring in 2025 per-season orapm/drapm
    base = base.merge(rapm_25, on="player_id", how="left")

    # Overlay 2026 RAPM for players with enough current-season data
    base = base.merge(rapm_26, on="player_id", how="left")

    # Final orapm/drapm: prefer 2026 if available, else 2025
    has_2026 = base["poss_2026"].notna() & (base["poss_2026"] >= MIN_POSS_2026)
    base["orapm"] = np.where(has_2026, base["orapm_2026"], base["orapm"])
    base["drapm"] = np.where(has_2026, base["drapm_2026"], base["drapm"])
    # Fill remaining with league average
    base["orapm"] = base["orapm"].fillna(0.0)
    base["drapm"] = base["drapm"].fillna(0.0)

    # net_rapm_reconstructed from 8-factor (fill 0 for no history)
    base["net_rapm_reconstructed"] = base["net_rapm_reconstructed"].fillna(0.0)

    # Team + minutes (most recent season)
    base = base.merge(teams, on="player_id", how="left")
    base["minutes"] = base["minutes"].fillna(0.0)
    base["minutes_per_game"] = base["minutes_per_game"].fillna(0.0)
    base["team_abbr"] = base["team_abbr"].fillna("UNK")

    # EC
    base = _fuzzy_join_ec(base, ec)

    # Final column selection
    out = base[[
        "player_id", "player_name", "team_abbr",
        "minutes", "minutes_per_game",
        "orapm", "drapm", "net_rapm_reconstructed",
        "oec", "dec", "ec",
    ]].copy()
    out = out.sort_values("minutes", ascending=False).reset_index(drop=True)

    if save:
        PLAYER_STORE.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(PLAYER_STORE, index=False)
        print(f"Saved {len(out)} players -> {PLAYER_STORE}")

    return out


def load() -> pd.DataFrame:
    """Load cached player store (build if missing)."""
    if not PLAYER_STORE.exists():
        return build(save=True)
    return pd.read_csv(PLAYER_STORE)


if __name__ == "__main__":
    df = build()
    print(df[["player_name", "team_abbr", "orapm", "drapm", "ec"]].head(20).to_string())
