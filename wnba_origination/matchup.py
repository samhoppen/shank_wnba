"""
Core line origination engine.

Two entry points:
  project_matchup(home_lineup, away_lineup, ...)  -> dict
  project_by_team(home_abbr, away_abbr, ...)      -> dict

Lineup format:  {player_id: minutes_share}
  minutes_share = fraction of team minutes (should sum to ~1.0 per team)
  OR raw projected minutes (auto-normalized to sum=1)
"""
import numpy as np
from scipy.stats import norm
import pandas as pd
from typing import Dict, Optional
import player_store
import pace as pace_module

# ── Default game constants ──────────────────────────────────────────────────
PACE_DEFAULT = 78       # fallback if pace cache unavailable
HCA = 2.2               # home-court advantage, points
BASE_PTS = 79           # league-average team scoring
SIGMA_MARGIN = 10.5     # std dev of game margin for win prob

TEAM_ID_MAP = {
    1611661313: "NYL",
    1611661317: "PHX",
    1611661319: "LVA",
    1611661320: "LAS",
    1611661321: "DAL",
    1611661322: "WAS",
    1611661323: "CON",
    1611661324: "MIN",
    1611661325: "IND",
    1611661328: "SEA",
    1611661329: "CHI",
    1611661330: "ATL",
    1611661331: "GSV",
}
ABBR_TO_TEAM_ID = {v: k for k, v in TEAM_ID_MAP.items()}


def _weighted_rapm(lineup: Dict[int, float], store: pd.DataFrame) -> tuple[float, float]:
    """Return (orapm, drapm) weighted by minutes shares in lineup."""
    idx = store.set_index("player_id")
    orapm_total = 0.0
    drapm_total = 0.0
    weight_total = 0.0

    for pid, mins in lineup.items():
        if pid in idx.index:
            row = idx.loc[pid]
            orapm_total += row["orapm"] * mins
            drapm_total += row["drapm"] * mins
        weight_total += mins

    if weight_total == 0:
        return 0.0, 0.0

    return orapm_total / weight_total, drapm_total / weight_total


def project_matchup(
    home_lineup: Dict[int, float],
    away_lineup: Dict[int, float],
    home_team_id: Optional[int] = None,
    away_team_id: Optional[int] = None,
    pace: Optional[float] = None,       # None = auto from pace model
    hca: float = HCA,
    base_pts: float = BASE_PTS,
    store: Optional[pd.DataFrame] = None,
    pace_cache: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Project spread + total from player lineups.

    home_lineup / away_lineup: {player_id: raw_minutes}  (normalized internally)
    pace: override game pace; if None, predicted dynamically from lineups + pace model
    Returns dict with: spread, total, home_pts, away_pts, home_orapm, home_drapm,
                       away_orapm, away_drapm, win_prob_home, pace
    """
    if store is None:
        store = player_store.load()

    # Dynamic pace prediction
    if pace is None:
        try:
            if pace_cache is None:
                pace_cache = pace_module.load_pace_cache()
            if home_team_id and away_team_id:
                pace = pace_module.predict_game_pace(
                    home_lineup, away_lineup,
                    home_team_id, away_team_id,
                    cache=pace_cache,
                )
            else:
                pace = PACE_DEFAULT
        except Exception:
            pace = PACE_DEFAULT

    h_orapm, h_drapm = _weighted_rapm(home_lineup, store)
    a_orapm, a_drapm = _weighted_rapm(away_lineup, store)

    home_pts = base_pts + (h_orapm - a_drapm) * pace / 100
    away_pts = base_pts + (a_orapm - h_drapm) * pace / 100

    spread = home_pts - away_pts + hca          # positive = home favored
    total = home_pts + away_pts

    win_prob_home = norm.cdf(spread / SIGMA_MARGIN)

    return {
        "spread": round(spread, 1),
        "total": round(total, 1),
        "home_pts": round(home_pts, 1),
        "away_pts": round(away_pts, 1),
        "home_orapm": round(h_orapm, 2),
        "home_drapm": round(h_drapm, 2),
        "away_orapm": round(a_orapm, 2),
        "away_drapm": round(a_drapm, 2),
        "win_prob_home": round(win_prob_home, 3),
        "pace": pace,
    }


def team_default_lineup(team_abbr: str, store: pd.DataFrame) -> Dict[int, float]:
    """Return default lineup for a team: all rostered players weighted by historical minutes."""
    players = store[store["team_abbr"] == team_abbr].copy()
    if players.empty:
        return {}
    players = players[players["minutes"] > 0]
    total_mins = players["minutes"].sum()
    if total_mins == 0:
        return {}
    return dict(zip(players["player_id"], players["minutes"] / total_mins))


def project_by_team(
    home_abbr: str,
    away_abbr: str,
    home_overrides: Optional[Dict[int, float]] = None,
    away_overrides: Optional[Dict[int, float]] = None,
    pace: Optional[float] = None,
    hca: float = HCA,
    base_pts: float = BASE_PTS,
    store: Optional[pd.DataFrame] = None,
    pace_cache: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Convenience wrapper: look up team rosters from player_store, apply optional minute overrides.
    pace=None triggers dynamic pace prediction from lineup composition.
    """
    if store is None:
        store = player_store.load()

    home_lineup = team_default_lineup(home_abbr, store)
    away_lineup = team_default_lineup(away_abbr, store)

    if home_overrides:
        home_lineup.update(home_overrides)
    if away_overrides:
        away_lineup.update(away_overrides)

    home_team_id = ABBR_TO_TEAM_ID.get(home_abbr)
    away_team_id = ABBR_TO_TEAM_ID.get(away_abbr)

    result = project_matchup(
        home_lineup, away_lineup,
        home_team_id=home_team_id, away_team_id=away_team_id,
        pace=pace, hca=hca, base_pts=base_pts,
        store=store, pace_cache=pace_cache,
    )
    result["home_team"] = home_abbr
    result["away_team"] = away_abbr
    return result


def ml_from_prob(p: float) -> tuple[int, int]:
    """
    Convert home win probability to American moneyline (home_ml, away_ml).
    Fair ML: both sides derived from the same odds ratio, opposite signs.
    """
    if p <= 0 or p >= 1:
        return 0, 0
    if p >= 0.5:
        ratio = p / (1 - p)
        home_ml = int(-round(ratio * 100))
        away_ml = int(+round(ratio * 100))
    else:
        ratio = (1 - p) / p
        home_ml = int(+round(ratio * 100))
        away_ml = int(-round(ratio * 100))
    return home_ml, away_ml


if __name__ == "__main__":
    result = project_by_team("LVA", "NYL")
    p = result["win_prob_home"]
    fav_ml, dog_ml = ml_from_prob(p)
    print(f"LVA vs NYL")
    print(f"  Spread: {result['spread']:+.1f} (LVA)")
    print(f"  Total:  {result['total']:.1f}")
    print(f"  Score:  LVA {result['home_pts']:.1f} – NYL {result['away_pts']:.1f}")
    print(f"  LVA win prob: {p:.1%}  ML: {fav_ml}/{dog_ml}")
