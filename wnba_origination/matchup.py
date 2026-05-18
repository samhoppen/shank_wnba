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
from pathlib import Path
from typing import Dict, Optional
import player_store
import pace as pace_module

# ── Default game constants (calibrated to 2026 league averages, per-40) ────
PACE_DEFAULT = 80       # 2026 league pace per 40 min (poss normalized for OT)
HCA = 2.2               # home-court advantage, points
LEAGUE_ORTG = 107.3     # 2026 league ORTG (pts per 100 poss) — read from pace_stats if available
BASE_PTS = round(LEAGUE_ORTG * PACE_DEFAULT / 100, 1)  # = league pts/team-game
SIGMA_MARGIN = 12.5     # std dev of margin around projected spread (WNBA empirical ~13)

# Cache the pace_stats DataFrame at module load
_PACE_STATS_PATH = Path(__file__).parent / "data" / "pace_stats.csv"
_PACE_STATS_CACHE: Optional[pd.DataFrame] = None


def _load_pace_stats() -> pd.DataFrame:
    global _PACE_STATS_CACHE
    if _PACE_STATS_CACHE is None:
        if _PACE_STATS_PATH.exists():
            _PACE_STATS_CACHE = pd.read_csv(_PACE_STATS_PATH)
        else:
            _PACE_STATS_CACHE = pd.DataFrame()
    return _PACE_STATS_CACHE


def _league_ortg() -> float:
    """Live 2026 league ORTG from pace_stats.csv (falls back to LEAGUE_ORTG)."""
    df = _load_pace_stats()
    if df.empty:
        return LEAGUE_ORTG
    sub = df[df["season"] == "2026_first8"]
    if sub.empty or "ortg" not in sub.columns:
        return LEAGUE_ORTG
    return float(sub["ortg"].mean())


def _team_pace_regressed(team_id: int, season: str = "2026", k: float = 5.0) -> float:
    """Bayesian-shrunk team pace from counted PBP possessions."""
    df = _load_pace_stats()
    if df.empty:
        return PACE_DEFAULT
    season_key = "2026_first8" if str(season) == "2026" else "2025_full"
    sub = df[df["season"] == season_key]
    if sub.empty:
        return PACE_DEFAULT
    league_mean = float(sub["pace"].mean())
    team_rows = sub[sub["team_id"] == team_id]
    if team_rows.empty:
        return league_mean
    n = len(team_rows)
    observed = float(team_rows["pace"].mean())
    return (n * observed + k * league_mean) / (n + k)


def predict_game_pace_v2(home_team_id: int, away_team_id: int, k: float = 5.0) -> float:
    """Predicted game pace = avg of home + away regressed team paces."""
    h = _team_pace_regressed(home_team_id, "2026", k)
    a = _team_pace_regressed(away_team_id, "2026", k)
    return round((h + a) / 2, 1)

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
    1611661327: "POR",   # Portland Fire (NBA Stats tricode = PDX)
    1611661328: "SEA",
    1611661329: "CHI",
    1611661330: "ATL",
    1611661331: "GSV",
    1611661332: "TOR",   # Toronto Tempo
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

    # Dynamic pace prediction — v2 uses pace_stats.csv (counted possessions,
    # includes 2026). Falls back to legacy pace_module on failure.
    if pace is None:
        if home_team_id and away_team_id:
            try:
                pace = predict_game_pace_v2(home_team_id, away_team_id)
            except Exception:
                try:
                    if pace_cache is None:
                        pace_cache = pace_module.load_pace_cache()
                    pace = pace_module.predict_game_pace(
                        home_lineup, away_lineup,
                        home_team_id, away_team_id,
                        cache=pace_cache,
                    )
                except Exception:
                    pace = PACE_DEFAULT
        else:
            pace = PACE_DEFAULT

    h_orapm, h_drapm = _weighted_rapm(home_lineup, store)
    a_orapm, a_drapm = _weighted_rapm(away_lineup, store)

    # ── Pace × ORTG decomposition ────────────────────────────────────────────
    # RAPM convention: oRAPM positive = better offense (pts added per 100 poss);
    # dRAPM positive = better defense (pts prevented per 100 poss).
    league_ortg = _league_ortg()
    home_off_rtg = league_ortg + h_orapm           # lineup's expected ORTG
    home_def_rtg = league_ortg - h_drapm           # lineup's expected DRTG (opp pts/100)
    away_off_rtg = league_ortg + a_orapm
    away_def_rtg = league_ortg - a_drapm
    # Opponent-adjusted ORTG: each side's offense vs the other side's defense,
    # relative to league. (Algebra collapses to LEAGUE_ORTG + h_oRAPM - a_dRAPM.)
    home_adj_ortg = home_off_rtg + (away_def_rtg - league_ortg)
    away_adj_ortg = away_off_rtg + (home_def_rtg - league_ortg)

    home_pts_pre = home_adj_ortg * pace / 100
    away_pts_pre = away_adj_ortg * pace / 100
    # Distribute HCA evenly across the two teams' displayed totals so
    # home_pts + hca/2 and away_pts - hca/2 sum to the same total and the
    # spread (home - away) equals (pre + hca). Otherwise the displayed
    # winning team and the spread disagreed when HCA flipped the result.
    home_pts = home_pts_pre + hca / 2
    away_pts = away_pts_pre - hca / 2

    spread = home_pts - away_pts          # positive = home favored
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
        "home_off_rtg": round(home_off_rtg, 1),
        "home_def_rtg": round(home_def_rtg, 1),
        "away_off_rtg": round(away_off_rtg, 1),
        "away_def_rtg": round(away_def_rtg, 1),
        "home_adj_ortg": round(home_adj_ortg, 1),
        "away_adj_ortg": round(away_adj_ortg, 1),
        "league_ortg": round(league_ortg, 1),
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
