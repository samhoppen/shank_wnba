"""
Season win total projections for all 14 WNBA teams.

Math:
  - Build each team's net rating from player_store (minutes-weighted RAPM)
  - Pythagorean win% (exponent = 10.80, calibrated for WNBA)
  - Scale team wins to sum = 308 (zero-sum constraint, 14 teams × 44 games / 2 × 2 = 616 total)
  - Threshold probabilities via normal CDF (σ ≈ 8.5 wins, continuity-corrected)

Historical base rates (sanity check):
  ≥36W: 2.6%   ≥34W: 4.9%   ≥32W: 9.6%   ≥30W: 12.8%
  <15W: 17.4%  <10W: 5.8%
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Optional
import player_store
import matchup as mx

PYTH_EXP = 10.80
GAMES = 44
TOTAL_WINS = GAMES * 14 / 2   # 308  (zero-sum across all teams)
SIGMA_WINS = 8.5
BASE_PTS = mx.BASE_PTS
PACE = mx.PACE_DEFAULT

TEAMS_2026 = [
    "NYL", "PHX", "LVA", "LAS", "DAL", "WAS",
    "CON", "MIN", "IND", "SEA", "CHI", "ATL", "GSV",
    "POR", "TOR",   # expansion teams
]


def team_net_rating(team_abbr: str, store: pd.DataFrame) -> tuple[float, float, float]:
    """Return (orapm, drapm, net_rapm) for a team weighted by player minutes."""
    lineup = mx.team_default_lineup(team_abbr, store)
    if not lineup:
        return 0.0, 0.0, 0.0
    orapm, drapm = mx._weighted_rapm(lineup, store)
    return round(orapm, 2), round(drapm, 2), round(orapm + drapm, 2)


def _pythagorean_win_pct(net_rtg: float, base: float = BASE_PTS) -> float:
    """Convert net rating to win% using Pythagorean formula."""
    ortg = base + net_rtg / 2
    drtg = base - net_rtg / 2
    if drtg <= 0:
        return 1.0
    return ortg ** PYTH_EXP / (ortg ** PYTH_EXP + drtg ** PYTH_EXP)


def project_all_teams(
    store: Optional[pd.DataFrame] = None,
    sigma: float = SIGMA_WINS,
) -> pd.DataFrame:
    """
    Project win totals for all teams. Returns DataFrame with one row per team.
    """
    if store is None:
        store = player_store.load()

    rows = []
    for team in TEAMS_2026:
        orapm, drapm, net = team_net_rating(team, store)
        if net == 0.0 and orapm == 0.0:
            # Expansion/missing team — treat as league average
            win_pct = 0.5
        else:
            win_pct = _pythagorean_win_pct(net)
        rows.append({
            "team": team,
            "orapm": orapm,
            "drapm": drapm,
            "net_rapm": net,
            "raw_win_pct": win_pct,
            "raw_wins": round(win_pct * GAMES, 1),
        })

    df = pd.DataFrame(rows)

    # Zero-sum rescale: adjust wins so they sum to TOTAL_WINS
    raw_total = df["raw_wins"].sum()
    scale = TOTAL_WINS / raw_total if raw_total > 0 else 1.0
    df["proj_wins"] = (df["raw_wins"] * scale).round(1)
    df["proj_win_pct"] = (df["proj_wins"] / GAMES).round(3)

    # Threshold probabilities (normal approximation, continuity correction)
    for threshold, label in [(30, "p_ge_30"), (32, "p_ge_32"), (34, "p_ge_34"), (36, "p_ge_36")]:
        df[label] = df["proj_wins"].apply(
            lambda mu: norm.sf(threshold - 0.5, loc=mu, scale=sigma)
        ).round(3)
    for threshold, label in [(15, "p_lt_15"), (10, "p_lt_10")]:
        df[label] = df["proj_wins"].apply(
            lambda mu: norm.cdf(threshold + 0.5, loc=mu, scale=sigma)
        ).round(3)

    return df.sort_values("proj_wins", ascending=False).reset_index(drop=True)


def fair_ml_win_total(proj_wins: float, line: float, sigma: float = SIGMA_WINS) -> dict:
    """
    Price the over/under for a win total line.
    Returns dict with p_over, p_under, over_ml, under_ml.
    """
    p_over = norm.sf(line - 0.5, loc=proj_wins, scale=sigma)
    p_under = 1.0 - p_over

    def to_ml(p: float) -> int:
        if p <= 0 or p >= 1:
            return 0
        if p >= 0.5:
            return int(-round(p / (1 - p) * 100))
        return int(round((1 - p) / p * 100))

    return {
        "proj_wins": round(proj_wins, 1),
        "line": line,
        "p_over": round(p_over, 3),
        "p_under": round(p_under, 3),
        "over_ml": to_ml(p_over),
        "under_ml": to_ml(p_under),
    }


if __name__ == "__main__":
    df = project_all_teams()
    print(df[["team", "orapm", "drapm", "net_rapm", "proj_wins",
              "p_ge_30", "p_ge_32", "p_ge_34", "p_ge_36",
              "p_lt_15", "p_lt_10"]].to_string())
