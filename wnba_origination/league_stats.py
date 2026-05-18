"""
League- and team-level statistics derived from the analysis CSVs synced
in from WNBA_RAPM/analysis.

All functions read from wnba_origination/data/ and return pandas
DataFrames or dicts ready for display.

Public API:
    league_baselines(season)       -> dict of mean stats
    team_profile(team, season)     -> dict
    team_profiles(season)          -> DataFrame indexed by team
    recent_games(n=20)             -> DataFrame of recent game totals
    bonus_summary(season)          -> DataFrame of bonus-reach % by quarter
    foul_rates(season)             -> DataFrame of foul rates
    regress(team, stat, season, k) -> Bayesian-shrunk team value

Season keys:
    "2025_full"   = full 2025
    "2026_first8" = 2026 so far (kept this key for backward compat with
                    ft_decomp.csv; really "2026-to-date" now)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# Tricode normalization: NBA Stats / legacy → app canonical
TRICODE_ALIAS = {
    "PDX": "POR",   # Portland Fire (NBA Stats abbr)
    "PHO": "PHX",   # legacy Phoenix
}


def _normalize_team_cols(df: "pd.DataFrame") -> "pd.DataFrame":
    """Apply tricode aliases to any team/team_abbr/off_team-style columns."""
    if df.empty:
        return df
    for col in ("team", "team_abbr"):
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].replace(TRICODE_ALIAS)
    return df

SEASON_KEYS = {
    "2025": "2025_full",
    "2026": "2026_first8",
}
SEASON_LABELS = {
    "2025_full":   "2025 (Full)",
    "2026_first8": "2026",
}


# ─────────────────────────────────────────────────────────────────────────────
# Raw loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load(name: str, **kwargs) -> pd.DataFrame:
    p = DATA_DIR / name
    if not p.exists():
        return pd.DataFrame()
    return _normalize_team_cols(pd.read_csv(p, **kwargs))


def load_pace() -> pd.DataFrame:
    df = _load("pace_stats.csv", dtype={"game_id": str})
    if not df.empty:
        df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    return df


def load_ft_decomp() -> pd.DataFrame:
    df = _load("ft_decomp.csv", dtype={"game_id": str})
    if not df.empty:
        df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    return df


def load_bonus() -> pd.DataFrame:
    df = _load("bonus_by_quarter.csv", dtype={"game_id": str}, parse_dates=["game_date"])
    if not df.empty:
        df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    return df


def load_foul_rates() -> pd.DataFrame:
    return _load("foul_violation_rates.csv")


def load_games(year: int) -> pd.DataFrame:
    df = _load(f"games_{year}_RS.csv", dtype={"game_id": str})
    if not df.empty:
        df["game_id"] = df["game_id"].astype(str).str.zfill(10)
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


def load_stints_rich(year: int) -> pd.DataFrame:
    df = _load(f"stints_rich_{year}.csv", dtype={"game_id": str})
    if not df.empty:
        df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    return df


def load_hth(season: int = 2026) -> pd.DataFrame:
    return _load(f"hth_players_{season}.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Aggregations
# ─────────────────────────────────────────────────────────────────────────────

def _season_key(season) -> str:
    """Accept '2025' / 2025 / '2025_full' and return canonical key."""
    s = str(season)
    if s in SEASON_KEYS:
        return SEASON_KEYS[s]
    if s in SEASON_KEYS.values():
        return s
    return s  # let it fail downstream with a clear message


def last_n_baselines(n: int = 30) -> dict:
    """League-wide baselines for the most recent N 2026 games.

    Only looks at the current (2026) season — does not span seasons.
    """
    pace = load_pace()
    if pace.empty:
        return {}

    # 2026 games only
    pace = pace[pace["season"] == "2026_first8"].copy()
    if pace.empty:
        return {}

    games = load_games(2026)
    if games.empty:
        return {}
    date_map = dict(zip(games["game_id"], games["game_date"]))
    pace["game_date"] = pace["game_id"].map(date_map)
    pace = pace.dropna(subset=["game_date"]).sort_values("game_date", ascending=False)

    last_n_gids = pace["game_id"].drop_duplicates().head(n).tolist()
    sub = pace[pace["game_id"].isin(last_n_gids)]
    if sub.empty:
        return {}

    out = {
        "n_team_games": int(len(sub)),
        "n_games":      int(sub["game_id"].nunique()),
        "pace":         float(sub["pace"].mean()),
        "ortg":         float(sub["ortg"].mean()),
        "pts":          float(sub["PTS"].mean()),
        "fga":          float(sub["FGA"].mean()),
        "fta":          float(sub["FTA"].mean()),
        "tov":          float(sub["TOV"].mean()),
        "oreb":         float(sub["OREB"].mean()),
        "ft_per_fga":   float((sub["FTA"] / sub["FGA"].replace(0, float("nan"))).mean()),
    }
    # Transition rate restricted to these 2026 game_ids
    sr = load_stints_rich(2026)
    if not sr.empty:
        sr = sr[sr["game_id"].isin(last_n_gids)]
        bad = sr.loc[sr["points"] > 5, "game_id"].unique()
        sr = sr[~sr["game_id"].isin(bad)]
        if not sr.empty:
            out["trans_pct"] = float(sr["trans_flag"].mean() * 100)
    # Bonus reach
    bonus = load_bonus()
    if not bonus.empty:
        bsub = bonus[bonus["game_id"].isin(last_n_gids)]
        if not bsub.empty:
            out["bonus_q_pct"] = float(bsub["bonus_reached"].mean() * 100)
    return out


def league_baselines(season) -> dict:
    """League-wide per-team-game means + derived rates."""
    sk = _season_key(season)
    pace = load_pace()
    sub = pace[pace["season"] == sk]
    if sub.empty:
        return {}
    out = {
        "n_team_games": int(len(sub)),
        "n_games":      int(sub["game_id"].nunique()),
        "pace":         float(sub["pace"].mean()),
        "ortg":         float(sub["ortg"].mean()),
        "pts":          float(sub["PTS"].mean()),
        "fga":          float(sub["FGA"].mean()),
        "fta":          float(sub["FTA"].mean()),
        "tov":          float(sub["TOV"].mean()),
        "oreb":         float(sub["OREB"].mean()),
        "ft_per_fga":   float((sub["FTA"] / sub["FGA"].replace(0, float("nan"))).mean()),
    }
    # Transition rate from stints_rich (per-poss)
    year = 2025 if "2025" in sk else 2026
    sr = load_stints_rich(year)
    if not sr.empty:
        # Drop games with cumulative-points bug
        bad = sr.loc[sr["points"] > 5, "game_id"].unique()
        sr = sr[~sr["game_id"].isin(bad)]
        out["trans_pct"] = float(sr["trans_flag"].mean() * 100)
    # Bonus-quarter rate
    bonus = load_bonus()
    if not bonus.empty and "season" in bonus.columns:
        bsub = bonus[bonus["season"] == sk]
        if not bsub.empty:
            out["bonus_q_pct"] = float(bsub["bonus_reached"].mean() * 100)
    return out


def team_profiles(season) -> pd.DataFrame:
    """Per-team season aggregates."""
    sk = _season_key(season)
    pace = load_pace()
    sub = pace[pace["season"] == sk]
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby("team").agg(
        n=("PTS", "count"),
        pace=("pace", "mean"),
        ortg=("ortg", "mean"),
        pts=("PTS", "mean"),
        fga=("FGA", "mean"),
        fta=("FTA", "mean"),
        tov=("TOV", "mean"),
        oreb=("OREB", "mean"),
    ).round(2)
    g["fta_per_fga"] = (g["fta"] / g["fga"].replace(0, float("nan"))).round(3)

    # Opponent-faced ratings — pts allowed = DRTG
    opp = sub.merge(sub, on="game_id", suffixes=("_self", "_opp"))
    opp = opp[opp["team_self"] != opp["team_opp"]]
    drtg = (
        opp.groupby("team_self")
        .apply(lambda x: (x["PTS_opp"] / x["pace_opp"]).mean() * 100, include_groups=False)
        .rename("drtg")
        .round(2)
    )
    g = g.join(drtg, how="left")
    g["net"] = (g["ortg"] - g["drtg"]).round(2)
    return g.reset_index().sort_values("net", ascending=False).reset_index(drop=True)


def team_profile(team: str, season) -> dict:
    df = team_profiles(season)
    if df.empty:
        return {}
    row = df[df["team"] == team]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def recent_games(n: int = 20, season="2026") -> pd.DataFrame:
    """Game-level totals, sorted most recent first."""
    sk = _season_key(season)
    pace = load_pace()
    sub = pace[pace["season"] == sk]
    if sub.empty:
        return pd.DataFrame()

    # Join game dates
    year = 2025 if "2025" in sk else 2026
    games = load_games(year)
    date_map = dict(zip(games["game_id"], games["game_date"])) if not games.empty else {}

    rows = []
    for gid, grp in sub.groupby("game_id"):
        if len(grp) != 2:
            continue
        a, b = grp.iloc[0], grp.iloc[1]
        rows.append({
            "game_id":  gid,
            "date":     date_map.get(gid),
            "matchup":  f'{a["team"]} {int(a["PTS"])} - {int(b["PTS"])} {b["team"]}',
            "team_a":   a["team"],
            "pts_a":    int(a["PTS"]),
            "team_b":   b["team"],
            "pts_b":    int(b["PTS"]),
            "total":    int(a["PTS"] + b["PTS"]),
            "pace":     round((a["pace"] + b["pace"]) / 2, 1),
            "fta_a":    int(a["FTA"]),
            "fta_b":    int(b["FTA"]),
        })
    out = pd.DataFrame(rows)
    if "date" in out.columns:
        out = out.sort_values("date", ascending=False, na_position="last")
    return out.head(n).reset_index(drop=True)


def bonus_summary(season) -> pd.DataFrame:
    """% of team-quarters reaching bonus, by quarter."""
    sk = _season_key(season)
    bonus = load_bonus()
    if bonus.empty:
        return pd.DataFrame()
    sub = bonus[bonus["season"] == sk]
    if sub.empty:
        return pd.DataFrame()
    g = (
        sub.groupby("period")
        .agg(team_quarters=("bonus_reached", "count"),
             reached=("bonus_reached", "sum"),
             avg_opp_poss_in_bonus=("opp_poss_in_bonus", "mean"),
             avg_fta_bonus=("fta_bonus", "mean"))
        .reset_index()
    )
    g["pct_reached"] = (g["reached"] / g["team_quarters"] * 100).round(1)
    return g


def foul_rates(season) -> pd.DataFrame:
    """Foul/violation rates per 100 possessions for a season."""
    sk = _season_key(season)
    fr = load_foul_rates()
    if fr.empty:
        return pd.DataFrame()
    return fr[fr["season"] == sk].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian shrinkage helper
# ─────────────────────────────────────────────────────────────────────────────

def regress(team: str, stat: str, season="2026", k: float = 5.0) -> float:
    """Shrink a team's observed per-team-game stat toward the season mean.

    posterior = (n*observed + k*league_mean) / (n + k)
    """
    sk = _season_key(season)
    pace = load_pace()
    sub = pace[pace["season"] == sk]
    if sub.empty:
        return float("nan")

    league_mean = float(sub[stat].mean()) if stat in sub.columns else float("nan")
    team_rows = sub[sub["team"] == team]
    if team_rows.empty:
        return league_mean
    n = len(team_rows)
    observed = float(team_rows[stat].mean())
    return (n * observed + k * league_mean) / (n + k)


def regress_team_pace_ortg(team: str, season="2026", k: float = 5.0) -> dict:
    return {
        "pace": regress(team, "pace", season, k),
        "ortg": regress(team, "ortg", season, k),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for season in ["2025", "2026"]:
        print(f"\n=== {season} league baselines ===")
        bl = league_baselines(season)
        for k, v in bl.items():
            print(f"  {k}: {v:,.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\n=== 2026 team profiles (top 5 net) ===")
    print(team_profiles("2026").head().to_string(index=False))
    print("\n=== Recent 2026 games ===")
    rg = recent_games(10, "2026")
    print(rg[["date", "matchup", "total", "pace"]].to_string(index=False))
    print("\n=== 2026 bonus summary ===")
    print(bonus_summary("2026").to_string(index=False))
    print("\n=== Regression test: MIN ===")
    pace_df = load_pace()
    min_obs = pace_df[(pace_df["season"] == "2026_first8") & (pace_df["team"] == "MIN")]["pace"].mean()
    print(f"  observed pace: {min_obs:.2f}")
    print(f"  regressed (k=5): {regress('MIN', 'pace', '2026', 5):.2f}")
    print(f"  regressed (k=10): {regress('MIN', 'pace', '2026', 10):.2f}")
