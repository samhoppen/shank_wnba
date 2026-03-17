#!/usr/bin/env python3
"""
race_to_10_analysis.py
----------------------
Computes "race to 10" statistics for all D1 teams using PBP + sportsbook
lines data, then projects probabilities for 2026 NCAA tournament matchups.

Steps:
  1. Fetch/cache 2026 season betting lines from CBBD API
  2. Load all PBP files → compute per-game race-to-10 stats
  3. Join with sportsbook spreads (home-team perspective)
  4. Aggregate per team, overall + by spread bucket
  5. Fit logistic model: P(race win) = sigmoid(k * team_spread + b)
  6. Project all 32 first-round tournament matchups

Outputs:
  race_to_10_team_stats.csv     -- per-team aggregated stats (all D1)
  race_to_10_tournament.csv     -- per-matchup projections with EV picks

Required env var:
  CBBD_API_KEY   -- your CollegeBasketballData.com bearer token

Usage:
  python race_to_10_analysis.py
"""

import glob
import os
import sys

import cbbd
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import expit as sigmoid

# ── Config ─────────────────────────────────────────────────────────────────
SEASON      = 2026
PBP_DIR     = "cbbd_data/pbp_flat"
GAMES_DIR   = "cbbd_data/games"
LINES_CACHE = "cbbd_data/lines/2026_lines.csv"
TARGET      = 10    # race to N points
HCA         = 3.5  # fallback home-court advantage (only used when no line exists)

# ── Bracket: tournament matchups ────────────────────────────────────────────
# (team_a, seed_a, team_b, seed_b, region)
# team names must match what's in the ratings/lines data as closely as possible
TOURNAMENT_MATCHUPS = [
    # EAST
    ("Duke",              1,  "Siena",              16, "East"),
    ("Ohio St.",          8,  "TCU",                 9, "East"),
    ("St. John's",        5,  "Northern Iowa",      12, "East"),
    ("Kansas",            4,  "Cal Baptist",        13, "East"),
    ("Louisville",        6,  "South Florida",      11, "East"),
    ("Michigan St.",      3,  "North Dakota St.",   14, "East"),
    ("UCLA",              7,  "UCF",                10, "East"),
    ("UConn",             2,  "Furman",             15, "East"),
    # SOUTH
    ("Florida",           1,  "Lehigh",             16, "South"),
    ("Clemson",           8,  "Iowa",                9, "South"),
    ("Vanderbilt",        5,  "McNeese",            12, "South"),
    ("Nebraska",          4,  "Troy",               13, "South"),
    ("North Carolina",    6,  "VCU",                11, "South"),
    ("Illinois",          3,  "Penn",               14, "South"),
    ("Saint Mary's",      7,  "Texas A&M",          10, "South"),
    ("Houston",           2,  "Idaho",              15, "South"),
    # WEST
    ("Arizona",           1,  "Long Island University", 16, "West"),
    ("Villanova",         8,  "Utah St.",            9, "West"),
    ("Wisconsin",         5,  "High Point",         12, "West"),
    ("Arkansas",          4,  "Hawaii",             13, "West"),
    ("BYU",               6,  "Texas",              11, "West"),
    ("Gonzaga",           3,  "Kennesaw St.",       14, "West"),
    ("Miami (FL)",        7,  "Missouri",           10, "West"),
    ("Purdue",            2,  "Queens University",  15, "West"),
    # MIDWEST
    ("Michigan",          1,  "Howard",             16, "Midwest"),
    ("Georgia",           8,  "Saint Louis",         9, "Midwest"),
    ("Texas Tech",        5,  "Akron",              12, "Midwest"),
    ("Alabama",           4,  "Hofstra",            13, "Midwest"),
    ("Tennessee",         6,  "SMU",                11, "Midwest"),
    ("Virginia",          3,  "Wright St.",         14, "Midwest"),
    ("Kentucky",          7,  "Santa Clara",        10, "Midwest"),
    ("Iowa St.",          2,  "Tennessee St.",      15, "Midwest"),
]

# Alternate name mappings for tournament team lookup
TEAM_ALIASES = {
    "St. John's":       ["St. John's (NY)", "Saint John's"],
    "Michigan St.":     ["Michigan State"],
    "North Dakota St.": ["North Dakota State"],
    "Cal Baptist":      ["California Baptist"],
    "South Florida":    ["USF"],
    "Ohio St.":         ["Ohio State"],
    "Saint Mary's":     ["Saint Mary's (CA)", "St. Mary's"],
    "Texas A&M":        ["Texas A&M"],
    "Long Island University": ["Long Island University"],
    "Utah St.":         ["Utah State"],
    "High Point":       ["High Point"],
    "Kennesaw St.":     ["Kennesaw State"],
    "Miami (FL)":       ["Miami FL", "Miami (FL)"],
    "Iowa St.":         ["Iowa State"],
    "Tennessee St.":    ["Tennessee State"],
    "North Carolina":   ["UNC", "North Carolina"],
    "Wright St.":       ["Wright State"],
    "Santa Clara":      ["Santa Clara"],
    "Gonzaga":          ["Gonzaga"],
}


# ── Helper ──────────────────────────────────────────────────────────────────
def elapsed_secs(period: pd.Series, secs_remaining: pd.Series) -> pd.Series:
    """Convert period + secondsRemaining to total elapsed seconds."""
    return (period - 1) * 1200 + (1200 - secs_remaining)


# ── 1. Fetch / load betting lines ───────────────────────────────────────────
def fetch_lines() -> pd.DataFrame:
    if os.path.exists(LINES_CACHE):
        print(f"  Loading lines from cache: {LINES_CACHE}")
        return pd.read_csv(LINES_CACHE)

    api_key = os.environ.get("CBBD_API_KEY", "").strip()
    if not api_key:
        print("  WARNING: CBBD_API_KEY not set. Skipping lines fetch; "
              "will fall back to ratings-based spreads.")
        return pd.DataFrame()

    print("  Fetching 2026 season lines from CBBD API...")
    configuration = cbbd.Configuration(access_token=api_key)
    rows = []
    with cbbd.ApiClient(configuration) as api_client:
        lines_api = cbbd.LinesApi(api_client)
        raw = lines_api.get_lines(season=SEASON)

    for gl in raw:
        spreads  = [l.spread       for l in gl.lines if l.spread       is not None]
        ous      = [l.over_under   for l in gl.lines if l.over_under   is not None]
        home_mls = [l.home_moneyline for l in gl.lines if l.home_moneyline is not None]
        away_mls = [l.away_moneyline for l in gl.lines if l.away_moneyline is not None]
        rows.append({
            "gameId":          gl.game_id,
            "homeTeam":        gl.home_team,
            "awayTeam":        gl.away_team,
            "startDate":       str(gl.start_date),
            # spread is home-team perspective: negative = home favored
            "spread":          float(np.mean(spreads))  if spreads  else np.nan,
            "overUnder":       float(np.mean(ous))      if ous      else np.nan,
            "homeMoneyline":   float(np.mean(home_mls)) if home_mls else np.nan,
            "awayMoneyline":   float(np.mean(away_mls)) if away_mls else np.nan,
            "n_providers":     len(gl.lines),
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(LINES_CACHE), exist_ok=True)
    df.to_csv(LINES_CACHE, index=False)
    print(f"  Saved {len(df):,} game lines → {LINES_CACHE}")
    return df


# ── 2. Process PBP: per-game race-to-10 stats ───────────────────────────────
def process_pbp() -> pd.DataFrame:
    print("  Loading PBP files...")
    files = sorted(glob.glob(f"{PBP_DIR}/*.csv"))
    if not files:
        sys.exit(f"No PBP files found in {PBP_DIR}/")

    chunks = []
    for f in files:
        df = pd.read_csv(f, usecols=[
            "gameId", "period", "secondsRemaining",
            "homeScore", "awayScore", "scoringPlay", "isHomeTeam",
        ])
        chunks.append(df)
    pbp = pd.concat(chunks, ignore_index=True)
    print(f"  {len(pbp):,} play rows from {len(files)} files")

    # Sort ascending by game time
    pbp["elapsed"] = elapsed_secs(pbp["period"], pbp["secondsRemaining"])
    pbp = pbp.sort_values(["gameId", "elapsed"])

    records = []
    for game_id, grp in pbp.groupby("gameId"):
        grp = grp.reset_index(drop=True)

        # ── Race to TARGET ─────────────────────────────────────────────────
        home_hit = grp[grp["homeScore"] >= TARGET]
        away_hit = grp[grp["awayScore"] >= TARGET]
        if home_hit.empty or away_hit.empty:
            continue

        home_t10 = home_hit.iloc[0]["elapsed"]
        away_t10 = away_hit.iloc[0]["elapsed"]
        home_wins_race = home_t10 < away_t10

        # ── Who scored first? ──────────────────────────────────────────────
        scoring = grp[grp["scoringPlay"] == True]
        if scoring.empty:
            continue
        first = scoring.iloc[0]
        home_scored_first = bool(first["isHomeTeam"]) if pd.notna(first["isHomeTeam"]) else None

        # ── Max early lead (before either team hits TARGET) ────────────────
        early = grp[(grp["homeScore"] < TARGET) & (grp["awayScore"] < TARGET)]
        if not early.empty:
            lead = early["homeScore"] - early["awayScore"]
            max_home_lead = int(lead.max())
            max_away_lead = int((-lead).max())
        else:
            max_home_lead = max_away_lead = 0

        # ── "Blitz": team reaches 10 before opponent reaches 5 ────────────
        away_5 = grp[grp["awayScore"] >= 5]
        home_5 = grp[grp["homeScore"] >= 5]
        home_blitz = bool(home_t10 < (away_5.iloc[0]["elapsed"] if not away_5.empty else np.inf))
        away_blitz = bool(away_t10 < (home_5.iloc[0]["elapsed"] if not home_5.empty else np.inf))

        records.append({
            "gameId":            game_id,
            "home_t10":          home_t10,
            "away_t10":          away_t10,
            "home_wins_race":    home_wins_race,
            "home_scored_first": home_scored_first,
            "max_home_lead":     max_home_lead,
            "max_away_lead":     max_away_lead,
            "home_blitz":        home_blitz,
            "away_blitz":        away_blitz,
        })

    out = pd.DataFrame(records)
    print(f"  Computed race-to-{TARGET} for {len(out):,} games")
    return out


# ── 3. Load games metadata ───────────────────────────────────────────────────
def load_games() -> pd.DataFrame:
    files = sorted(glob.glob(f"{GAMES_DIR}/*.csv"))
    games = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    games = games[games["status"] == "final"].drop_duplicates("id")
    return games[["id", "homeTeam", "awayTeam", "neutralSite"]].rename(columns={"id": "gameId"})


# ── 4. Build team-level records ──────────────────────────────────────────────
def build_team_games(race_df: pd.DataFrame,
                     games: pd.DataFrame,
                     lines: pd.DataFrame) -> pd.DataFrame:
    """
    Merge race data with game metadata and sportsbook spread.
    Returns one row per (team, game), with team_spread expressed as:
      positive  → team is the favorite
      negative  → team is the underdog
    """
    df = race_df.merge(games, on="gameId", how="inner")

    if not lines.empty:
        lines_slim = lines[["gameId", "spread"]].dropna(subset=["spread"])
        df = df.merge(lines_slim, on="gameId", how="left")
        # spread from API: negative = home favored
        # → home team's expected margin = -spread
        df["home_team_spread"] = -df["spread"]
        df["away_team_spread"] =  df["spread"]
    else:
        # No lines: flag as NaN; model will still run but without real lines
        df["home_team_spread"] = np.nan
        df["away_team_spread"] = np.nan

    # Build two rows per game (home perspective + away perspective)
    home_rows = pd.DataFrame({
        "gameId":            df["gameId"],
        "team":              df["homeTeam"],
        "opponent":          df["awayTeam"],
        "is_home":           True,
        "team_spread":       df["home_team_spread"],
        "team_t10":          df["home_t10"],
        "opp_t10":           df["away_t10"],
        "team_wins_race":    df["home_wins_race"],
        "team_scored_first": df["home_scored_first"],
        "max_team_lead":     df["max_home_lead"],
        "team_blitz":        df["home_blitz"],
    })
    away_rows = pd.DataFrame({
        "gameId":            df["gameId"],
        "team":              df["awayTeam"],
        "opponent":          df["homeTeam"],
        "is_home":           False,
        "team_spread":       df["away_team_spread"],
        "team_t10":          df["away_t10"],
        "opp_t10":           df["home_t10"],
        "team_wins_race":    ~df["home_wins_race"],
        "team_scored_first": df["home_scored_first"].map(
                                lambda x: (not x) if x is not None else None),
        "max_team_lead":     df["max_away_lead"],
        "team_blitz":        df["away_blitz"],
    })

    tg = pd.concat([home_rows, away_rows], ignore_index=True)
    tg["team_wins_race"] = tg["team_wins_race"].astype(float)
    tg["team_blitz"]     = tg["team_blitz"].astype(float)
    return tg


# ── 5. Spread bucket helper ──────────────────────────────────────────────────
BUCKETS = [
    ("Fav_10p",  10,   999),
    ("Fav_5_10",  5,    10),
    ("Close",    -5,     5),
    ("Dog_5_10", -10,   -5),
    ("Dog_10p",  -999, -10),
]


def spread_bucket(s: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= s < hi:
            return name
    return "Fav_10p" if s >= 10 else "Dog_10p"


# ── 6. Aggregate per team ────────────────────────────────────────────────────
def aggregate_team_stats(tg: pd.DataFrame) -> pd.DataFrame:
    tg["bucket"] = tg["team_spread"].apply(
        lambda x: spread_bucket(x) if pd.notna(x) else "No_line")

    base = tg.groupby("team").agg(
        games             = ("gameId",          "count"),
        race_win_pct      = ("team_wins_race",  "mean"),
        avg_t10_secs      = ("team_t10",        "mean"),
        scored_first_pct  = ("team_scored_first","mean"),
        avg_max_lead_early= ("max_team_lead",   "mean"),
        blitz_rate        = ("team_blitz",      "mean"),
    ).round(3)

    # Win rate in each spread bucket
    for bucket_name, _, _ in BUCKETS:
        sub = tg[tg["bucket"] == bucket_name]
        col = sub.groupby("team")["team_wins_race"].agg(
            wr=("mean"), n=("count")
        )
        base[f"wr_{bucket_name}"]    = col["wr"].round(3)
        base[f"n_{bucket_name}"]     = col["n"]

    return base.reset_index()


# ── 7. Fit logistic model ────────────────────────────────────────────────────
def fit_model(tg: pd.DataFrame):
    """Fit P(race win) = sigmoid(k * team_spread + b) on all games with lines."""
    model_df = tg.dropna(subset=["team_spread", "team_wins_race"])
    if len(model_df) < 100:
        print("  WARNING: Too few games with spread data to fit model reliably.")
        return 0.05, 0.0  # neutral fallback

    X = model_df["team_spread"].values.astype(float)
    y = model_df["team_wins_race"].values.astype(float)

    def logistic(x, k, b):
        return sigmoid(k * x + b)

    popt, _ = curve_fit(logistic, X, y, p0=[0.05, 0.0], maxfev=20000)
    k, b = popt
    print(f"  Logistic fit: P(win race) = sigmoid({k:.4f} * spread + {b:.4f})")

    # Sanity check: 1-pt fav should win race ~slightly over 50%
    p1 = float(sigmoid(k * 1 + b))
    p10 = float(sigmoid(k * 10 + b))
    print(f"    P(race win) — 1pt fav: {p1:.1%}  |  10pt fav: {p10:.1%}  |  neutral: {sigmoid(b):.1%}")
    return k, b


# ── 8. Tournament projections ────────────────────────────────────────────────
def get_team_stats(team: str, stats_df: pd.DataFrame) -> dict:
    """Look up a team's race-to-10 stats, trying aliases if needed."""
    aliases = [team] + TEAM_ALIASES.get(team, [])
    for name in aliases:
        row = stats_df[stats_df["team"] == name]
        if not row.empty:
            return row.iloc[0].to_dict()
    return {}


def project_tournament(stats_df: pd.DataFrame, k: float, b: float) -> pd.DataFrame:
    rows = []
    for team_a, seed_a, team_b, seed_b, region in TOURNAMENT_MATCHUPS:
        sa = get_team_stats(team_a, stats_df)
        sb = get_team_stats(team_b, stats_df)

        # Tournament games are neutral site, so spread ≈ difference in net ratings
        # Use adj_net if available; fall back to NaN
        # For neutral site: team_spread_a ≈ (adj_net_a - adj_net_b)
        # We'll compute from race_win_pct calibrated by the model instead:
        # best estimate = model using difference in per-team t10 times? No —
        # we need a spread estimate. Use ratings-cache spread approximation.
        net_a = sa.get("adj_net", np.nan)
        net_b = sb.get("adj_net", np.nan)

        if pd.notna(net_a) and pd.notna(net_b):
            spread_a = net_a - net_b   # neutral site; positive = A favored
            p_a = float(sigmoid(k * spread_a + b))
        else:
            # No rating data: use historical race win pct (opponent-agnostic)
            p_a = float(sa.get("race_win_pct", 0.5)) / (
                float(sa.get("race_win_pct", 0.5)) +
                float(sb.get("race_win_pct", 0.5))
            )
            spread_a = np.nan

        p_b = 1.0 - p_a

        # EV calculation (contest scoring)
        seed_diff = abs(seed_a - seed_b)
        bonus = seed_diff / 4.0
        if seed_a < seed_b:   # A is the higher seed (favored conventionally)
            ev_a = p_a * 1.0
            ev_b = p_b * (1.0 + bonus)
        else:
            ev_a = p_a * (1.0 + bonus)
            ev_b = p_b * 1.0

        pick = team_a if ev_a >= ev_b else team_b
        pick_ev = max(ev_a, ev_b)

        rows.append({
            "region":         region,
            "team_a":         team_a,
            "seed_a":         seed_a,
            "team_b":         team_b,
            "seed_b":         seed_b,
            "spread_a":       round(spread_a, 1) if pd.notna(spread_a) else np.nan,
            "p_a_wins_race":  round(p_a, 3),
            "p_b_wins_race":  round(p_b, 3),
            "wr_a_season":    round(sa.get("race_win_pct", np.nan), 3) if sa else np.nan,
            "wr_b_season":    round(sb.get("race_win_pct", np.nan), 3) if sb else np.nan,
            "wr_a_close":     sa.get("wr_Close", np.nan),
            "wr_b_close":     sb.get("wr_Close", np.nan),
            "blitz_a":        round(sa.get("blitz_rate", np.nan), 3) if sa else np.nan,
            "blitz_b":        round(sb.get("blitz_rate", np.nan), 3) if sb else np.nan,
            "avg_t10_a":      round(sa.get("avg_t10_secs", np.nan), 1) if sa else np.nan,
            "avg_t10_b":      round(sb.get("avg_t10_secs", np.nan), 1) if sb else np.nan,
            "ev_a":           round(ev_a, 3),
            "ev_b":           round(ev_b, 3),
            "ev_pick":        pick,
            "ev_pick_value":  round(pick_ev, 3),
        })

    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("RACE TO 10 ANALYSIS — 2026 NCAA Tournament")
    print("=" * 60)

    print("\n[1/5] Lines data")
    lines = fetch_lines()

    print("\n[2/5] PBP processing")
    race_df = process_pbp()

    print("\n[3/5] Games metadata")
    games = load_games()

    print("\n[4/5] Building team-game records")
    tg = build_team_games(race_df, games, lines)
    has_lines = tg["team_spread"].notna().sum()
    print(f"  {len(tg):,} team-game records | {has_lines:,} with sportsbook spread")

    print("\n[5a/5] Aggregating team stats")
    stats = aggregate_team_stats(tg)

    # Attach adj_net from ratings cache for tournament projections
    if os.path.exists("team_ratings_cache.csv"):
        ratings = pd.read_csv("team_ratings_cache.csv")[["team", "adj_net", "tempo"]]
        stats = stats.merge(ratings, on="team", how="left")

    # Flag tournament teams
    tourn_teams = set()
    for t_a, _, t_b, _, _ in TOURNAMENT_MATCHUPS:
        tourn_teams.update([t_a, t_b])
        for alias in TEAM_ALIASES.get(t_a, []):
            tourn_teams.add(alias)
        for alias in TEAM_ALIASES.get(t_b, []):
            tourn_teams.add(alias)
    stats["in_tournament"] = stats["team"].isin(tourn_teams)

    stats.to_csv("race_to_10_team_stats.csv", index=False)
    print(f"  Saved race_to_10_team_stats.csv ({len(stats):,} teams)")

    print("\n[5b/5] Fitting logistic model")
    k, b = fit_model(tg)

    print("\n[5c/5] Tournament projections")
    tourn = project_tournament(stats, k, b)
    tourn.to_csv("race_to_10_tournament.csv", index=False)
    print(f"  Saved race_to_10_tournament.csv ({len(tourn):,} matchups)")

    print("\n── Top Race-to-10 picks (by EV) ─────────────────────────────")
    display = tourn.sort_values("ev_pick_value", ascending=False).head(10)
    for _, row in display.iterrows():
        print(f"  {row['ev_pick']:20s}  EV={row['ev_pick_value']:.3f}  "
              f"(#{row['seed_a']} {row['team_a']} vs #{row['seed_b']} {row['team_b']}, "
              f"{row['region']})")

    print("\n── 1-point favorite race-to-10 baseline ─────────────────────")
    from scipy.special import expit
    p_1pt = float(expit(k * 1 + b))
    p_3pt = float(expit(k * 3 + b))
    p_5pt = float(expit(k * 5 + b))
    print(f"  1-pt fav → {p_1pt:.1%}")
    print(f"  3-pt fav → {p_3pt:.1%}")
    print(f"  5-pt fav → {p_5pt:.1%}")
    print(f"  Neutral  → {float(expit(b)):.1%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
