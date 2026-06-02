"""
WNBA Game Log — four/five factors per team per game, computed from raw PBP JSON.

Five factors:
  1. eFG%   = (FGM + 0.5 * 3PM) / FGA
  2. TOV%   = TOV / (FGA + 0.44 * FTA + TOV)
  3. OREB%  = OREB / (OREB + Opp_DREB)
  4. FT Rate = FTA / FGA
  5. Pace   = total possessions (from stints; estimated from PBP if unavailable)

Run directly to build/rebuild cache:
  python game_log.py              # all years
  python game_log.py --year 2025  # single year
"""
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

from paths import RAPM_DIR, REPO_ROOT, DATA, RAW_PBP_DIR, stints as stints_path

GAME_LOG_CACHE = DATA / "game_log.csv"

# Pull in the PBP possession walker from the wnba_rapm submodule for accurate
# pace counts. Falls back to the box-score formula if the symbol isn't found.
import sys as _sys
_sys.path.insert(0, str(REPO_ROOT / "wnba_rapm"))
try:
    from pbp_shares import walk_possessions
except ImportError:
    walk_possessions = None

# action_type values
FT_TYPES = {"Free Throw"}
REBOUND_TYPES = {"Rebound"}
TURNOVER_TYPES = {"Turnover"}


def _year_from_game_id(game_id: str) -> int:
    """1022500001 → 2025"""
    return 2000 + int(str(game_id)[3:5])


def _load_games_meta(years: list[int]) -> pd.DataFrame:
    """Load home/away team mapping from games CSVs."""
    frames = []
    for yr in years:
        p = RAPM_DIR / f"games_{yr}_Regular_Season.csv"
        if p.exists():
            df = pd.read_csv(p, usecols=["game_id", "game_date", "home_team_id",
                                          "away_team_id", "matchup"])
            df["season"] = yr
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    meta = pd.concat(frames, ignore_index=True)
    meta["game_id"] = meta["game_id"].astype(str)
    return meta


def _compute_factors_from_pbp(pbp: list[dict]) -> dict[int, dict]:
    """
    Compute per-team raw counting stats from a single game's PBP events.
    Returns {team_id: {fga, fgm, fta, ftm, tpa, tpm, oreb, dreb, tov, pts}}
    """
    stats: dict[int, dict] = {}

    def _get(tid):
        if tid not in stats:
            stats[tid] = dict(fga=0, fgm=0, fta=0, ftm=0,
                              tpa=0, tpm=0, oreb=0, dreb=0, tov=0, pts=0)
        return stats[tid]

    for ev in pbp:
        tid = ev.get("team_id", 0)
        if not tid:
            continue
        atype = ev.get("action_type", "")
        sub = ev.get("sub_type", "")
        is_fg = ev.get("isFieldGoal", 0)
        shot_val = ev.get("shotValue", 0)
        result = ev.get("shotResult", "")
        pts = ev.get("pointsTotal", 0)

        s = _get(tid)

        if is_fg:
            s["fga"] += 1
            if result == "Made":
                s["fgm"] += 1
                if shot_val == 3:
                    s["tpm"] += 1
            if shot_val == 3:
                s["tpa"] += 1

        elif atype in FT_TYPES:
            if sub == "Free Throw Technical":
                continue   # skip techs — don't count against FGA
            s["fta"] += 1
            if result == "Made":
                s["ftm"] += 1

        elif atype in REBOUND_TYPES:
            desc = ev.get("description", "")
            if "Off:1" in desc:
                s["oreb"] += 1
            else:
                s["dreb"] += 1

        elif atype in TURNOVER_TYPES:
            s["tov"] += 1

    return stats


def _four_factors(s: dict, opp: dict) -> dict:
    """Compute four/five factors from raw counting stats."""
    fga = s["fga"] or 1
    opp_dreb = opp["dreb"] or 1
    opp_oreb = opp["oreb"] or 1
    total_reb = s["oreb"] + opp["dreb"]

    efg = (s["fgm"] + 0.5 * s["tpm"]) / fga
    tov_pct = s["tov"] / (fga + 0.44 * s["fta"] + s["tov"]) if (fga + 0.44 * s["fta"] + s["tov"]) > 0 else 0
    oreb_pct = s["oreb"] / total_reb if total_reb > 0 else 0
    ft_rate = s["fta"] / fga

    return {
        "efg_pct": round(efg, 3),
        "tov_pct": round(tov_pct, 3),
        "oreb_pct": round(oreb_pct, 3),
        "ft_rate": round(ft_rate, 3),
        "fgm": s["fgm"], "fga": s["fga"],
        "tpm": s["tpm"], "tpa": s["tpa"],
        "ftm": s["ftm"], "fta": s["fta"],
        "oreb": s["oreb"], "dreb": s["dreb"],
        "tov": s["tov"],
    }


def _pace_from_stints(game_id: str, year: int) -> float | None:
    """
    Look up possession count for a game from stints file.
    Each row in stints is one possession (alternating teams), so total rows = 2x per-team pace.
    Divide by 2 to get per-team possessions.
    """
    sp = stints_path(year)
    if not sp.exists():
        return None
    try:
        stints = pd.read_csv(sp, usecols=["game_id"])
        count = (stints["game_id"].astype(str) == game_id).sum()
        return round(count / 2.0, 1) if count > 0 else None
    except Exception:
        return None


def build(years: list[int] | None = None, verbose: bool = True) -> pd.DataFrame:
    """
    Parse all raw PBP JSON files and compute game-level four factors.
    Saves to GAME_LOG_CACHE.
    """
    if years is None:
        years = list(range(2017, 2027))

    meta = _load_games_meta(years)
    meta_idx = meta.set_index("game_id") if not meta.empty else None

    pbp_files = sorted(RAW_PBP_DIR.glob("*_pbp.json"))
    if verbose:
        print(f"Found {len(pbp_files)} PBP files")

    rows = []
    for path in pbp_files:
        game_id = path.stem.replace("_pbp", "")
        year = _year_from_game_id(game_id)
        if year not in years:
            continue

        try:
            with open(path, encoding="utf-8") as f:
                pbp = json.load(f)
        except Exception:
            continue

        team_stats = _compute_factors_from_pbp(pbp)
        if len(team_stats) < 2:
            continue

        # Get home/away from meta
        game_date, home_tid, away_tid, matchup = None, None, None, ""
        if meta_idx is not None and game_id in meta_idx.index:
            row = meta_idx.loc[game_id]
            game_date = row["game_date"]
            home_tid = int(row["home_team_id"])
            away_tid = int(row["away_team_id"])
            matchup = row["matchup"]

        team_ids = list(team_stats.keys())
        if home_tid not in team_ids:
            # Fall back: just use first two teams found
            home_tid, away_tid = team_ids[0], team_ids[1]

        h = team_stats.get(home_tid, {})
        a = team_stats.get(away_tid, {})
        if not h or not a:
            continue

        # Possessions: COUNT actual possessions via PBP walker per team
        # (each Made FG, Made-last-FT, DREB, TOV, or period-end ends a possession).
        # Falls back to box-score formula on walker failure.
        poss = None  # raw per-team possessions (avg of home/away)
        if walk_possessions is not None:
            try:
                pbp_df = pd.DataFrame(pbp)
                teams = [t for t in pbp_df["teamTricode"].dropna().unique() if t]
                counts = [len(walk_possessions(pbp_df, t)) for t in teams]
                poss = round(sum(counts) / len(counts), 1) if counts else None
            except Exception:
                poss = None
        if poss is None:
            def _est_poss(s):
                return s["fga"] - s["oreb"] + s["tov"] + 0.44 * s["fta"]
            poss = round((_est_poss(h) + _est_poss(a)) / 2, 1)

        # OT detection from PBP — max period > 4 means OT.
        # game_minutes = 40 (reg) + 5 × ot_periods.
        max_period = 4
        for ev in pbp:
            try:
                p = int(ev.get("period", 0) or 0)
                if p > max_period:
                    max_period = p
            except Exception:
                pass
        ot_periods = max(0, max_period - 4)
        game_minutes = 40 + ot_periods * 5
        # Pace = possessions normalized to 40-minute game length.
        pace = round(poss * 40.0 / game_minutes, 1) if game_minutes else poss

        hf = _four_factors(h, a)
        af = _four_factors(a, h)

        # Scores from last event (cast to numeric — some PBP feeds store as str)
        last = pbp[-1] if pbp else {}
        home_pts = pd.to_numeric(last.get("score_home"), errors="coerce")
        away_pts = pd.to_numeric(last.get("score_away"), errors="coerce")

        rows.append({
            "game_id": game_id,
            "season": year,
            "game_date": game_date,
            "matchup": matchup,
            "home_team_id": home_tid,
            "away_team_id": away_tid,
            "home_pts": home_pts,
            "away_pts": away_pts,
            "margin": (home_pts - away_pts) if pd.notna(home_pts) and pd.notna(away_pts) else np.nan,
            "pace": pace,
            "poss": poss,
            "ot_periods": ot_periods,
            # Home five factors
            "h_efg": hf["efg_pct"], "h_tov": hf["tov_pct"],
            "h_oreb": hf["oreb_pct"], "h_ftr": hf["ft_rate"],
            "h_fga": hf["fga"], "h_fgm": hf["fgm"],
            "h_tpa": hf["tpa"], "h_tpm": hf["tpm"],
            "h_fta": hf["fta"], "h_ftm": hf["ftm"],
            "h_oreb_n": hf["oreb"], "h_dreb_n": hf["dreb"], "h_tov_n": hf["tov"],
            # Away five factors
            "a_efg": af["efg_pct"], "a_tov": af["tov_pct"],
            "a_oreb": af["oreb_pct"], "a_ftr": af["ft_rate"],
            "a_fga": af["fga"], "a_fgm": af["fgm"],
            "a_tpa": af["tpa"], "a_tpm": af["tpm"],
            "a_fta": af["fta"], "a_ftm": af["ftm"],
            "a_oreb_n": af["oreb"], "a_dreb_n": af["dreb"], "a_tov_n": af["tov"],
        })

    df = pd.DataFrame(rows).sort_values(["season", "game_date", "game_id"])
    GAME_LOG_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(GAME_LOG_CACHE, index=False)
    if verbose:
        print(f"Saved {len(df)} games -> {GAME_LOG_CACHE}")
    return df


def load() -> pd.DataFrame:
    if not GAME_LOG_CACHE.exists():
        return build()
    return pd.read_csv(GAME_LOG_CACHE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None)
    args = parser.parse_args()
    years = [args.year] if args.year else None
    df = build(years=years)
    print(df[["game_date", "matchup", "home_pts", "away_pts", "pace",
              "h_efg", "h_tov", "h_oreb", "h_ftr",
              "a_efg", "a_tov", "a_oreb", "a_ftr"]].tail(10).to_string())
