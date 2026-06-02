"""
regen_analysis.py — rebuild the analysis CSVs the wnba_rapm submodule omits.

Produces, into wnba_origination/data/:
    pace_stats.csv          — per (game_id, team): pace, ortg, PTS, FGA, FTA, TOV, OREB
    bonus_by_quarter.csv    — per (game_id, team, period): bonus_reached, opp poss, fta_bonus
    ft_decomp.csv           — per (game_id, team): fta_shooting / fta_bonus / fta_tech totals
    foul_violation_rates.csv — per (season, category, sub_type): total + per_100_poss

Sources:
    - Raw PBP JSON in RAW_PBP_DIR (populated by scripts/fetch_pbp.py)
    - games_{year}_Regular_Season.csv from the wnba_rapm submodule
    - stints_rich_{year}_RS.csv from the wnba_rapm submodule

Usage:
    python -m wnba_origination.scripts.regen_analysis --year 2026
    python -m wnba_origination.scripts.regen_analysis --year 2026 --append-2025
    python -m wnba_origination.scripts.regen_analysis --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from paths import RAPM_DIR, DATA, RAW_PBP_DIR, stints as stints_path  # noqa: E402
from game_log import _compute_factors_from_pbp  # noqa: E402

# Map calendar year → season key used by league_stats.SEASON_KEYS.
# 2025 is treated as the full season; everything else is "to-date" with the
# legacy "_first8" suffix the dashboard already knows.
SEASON_KEY = {
    2025: "2025_full",
    2026: "2026_first8",
}


def _season_key(year: int) -> str:
    return SEASON_KEY.get(year, f"{year}_first8")


def _games_meta(year: int) -> pd.DataFrame:
    p = RAPM_DIR / f"games_{year}_Regular_Season.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, dtype={"game_id": str})
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


def _team_id_to_tricode(meta: pd.DataFrame) -> dict[int, str]:
    """Best-effort team_id → tricode lookup from games CSV `matchup` strings."""
    out: dict[int, str] = {}
    if meta.empty:
        return out
    import re as _re
    for _, row in meta.iterrows():
        matchup = str(row.get("matchup", ""))
        parts = _re.split(r"\s+vs\.?\s+", matchup)
        if len(parts) != 2:
            continue
        home_abbr, away_abbr = parts[0].strip(), parts[1].strip()
        try:
            home_id = int(row["home_team_id"])
            away_id = int(row["away_team_id"])
        except Exception:
            continue
        out.setdefault(home_id, home_abbr)
        out.setdefault(away_id, away_abbr)
    return out


def _pbp_files_for_year(year: int) -> list[Path]:
    """All raw PBP files for one calendar year, matched by game_id prefix."""
    prefix = f"10{str(year)[2:]}"  # e.g. 1025 for 2025
    return sorted(RAW_PBP_DIR.glob(f"{prefix}*_pbp.json"))


def _stints_possessions_per_team_game(year: int) -> dict[tuple[str, int], int]:
    """Return {(game_id, team_id): possessions} from the year's stints file."""
    sp = stints_path(year)
    if not sp.exists():
        return {}
    df = pd.read_csv(sp, usecols=["game_id", "off_team"], dtype={"game_id": str})
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    counts = df.groupby(["game_id", "off_team"]).size().to_dict()
    return {(gid, int(tid)): n for (gid, tid), n in counts.items()}


# ── pace_stats + ft_decomp + bonus + foul_rates (single PBP walk) ────────────

def _walk_one_game(
    game_id: str,
    pbp: list[dict],
    tricode: dict[int, str],
    poss_by_team: dict[tuple[str, int], int],
    year: int,
    game_date: str | None,
) -> dict:
    """One pass through a game's PBP. Returns a dict with four per-game frames."""
    season = _season_key(year)
    stats = _compute_factors_from_pbp(pbp)

    # OT periods → pace normalisation
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

    # ── Per-team accumulators for FT decomp + foul counts per quarter ───────
    fta_decomp: dict[int, dict] = {}  # team_id -> {shooting, bonus, tech}
    team_period_fouls: dict[tuple[int, int], int] = {}   # (team, period) -> shooting+personal-LB
    period_team_fouls: dict[tuple[int, int], dict] = {}  # (team, period) -> {shooting, personal_lb}
    opp_poss_after_bonus: dict[tuple[int, int], int] = {}
    fta_bonus_count: dict[tuple[int, int], int] = {}
    bonus_threshold = 5  # WNBA team-foul bonus

    poss_counter: dict[tuple[int, int], int] = {}  # (team, period) -> non-team total possessions seen

    def _ftd(tid: int) -> dict:
        if tid not in fta_decomp:
            fta_decomp[tid] = dict(fta_shooting=0, fta_bonus=0, fta_tech=0)
        return fta_decomp[tid]

    last_off_team: int | None = None

    for ev in pbp:
        try:
            tid = int(ev.get("team_id", 0) or 0)
        except Exception:
            tid = 0
        try:
            period = int(ev.get("period", 0) or 0)
        except Exception:
            period = 0
        atype = str(ev.get("action_type", "") or "")
        sub = str(ev.get("sub_type", "") or "")
        result = str(ev.get("shotResult", "") or "")
        is_fg = ev.get("isFieldGoal", 0)
        desc = str(ev.get("description", "") or "")

        # Approximate possession counting for the opposing team while in bonus
        # (used by bonus_by_quarter). True possession-walk lives in pace_stats.
        if is_fg and result == "Made":
            last_off_team = tid
        elif atype == "Free Throw" and result == "Made" and "1 of 1" in (sub + " " + desc).lower():
            last_off_team = tid
        elif atype == "Turnover":
            last_off_team = tid  # turnover ends a possession on `tid`
        elif atype == "Rebound" and "Off:0" in desc:
            # defensive rebound — possession switches
            last_off_team = tid

        # ── FT decomposition ───────────────────────────────────────────────
        if atype == "Free Throw" and tid:
            d = _ftd(tid)
            sub_l = sub.lower()
            if "technical" in sub_l or "flagrant" in sub_l:
                d["fta_tech"] += 1
            elif "free throw 1 of 1" in sub_l:
                # 1-of-1: could be and-1 or bonus. Heuristic: bonus if
                # the team was already in bonus this quarter.
                fouls = team_period_fouls.get((_opp(tid, tricode), period), 0)
                if fouls >= bonus_threshold:
                    d["fta_bonus"] += 1
                else:
                    d["fta_shooting"] += 1
            else:
                d["fta_shooting"] += 1

            # Track FTAs accumulated against the opposing team while in bonus
            opp = _opp(tid, tricode)
            if opp and team_period_fouls.get((opp, period), 0) >= bonus_threshold:
                fta_bonus_count[(opp, period)] = fta_bonus_count.get((opp, period), 0) + 1

        # ── Foul tally for bonus tracking ──────────────────────────────────
        if atype == "Foul" and tid:
            pf = period_team_fouls.setdefault((tid, period), dict(shooting=0, personal_lb=0))
            if "Shooting" in sub:
                pf["shooting"] += 1
                team_period_fouls[(tid, period)] = team_period_fouls.get((tid, period), 0) + 1
            elif sub in ("Personal", "Loose Ball"):
                pf["personal_lb"] += 1
                team_period_fouls[(tid, period)] = team_period_fouls.get((tid, period), 0) + 1
            # Offensive / Technical / Flagrant don't count toward team-foul bonus

        # ── Possession reach tracking once opposing team is in bonus ───────
        if last_off_team and tid and tid != last_off_team:
            opp_team = last_off_team
            if team_period_fouls.get((opp_team, period), 0) >= bonus_threshold:
                # Whoever has the ball is taking possessions vs a bonus-team defense
                if is_fg or atype in ("Turnover",):
                    key = (opp_team, period)
                    opp_poss_after_bonus[key] = opp_poss_after_bonus.get(key, 0) + 1

    # ── Build per-game frames ─────────────────────────────────────────────
    # pace_stats rows
    pace_rows: list[dict] = []
    team_ids = list(stats.keys())
    for tid in team_ids:
        s = stats[tid]
        team_tricode = tricode.get(tid, str(tid))
        poss = poss_by_team.get((game_id, tid))
        if poss is None or poss == 0:
            poss = s["fga"] - s["oreb"] + s["tov"] + 0.44 * s["fta"]
        poss = float(poss)
        pace = round(poss * 40.0 / game_minutes, 1) if game_minutes else round(poss, 1)
        ortg = round(s["pts"] / poss * 100, 1) if poss else 0.0
        pace_rows.append({
            "season":  _season_key(year),
            "game_id": game_id,
            "team":    team_tricode,
            "team_id": tid,
            "FGA":     s["fga"],
            "FTA":     s["fta"],
            "TOV":     s["tov"],
            "OREB":    s["oreb"],
            "PTS":     s["pts"],
            "poss":    round(poss, 1),
            "pace":    pace,
            "ortg":    ortg,
            "ot_periods":   ot_periods,
            "game_minutes": game_minutes,
        })

    # ft_decomp rows (minimal schema — the loader exists but isn't actively
    # consumed by the dashboard at the moment)
    ft_rows = []
    for tid, d in fta_decomp.items():
        ft_rows.append({
            "game_id": game_id,
            "team": tricode.get(tid, str(tid)),
            "team_id": tid,
            "season": _season_key(year),
            "fta_shooting": d["fta_shooting"],
            "fta_bonus":    d["fta_bonus"],
            "fta_tech":     d["fta_tech"],
            "fta_total":    d["fta_shooting"] + d["fta_bonus"] + d["fta_tech"],
        })

    # bonus_by_quarter rows
    bonus_rows = []
    periods_seen = {p for (_t, p) in team_period_fouls.keys()}
    periods_seen |= {p for (_t, p) in period_team_fouls.keys()}
    for tid in team_ids:
        for period in sorted(p for p in periods_seen if p):
            if period > 4:
                continue
            pf = period_team_fouls.get((tid, period), dict(shooting=0, personal_lb=0))
            tf = team_period_fouls.get((tid, period), 0)
            reached = tf >= bonus_threshold
            opp_p = opp_poss_after_bonus.get((tid, period), 0)
            ftab = fta_bonus_count.get((tid, period), 0)
            bonus_rows.append({
                "game_id":           game_id,
                "period":            period,
                "fouling_team":      tricode.get(tid, str(tid)),
                "bonus_reached":     bool(reached),
                "team_fouls":        tf,
                "shooting_fouls":    pf["shooting"],
                "personal_lb_fouls": pf["personal_lb"],
                "opp_poss_in_bonus": opp_p,
                "fta_bonus":         ftab,
                "season":            _season_key(year),
                "game_date":         game_date or "",
            })

    return {
        "pace_stats":         pace_rows,
        "ft_decomp":          ft_rows,
        "bonus_by_quarter":   bonus_rows,
        "foul_events":        _foul_events_from_pbp(pbp, _season_key(year)),
        "team_possessions":   {tid: poss_by_team.get((game_id, tid), 0) for tid in team_ids},
    }


def _opp(team_id: int, tricode: dict[int, str]) -> int:
    """Return the OTHER team_id we know about. Returns 0 if unable."""
    others = [t for t in tricode.keys() if t != team_id]
    return others[0] if len(others) == 1 else 0


def _foul_events_from_pbp(pbp: list[dict], season_key: str) -> list[tuple]:
    """Return [(season, category, sub_type)] rows for the foul-rate aggregation."""
    rows = []
    for ev in pbp:
        atype = str(ev.get("action_type", "") or "")
        sub = str(ev.get("sub_type", "") or "")
        if not sub:
            continue
        if atype == "Foul":
            rows.append((season_key, "Foul", sub))
        elif atype == "Violation":
            rows.append((season_key, "Violation", sub))
        elif atype == "Turnover" and sub in ("Offensive Foul Turnover", "Traveling"):
            rows.append((season_key, "Called Violation", sub))
    return rows


# ── Top-level builders ───────────────────────────────────────────────────────

def build_for_year(year: int) -> dict:
    meta = _games_meta(year)
    if meta.empty:
        print(f"  ! games_{year}_Regular_Season.csv missing — skipping {year}")
        return {"pace_stats": [], "ft_decomp": [], "bonus_by_quarter": [],
                "foul_events": [], "n_games": 0, "total_poss": 0}
    tricode = _team_id_to_tricode(meta)
    date_map = dict(zip(meta["game_id"], meta["game_date"].dt.strftime("%Y-%m-%d")))

    poss_by_team = _stints_possessions_per_team_game(year)

    pbp_files = _pbp_files_for_year(year)
    print(f"  {year}: {len(pbp_files)} PBP files in cache")
    if not pbp_files:
        return {"pace_stats": [], "ft_decomp": [], "bonus_by_quarter": [],
                "foul_events": [], "n_games": 0, "total_poss": 0}

    all_pace: list[dict] = []
    all_ftd: list[dict] = []
    all_bonus: list[dict] = []
    all_foul_events: list[tuple] = []
    total_poss = 0

    for path in pbp_files:
        gid = path.stem.replace("_pbp", "")
        try:
            with open(path, encoding="utf-8") as f:
                pbp = json.load(f)
        except Exception as exc:
            print(f"  ! {gid}: {exc}")
            continue
        out = _walk_one_game(
            game_id=gid,
            pbp=pbp,
            tricode=tricode,
            poss_by_team=poss_by_team,
            year=year,
            game_date=date_map.get(gid),
        )
        all_pace.extend(out["pace_stats"])
        all_ftd.extend(out["ft_decomp"])
        all_bonus.extend(out["bonus_by_quarter"])
        all_foul_events.extend(out["foul_events"])
        total_poss += sum(out["team_possessions"].values())

    return {
        "pace_stats": all_pace,
        "ft_decomp": all_ftd,
        "bonus_by_quarter": all_bonus,
        "foul_events": all_foul_events,
        "n_games": len(pbp_files),
        "total_poss": total_poss,
    }


def _foul_rates_frame(per_year: dict[int, list[tuple]],
                      poss_per_year: dict[int, int]) -> pd.DataFrame:
    rows = []
    for year, events in per_year.items():
        season = _season_key(year)
        total_poss = poss_per_year.get(year, 0)
        if not events:
            continue
        df = pd.DataFrame(events, columns=["season", "category", "sub_type"])
        agg = df.groupby(["season", "category", "sub_type"]).size().reset_index(name="total")
        # per-game using games CSV count
        gm = _games_meta(year)
        n_games = int(gm["game_id"].nunique()) if not gm.empty else 0
        agg["per_game"] = (agg["total"] / n_games).round(3) if n_games else 0.0
        if total_poss > 0:
            agg["per_100_poss"] = (agg["total"] / total_poss * 100).round(4)
        else:
            agg["per_100_poss"] = 0.0
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, help="Single year to (re)build")
    parser.add_argument("--append-2025", action="store_true",
                        help="Also process 2025 and concat into the same CSVs")
    parser.add_argument("--all", action="store_true",
                        help="Rebuild for all years 2017-2026")
    args = parser.parse_args()

    if args.all:
        years = list(range(2017, 2027))
    elif args.year:
        years = [args.year] + ([2025] if args.append_2025 and args.year != 2025 else [])
    else:
        parser.error("provide --year (optionally with --append-2025) or --all")

    pace_frames: list[pd.DataFrame] = []
    ftd_frames: list[pd.DataFrame] = []
    bonus_frames: list[pd.DataFrame] = []
    foul_events_by_year: dict[int, list[tuple]] = {}
    poss_by_year: dict[int, int] = {}

    for y in years:
        out = build_for_year(y)
        if out["pace_stats"]:
            pace_frames.append(pd.DataFrame(out["pace_stats"]))
        if out["ft_decomp"]:
            ftd_frames.append(pd.DataFrame(out["ft_decomp"]))
        if out["bonus_by_quarter"]:
            bonus_frames.append(pd.DataFrame(out["bonus_by_quarter"]))
        foul_events_by_year[y] = out["foul_events"]
        poss_by_year[y] = out["total_poss"]

    DATA.mkdir(parents=True, exist_ok=True)

    if pace_frames:
        pd.concat(pace_frames, ignore_index=True).to_csv(DATA / "pace_stats.csv", index=False)
        print(f"wrote {DATA / 'pace_stats.csv'}")
    if ftd_frames:
        pd.concat(ftd_frames, ignore_index=True).to_csv(DATA / "ft_decomp.csv", index=False)
        print(f"wrote {DATA / 'ft_decomp.csv'}")
    if bonus_frames:
        pd.concat(bonus_frames, ignore_index=True).to_csv(DATA / "bonus_by_quarter.csv", index=False)
        print(f"wrote {DATA / 'bonus_by_quarter.csv'}")

    fr = _foul_rates_frame(foul_events_by_year, poss_by_year)
    if not fr.empty:
        fr.to_csv(DATA / "foul_violation_rates.csv", index=False)
        print(f"wrote {DATA / 'foul_violation_rates.csv'}")


if __name__ == "__main__":
    main()
