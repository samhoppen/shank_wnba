"""
Cross-reference our PBP-walker minute counts vs ESPN box-score minutes
from data/espn_box_2026.tsv.

Reports per-(player, game) deltas and summary stats.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

_CLOCK_PAT = re.compile(r"PT(\d+)M([\d.]+)S")
_SUB_PAT   = re.compile(r"SUB:\s*(.+?)\s+FOR\s+(.+)")

# Pull walker function out of app.py
APP_SRC = (HERE / "app.py").read_text(encoding="utf-8")
walker_src = APP_SRC[APP_SRC.find("def _game_presence_from_pbp"):APP_SRC.find("def _recent_game_presence")]
exec(walker_src, globals())

from paths import RAPM_DIR
from matchup import ABBR_TO_TEAM_ID


def mp_to_min(mp_str: str) -> float:
    m, _, s = mp_str.partition(":")
    return int(m) + int(s) / 60.0


def main():
    box = pd.read_csv(HERE / "data" / "espn_box_2026.tsv", sep="\t")
    games_2026 = pd.read_csv(RAPM_DIR / "games_2026_Regular_Season.csv", dtype={"game_id": str})
    games_2026["game_id"] = games_2026["game_id"].astype(str)
    games_2026["game_date"] = pd.to_datetime(games_2026["game_date"]).dt.strftime("%Y-%m-%d")

    # Find game_id by (date, team home or away)
    def find_game(date: str, team: str) -> str | None:
        tid = ABBR_TO_TEAM_ID.get(team)
        if tid is None:
            return None
        g = games_2026[(games_2026["game_date"] == date)
                       & ((games_2026["home_team_id"] == tid) | (games_2026["away_team_id"] == tid))]
        if g.empty:
            return None
        return str(g.iloc[0]["game_id"]).zfill(10)

    rows = []
    # Cache walker output per (game_id, team) so we don't re-walk for every player
    cache: dict = {}

    for _, r in box.iterrows():
        player = r["Player"]
        team = r["Team"]
        date = r["Date"]
        espn_mp = mp_to_min(r["MP"])
        gid = find_game(date, team)
        if gid is None:
            rows.append({"player": player, "team": team, "date": date,
                         "espn_min": espn_mp, "ours_min": None, "delta": None,
                         "note": "no game_id"})
            continue
        tid = ABBR_TO_TEAM_ID[team]

        key = (gid, tid)
        if key not in cache:
            pbp_path = RAPM_DIR / "raw_pbp" / f"{gid}_pbp.json"
            sp_path  = RAPM_DIR / "raw_pbp" / f"{gid}_starters.json"
            if not pbp_path.exists():
                cache[key] = None
            else:
                with open(pbp_path, encoding="utf-8") as f:
                    pbp = json.load(f)
                starters_team: set = set()
                if sp_path.exists():
                    with open(sp_path, encoding="utf-8") as f:
                        sd = json.load(f)
                    raw = sd.get(str(tid))
                    if raw:
                        starters_team = {int(x) for x in raw}
                cache[key] = _game_presence_from_pbp(pbp, tid, starters_team)  # noqa: F821
        result = cache[key]
        if result is None:
            pres, mins = None, None
        else:
            pres, mins = result
        if pres is None:
            rows.append({"player": player, "team": team, "date": date,
                         "espn_min": espn_mp, "ours_min": None, "delta": None,
                         "note": "no pbp"})
            continue

        # Find player_id by name in PBP (last-name match)
        # Build a name -> pid lookup from PBP for this team
        pbp_path = RAPM_DIR / "raw_pbp" / f"{gid}_pbp.json"
        with open(pbp_path, encoding="utf-8") as f:
            pbp_for_lookup = json.load(f)
        name_to_pid: dict = {}
        for ev in pbp_for_lookup:
            if int(ev.get("team_id", 0) or 0) != tid:
                continue
            pid = ev.get("person_id")
            pname = str(ev.get("playerName", "") or "").strip()
            if pid and pname:
                try:
                    name_to_pid[pname] = int(pid)
                except Exception:
                    pass

        # Match player name — try last-name, full-name, last-name lowercase
        last = player.split()[-1]
        pid = name_to_pid.get(last)
        if pid is None:
            # try exact match on any name
            for k, v in name_to_pid.items():
                if k.split()[-1] == last:
                    pid = v
                    break
        if pid is None:
            rows.append({"player": player, "team": team, "date": date,
                         "espn_min": espn_mp, "ours_min": None, "delta": None,
                         "note": f"no pid for {last}"})
            continue

        ours = mins.get(pid, 0.0) if mins else float(sum(pres.get(pid, [False] * 40)))
        delta = ours - espn_mp
        rows.append({"player": player, "team": team, "date": date,
                     "espn_min": round(espn_mp, 2), "ours_min": round(ours, 2),
                     "delta": round(delta, 2), "note": ""})

    df = pd.DataFrame(rows)
    df["abs_delta"] = df["delta"].abs()
    # Summary
    print(f"Total comparisons: {len(df)}")
    matched = df.dropna(subset=["ours_min"])
    print(f"Matched: {len(matched)}")
    print(f"Mean abs delta: {matched['abs_delta'].mean():.2f} min")
    print(f"Median abs delta: {matched['abs_delta'].median():.2f} min")
    print()

    # Distribution of abs_delta
    print("Delta buckets:")
    for thr in [1, 2, 3, 5, 10]:
        cnt = (matched["abs_delta"] <= thr).sum()
        print(f"  within ±{thr} min: {cnt}/{len(matched)} ({cnt/len(matched)*100:.0f}%)")
    print()

    # Top 20 worst mismatches
    print("=== Top 20 worst (sign matters) ===")
    print(matched.nlargest(20, "abs_delta")[["player","team","date","espn_min","ours_min","delta"]].to_string(index=False))


if __name__ == "__main__":
    main()
