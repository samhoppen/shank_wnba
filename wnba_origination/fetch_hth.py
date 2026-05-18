"""
Fetch RAPM data from helpthehelper.vercel.app.

The site server-renders the players table and embeds the full dataset as a
`window.PLAYERS = [...]` JavaScript variable. We pull the page, extract the
JSON array, and save to data/hth_players_{season}.csv.

Run:  python fetch_hth.py [season]
Default season: 2026.
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

# Columns we care about for our blended-RAPM use case.
CORE_COLS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE",
    "GP", "MIN", "POSSESSIONS",
    "ORAPM", "DRAPM",      # raw RAPM
    "ORAPM_RANK", "DRAPM_RANK",
    "OFF_RATING_ON", "DEF_RATING_ON", "NET_RATING_ON",
    "OFF_RATING_ON_OFF_DIFF", "DEF_RATING_ON_OFF_DIFF", "NET_RATING_ON_OFF_DIFF",
    "USG_PCT", "TS_PCT", "EFG_PCT",
    "AST_PCT", "TOV_PCT", "OREB_PCT", "DREB_PCT", "STL_PCT", "BLK_PCT",
    "PIE",
]


def fetch_page(season: int = 2026) -> str:
    url = f"https://helpthehelper.vercel.app/players?season={season}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def extract_players(html: str) -> list:
    """Pull `window.PLAYERS = [...]` array out of the page source."""
    m = re.search(r"window\.PLAYERS\s*=\s*(\[.*?\]);", html, re.DOTALL)
    if not m:
        raise RuntimeError("Could not find window.PLAYERS in page HTML")
    return json.loads(m.group(1))


def fetch_and_save(season: int = 2026) -> pd.DataFrame:
    print(f"Fetching helpthehelper players for {season}...")
    html = fetch_page(season)
    players = extract_players(html)
    print(f"  -> parsed {len(players)} players")

    df = pd.DataFrame(players)
    keep = [c for c in CORE_COLS if c in df.columns]
    df = df[keep + [c for c in df.columns if c not in keep]]  # core cols first

    out = DATA_DIR / f"hth_players_{season}.csv"
    df.to_csv(out, index=False)
    print(f"  -> saved {len(df)} rows to {out}")
    return df


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    df = fetch_and_save(season)
    # Quick sanity preview
    cols = ["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN", "ORAPM", "DRAPM"]
    cols = [c for c in cols if c in df.columns]
    if cols:
        print("\nTop 10 by ORAPM:")
        print(df.sort_values("ORAPM", ascending=False)[cols].head(10).to_string(index=False))
        print("\nTop 10 by DRAPM (lowest = best defender if DRAPM is points-allowed style):")
        print(df.sort_values("DRAPM", ascending=False)[cols].head(10).to_string(index=False))
