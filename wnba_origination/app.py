"""
WNBA Line Origination — Streamlit UI

Tabs:
  Game        — rotation editor (minutes 0-40, inline RAPM edits) + spread/total output
  Season      — win total projections for all 14+ teams
  Roster      — persistent player overrides: team assignments, RAPM edits, CSV upload
  Performance — model vs actuals + market lines
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import player_store
import matchup as mx
import win_totals as wt
import performance as perf
import roster as roster_mod
import pace as pace_module
from matchup import ml_from_prob, ABBR_TO_TEAM_ID
from components.rotation_chart import rotation_chart

st.set_page_config(
    page_title="WNBA Lines",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TEAMS = sorted([
    "ATL", "CHI", "CON", "DAL", "GSV", "IND",
    "LAS", "LVA", "MIN", "NYL", "PHX", "SEA",
    "WAS", "POR", "TOR",
])

# ── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading player data...")
def get_store() -> pd.DataFrame:
    base = player_store.load()
    return roster_mod.apply(base)


@st.cache_data(ttl=3600, show_spinner="Loading pace model...")
def get_pace_cache() -> pd.DataFrame:
    try:
        return pace_module.load_pace_cache()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner="Projecting win totals...")
def get_win_totals() -> pd.DataFrame:
    return wt.project_all_teams(store=get_store())


@st.cache_data(ttl=3600, show_spinner="Loading performance data...")
def get_performance() -> pd.DataFrame:
    return perf.load_results()


def _game_log_mtime() -> float:
    """Return mtime of game log CSV so cache busts when file is rebuilt."""
    import game_log as gl
    p = gl.GAME_LOG_CACHE
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner="Loading game log...")
def get_game_log(_mtime: float = 0.0) -> pd.DataFrame:
    import game_log as gl
    return gl.load()


@st.cache_data(ttl=86400, show_spinner="Loading EC historical data...")
def get_ec_historical(store: pd.DataFrame) -> pd.DataFrame:
    from paths import EC_ALL_SEASONS
    if not EC_ALL_SEASONS.exists():
        return pd.DataFrame()

    ec = pd.read_csv(EC_ALL_SEASONS)

    # Build name -> (player_id, orapm, drapm, minutes) lookup from store
    # Deduplicate on player_name keeping highest-minutes row
    store_dedup = store.sort_values("minutes", ascending=False).drop_duplicates("player_name")
    store_lookup = store_dedup.set_index("player_name")[["player_id", "orapm", "drapm", "minutes"]]
    store_lookup_lower = store_dedup.copy()
    store_lookup_lower["player_name_lower"] = store_lookup_lower["player_name"].str.lower()
    store_lookup_lower = store_lookup_lower.drop_duplicates("player_name_lower").set_index("player_name_lower")[["player_id", "orapm", "drapm", "minutes"]]

    def _match(name):
        if name in store_lookup.index:
            return store_lookup.loc[name]
        nl = str(name).lower()
        if nl in store_lookup_lower.index:
            return store_lookup_lower.loc[nl]
        return pd.Series({"player_id": np.nan, "orapm": np.nan, "drapm": np.nan, "minutes": np.nan})

    rapm_cols = ec["Player"].apply(_match)
    ec = ec.join(rapm_cols.reset_index(drop=True))
    ec["net_rapm"] = ec["orapm"] + ec["drapm"]

    # Rename for display
    ec = ec.rename(columns={"Season": "season", "Player": "player", "Team": "team",
                             "Mins": "mins", "minutes": "curr_mins_2025"})
    return ec


# ── Helpers ──────────────────────────────────────────────────────────────────

def spread_display(spread: float, home: str, away: str) -> str:
    if spread > 0:
        return f"{home} -{abs(spread):.1f}"
    elif spread < 0:
        return f"{away} -{abs(spread):.1f}"
    return "PK"


def _default_minutes(team_abbr: str, store: pd.DataFrame) -> pd.DataFrame:
    """Return top-12 players for a team with projected minutes (0-40 scale)."""
    roster = store[store["team_abbr"] == team_abbr].copy()
    roster = roster[roster["minutes"] > 0].sort_values("minutes", ascending=False).head(12)
    if roster.empty:
        return roster

    # Scale historical minutes share to projected game minutes (sum ~200 for 5 players × 40 min)
    total = roster["minutes"].sum()
    roster["proj_minutes"] = (roster["minutes"] / total * 200).clip(0, 40).round(1)
    return roster.reset_index(drop=True)


def _build_lineup_from_editor(editor_state: dict) -> dict[int, float]:
    """Convert session state from rotation editor into {player_id: minutes}."""
    return {pid: mins for pid, mins in editor_state.items() if mins > 0}


# Team colors for stint blocks
TEAM_COLORS: dict[str, dict[str, str]] = {
    #        bg (primary)   fg (secondary / text)
    "ATL": {"bg": "#C8102E", "fg": "#418FDE"},   # red    / blue
    "CHI": {"bg": "#418FDE", "fg": "#FFCD00"},   # blue   / yellow
    "CON": {"bg": "#F05023", "fg": "#041E42"},   # orange / navy
    "DAL": {"bg": "#C4D600", "fg": "#00235D"},   # lime   / navy
    "GSV": {"bg": "#AD96DC", "fg": "#010101"},   # valkyrie violet / black
    "IND": {"bg": "#041E42", "fg": "#C8102E"},   # navy   / red
    "LAS": {"bg": "#702F8A", "fg": "#FFC72C"},   # purple / gold
    "LVA": {"bg": "#010101", "fg": "#BA0C2F"},   # black  / aces red
    "MIN": {"bg": "#266092", "fg": "#FFFFFF"},   # blue   / white
    "NYL": {"bg": "#6ECEB2", "fg": "#FF671F"},   # teal   / orange
    "PHX": {"bg": "#CB6015", "fg": "#201747"},   # orange / navy
    "POR": {"bg": "#E93CAC", "fg": "#C8102E"},   # pink   / red
    "SEA": {"bg": "#2C5234", "fg": "#FBE122"},   # green  / yellow
    "TOR": {"bg": "#612C51", "fg": "#B8CCEA"},   # bordeaux / sky blue
    "WAS": {"bg": "#0C2340", "fg": "#C8102E"},   # navy   / red
}

def _team_bg(abbr: str) -> str:
    return TEAM_COLORS.get(abbr, {"bg": "#1e3a5f"})["bg"]

def _team_fg(abbr: str) -> str:
    return TEAM_COLORS.get(abbr, {"fg": "#FFFFFF"})["fg"]

def team_badge(abbr: str, size: str = "1.3rem") -> str:
    """Return an HTML span styled as a team color badge."""
    bg, fg = _team_bg(abbr), _team_fg(abbr)
    return (
        f'<span style="background:{bg};color:{fg};padding:4px 10px;'
        f'border-radius:5px;font-weight:700;font-size:{size};'
        f'letter-spacing:0.05em;display:inline-block;line-height:1.4">{abbr}</span>'
    )


# Columns treated as pre-multiplied percentages (already ×100) in game log table
_PCT_COLS = {"eFG%", "TOV%", "OREB%", "FTR",
             "H eFG%", "H TOV%", "H OREB%", "H FTR",
             "A eFG%", "A TOV%", "A OREB%", "A FTR"}
_INT_COLS = {"FGA", "3PA", "FTA", "OREB", "TOV", "Pts", "Opp",
             "home_pts", "away_pts", "Pace", "pace"}


def _html_table(df: pd.DataFrame, height: int = 580, table_id: str = "gl") -> str:
    """Render a DataFrame as a styled, sortable HTML table with team badge cells."""
    TH = ("background:#f8f8f8;position:sticky;top:0;z-index:2;"
          "padding:6px 10px;text-align:left;font-size:12px;cursor:pointer;"
          "color:#555;border-bottom:2px solid #e0e0e0;white-space:nowrap;"
          "user-select:none")
    TD_BASE = "padding:5px 10px;font-size:12px;border-bottom:1px solid #f0f0f0"
    TD_NUM  = TD_BASE + ";text-align:right"

    header = "".join(
        f'<th style="{TH}" onclick="sortTable(\'{table_id}\',{i})">'
        f'{c} <span id="{table_id}-arrow-{i}" style="font-size:9px;color:#aaa">⇅</span></th>'
        for i, c in enumerate(df.columns)
    )

    rows_html = []
    for i, (_, row) in enumerate(df.iterrows()):
        bg = "#fafafa" if i % 2 == 0 else "#ffffff"
        cells = []
        for col in df.columns:
            val = row[col]
            if col in ("Home", "Away"):
                # store raw abbr in data-sort so JS can sort by it
                cell = (f'<td style="{TD_BASE}" data-sort="{val}">'
                        f'{team_badge(str(val), "0.72rem")}</td>')
            elif col == "game_date":
                cell = f'<td style="{TD_BASE};color:#777" data-sort="{val}">{val}</td>'
            elif col in _PCT_COLS:
                txt = f"{val:.1f}%" if pd.notna(val) else "—"
                sv  = val if pd.notna(val) else -999
                cell = f'<td style="{TD_NUM}" data-sort="{sv}">{txt}</td>'
            elif col in ("margin", "Margin"):
                if pd.notna(val):
                    color = "#2a9d8f" if val > 0 else "#e76f51" if val < 0 else "#777"
                    sign  = "+" if val > 0 else ""
                    txt   = f"{sign}{int(val)}"
                    sv    = val
                else:
                    color, txt, sv = "#777", "—", -9999
                cell = (f'<td style="{TD_NUM};color:{color};font-weight:600" '
                        f'data-sort="{sv}">{txt}</td>')
            elif col in _INT_COLS or col in ("Pace",):
                sv  = val if pd.notna(val) else -999
                txt = f"{int(val)}" if pd.notna(val) else "—"
                cell = f'<td style="{TD_NUM}" data-sort="{sv}">{txt}</td>'
            elif isinstance(val, (int, float)):
                sv  = val if pd.notna(val) else -999
                txt = f"{val:.1f}" if pd.notna(val) else "—"
                cell = f'<td style="{TD_NUM}" data-sort="{sv}">{txt}</td>'
            else:
                cell = f'<td style="{TD_BASE}" data-sort="{val}">{val}</td>'
            cells.append(cell)
        rows_html.append(f'<tr style="background:{bg}">{"".join(cells)}</tr>')

    sort_js = f"""
<script>
(function(){{
  var _dir = {{}};
  window.sortTable = function(tid, col) {{
    var tbl = document.getElementById(tid);
    var tbody = tbl.querySelector('tbody');
    var rows = Array.from(tbody.querySelectorAll('tr'));
    _dir[col] = !_dir[col];
    // update all arrows
    tbl.querySelectorAll('[id^="' + tid + '-arrow-"]').forEach(function(el) {{
      el.textContent = '⇅'; el.style.color='#aaa';
    }});
    var arrow = document.getElementById(tid + '-arrow-' + col);
    if (arrow) {{ arrow.textContent = _dir[col] ? ' ▲' : ' ▼'; arrow.style.color='#333'; }}
    rows.sort(function(a, b) {{
      var av = a.cells[col].getAttribute('data-sort');
      var bv = b.cells[col].getAttribute('data-sort');
      var an = parseFloat(av), bn = parseFloat(bv);
      if (!isNaN(an) && !isNaN(bn)) return _dir[col] ? an-bn : bn-an;
      return _dir[col] ? av.localeCompare(bv) : bv.localeCompare(av);
    }});
    // restripe after sort
    rows.forEach(function(r, i) {{
      r.style.background = i%2===0 ? '#fafafa' : '#ffffff';
      tbody.appendChild(r);
    }});
  }};
}})();
</script>"""

    table_html = (
        f'<div style="overflow-y:auto;max-height:{height}px;border:1px solid #e8e8e8;border-radius:6px">'
        f'<table id="{table_id}" style="width:100%;border-collapse:collapse;font-family:system-ui,sans-serif">'
        f'<thead><tr>{header}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table></div>'
    )
    # Wrap in full document so script executes inside the st.components.v1.html iframe
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>body{{margin:0;padding:0;background:transparent}}</style>
</head><body>{table_html}{sort_js}</body></html>"""


# ── Game Tab ─────────────────────────────────────────────────────────────────

def _players_for_chart(team_abbr: str, store: pd.DataFrame) -> list[dict]:
    """Build player list for the rotation chart component.
    If a saved rotation exists it is the sole source of who appears — player_store
    only supplies RAPM values, not the roster itself."""
    saved = roster_mod.get_rotation(team_abbr)

    if saved is not None:
        # Saved rotation drives the player list
        store_idx = store.set_index("player_id")
        players = []
        for _, row in saved.iterrows():
            pid = row["player_id"]
            if pd.notna(pid):
                pid = int(pid)
                orapm = float(store_idx.loc[pid, "orapm"]) if pid in store_idx.index and pd.notna(store_idx.loc[pid, "orapm"]) else 0.0
                drapm = float(store_idx.loc[pid, "drapm"]) if pid in store_idx.index and pd.notna(store_idx.loc[pid, "drapm"]) else 0.0
            else:
                pid = -abs(hash(str(row["player_name"]))) % 10**7  # stable fake id
                orapm, drapm = 0.0, 0.0
            players.append({
                "player_id": pid,
                "player_name": str(row["player_name"]),
                "default_minutes": float(row["projected_minutes"]),
                "orapm": orapm,
                "drapm": drapm,
            })
        return players

    # No saved rotation — fall back to player_store
    roster = _default_minutes(team_abbr, store)
    if roster.empty:
        return []
    return [
        {
            "player_id": int(row["player_id"]),
            "player_name": str(row["player_name"]),
            "default_minutes": float(row["proj_minutes"]),
            "orapm": float(row["orapm"]) if pd.notna(row["orapm"]) else 0.0,
            "drapm": float(row["drapm"]) if pd.notna(row["drapm"]) else 0.0,
        }
        for _, row in roster.iterrows()
    ]


@st.cache_data(ttl=3600, show_spinner=False)
def _recent_game_presence(team_abbr: str, n: int = 5, year: int = 2025) -> list[dict]:
    """
    For each of the last N games, return minute-level player presence derived from stints.
    Each row index is mapped to a game-minute bucket (0-39).
    Returns list of {game_date, opponent, is_home, presence: {player_id: [bool]*40}}
    ordered most-recent first.
    """
    import re as _re
    from paths import RAPM_DIR, stints as stints_path
    from matchup import ABBR_TO_TEAM_ID

    team_id = ABBR_TO_TEAM_ID.get(team_abbr)
    if team_id is None:
        return []

    games_p = RAPM_DIR / f"games_{year}_Regular_Season.csv"
    if not games_p.exists():
        return []
    games = pd.read_csv(games_p, usecols=["game_id", "game_date", "home_team_id", "away_team_id", "matchup"])
    games["game_id"] = games["game_id"].astype(str)
    team_games = games[
        (games["home_team_id"] == team_id) | (games["away_team_id"] == team_id)
    ].sort_values("game_date", ascending=False).head(n)

    if team_games.empty:
        return []

    sp = stints_path(year)
    if not sp.exists():
        return []
    stints_df = pd.read_csv(sp, dtype={"game_id": str})

    results = []
    for _, grow in team_games.iterrows():
        gid = str(grow["game_id"])
        gdate = str(grow["game_date"])[:10]
        matchup = str(grow.get("matchup", ""))
        is_home = int(grow["home_team_id"]) == team_id

        # Opponent abbreviation
        parts = _re.split(r"\s+vs\.?\s+", matchup)
        if len(parts) == 2:
            home_abbr, away_abbr = parts[0].strip(), parts[1].strip()
            opp = away_abbr if is_home else f"@{home_abbr}"
        else:
            opp = "?"

        gs = stints_df[stints_df["game_id"] == gid].reset_index(drop=True)
        if gs.empty:
            continue

        total = len(gs)
        presence: dict[int, list[bool]] = {}

        for row_i, row in gs.iterrows():
            game_min = min(int(row_i * 40 / total), 39)

            if row["off_team"] == team_id:
                p_cols = ["off_p1", "off_p2", "off_p3", "off_p4", "off_p5"]
            elif row["def_team"] == team_id:
                p_cols = ["def_p1", "def_p2", "def_p3", "def_p4", "def_p5"]
            else:
                continue

            for col in p_cols:
                val = row.get(col)
                if pd.notna(val):
                    pid = int(val)
                    if pid not in presence:
                        presence[pid] = [False] * 40
                    presence[pid][game_min] = True

        results.append({
            "game_id": gid,
            "game_date": gdate,
            "opponent": opp,
            "is_home": is_home,
            "presence": presence,
        })

    return results  # most-recent first


@st.cache_data(ttl=86400, show_spinner=False)
def _game_box_score(game_id: str) -> pd.DataFrame:
    """Parse raw PBP and return per-player box score (pts/fgm/fga/3pm/3pa/ftm/fta/reb/ast/tov)."""
    import json as _json, re as _re
    from paths import RAPM_DIR
    pbp_path = RAPM_DIR / "raw_pbp" / f"{game_id}_pbp.json"
    if not pbp_path.exists():
        return pd.DataFrame()
    with open(pbp_path, encoding="utf-8") as f:
        pbp = _json.load(f)

    stats: dict[int, dict] = {}
    name_to_pid: dict[str, int] = {}   # last-name → person_id for assist matching

    def _get(pid, tid, full_name, last_name):
        if pid not in stats:
            stats[pid] = dict(team_id=tid, name=full_name,
                              pts=0, fgm=0, fga=0, tpm=0, tpa=0,
                              ftm=0, fta=0, reb=0, ast=0, tov=0)
        name_to_pid.setdefault(last_name, pid)
        return stats[pid]

    for ev in pbp:
        pid  = ev.get("person_id", 0)
        tid  = ev.get("team_id", 0)
        name = ev.get("playerNameI", ev.get("playerName", ""))
        last = ev.get("playerName", "").split()[-1] if ev.get("playerName") else ""
        if not pid or not tid:
            continue
        at   = ev.get("action_type", "")
        sv   = ev.get("shotValue", 0)
        desc = ev.get("description", "")
        s    = _get(pid, tid, name, last)

        if at == "Made Shot":
            s["pts"] += sv if sv else 2   # shotValue = 2 or 3
            s["fgm"] += 1; s["fga"] += 1
            if sv == 3:
                s["tpm"] += 1; s["tpa"] += 1
            # Parse assist: "(Wilson 1 AST)"
            m = _re.search(r"\((\w+)\s+\d+\s+AST\)", desc)
            if m:
                a_last = m.group(1)
                # Store for second pass (assister may not be seen yet)
                s.setdefault("_ast_pending", []).append(a_last)
        elif at == "Missed Shot":
            s["fga"] += 1
            if sv == 3:
                s["tpa"] += 1
        elif at == "Free Throw":
            s["fta"] += 1
            if ev.get("shotResult") == "Made":
                s["ftm"] += 1; s["pts"] += 1
        elif at == "Rebound":
            s["reb"] += 1
        elif at == "Turnover":
            s["tov"] += 1

    # Second pass: credit assists
    for pid, s in stats.items():
        for a_last in s.pop("_ast_pending", []):
            a_pid = name_to_pid.get(a_last)
            if a_pid and a_pid in stats:
                stats[a_pid]["ast"] += 1

    rows = [{"player_id": pid, "name": s["name"], "team_id": s["team_id"],
             "pts": s["pts"], "fg": f"{s['fgm']}-{s['fga']}",
             "3p": f"{s['tpm']}-{s['tpa']}", "ft": f"{s['ftm']}-{s['fta']}",
             "reb": s["reb"], "ast": s["ast"], "tov": s["tov"]}
            for pid, s in stats.items() if s["fga"] + s["fta"] + s["reb"] + s["tov"] > 0]
    return pd.DataFrame(rows)


def _presence_grid_html(games_data: list[dict], player_id: int, team_abbr: str) -> tuple[str, int]:
    """
    Minute-by-minute on/off heatmap for one player across recent games.
    Rows = games (most-recent first). Shows total mins on the right.
    """
    tbg  = _team_bg(team_abbr)
    CELL_W  = 10
    CELL_H  = 18
    LABEL_W = 72
    TOT_W   = 30

    q_header = (
        f'<tr><td style="width:{LABEL_W}px"></td>'
        + "".join(
            f'<td colspan="10" style="font-size:9px;color:#888;text-align:center;'
            f'border-left:2px solid #ccc;padding:1px 0">Q{q+1}</td>'
            for q in range(4)
        )
        + f'<td style="width:{TOT_W}px"></td></tr>'
    )

    rows_html = []
    for game in games_data:
        presence = game["presence"].get(player_id, [False] * 40)
        gdate = game["game_date"][5:]
        opp   = game["opponent"]
        total_on = sum(presence)
        cells = ""
        for m in range(40):
            on = presence[m]
            border = "border-left:2px solid #ccc;" if m % 10 == 0 else "border-left:1px solid #e8e8e8;"
            bg = tbg if on else "#ececec"
            cells += f'<td style="width:{CELL_W}px;height:{CELL_H}px;{border}background:{bg};padding:0"></td>'
        rows_html.append(
            f'<tr>'
            f'<td style="width:{LABEL_W}px;font-size:10px;color:#555;padding:1px 5px 1px 2px;'
            f'white-space:nowrap;text-align:right">{gdate}&nbsp;<b>{opp}</b></td>'
            + cells +
            f'<td style="width:{TOT_W}px;font-size:10px;color:#888;padding-left:4px">{total_on}m</td>'
            f'</tr>'
        )

    total_h = (len(games_data) + 1) * (CELL_H + 2) + 20
    table = (
        f'<table style="border-collapse:collapse;font-family:system-ui,sans-serif">'
        f'<thead>{q_header}</thead><tbody>{"".join(rows_html)}</tbody>'
        f'</table>'
    )
    return f"<!DOCTYPE html><html><body style='margin:0;padding:0'>{table}</body></html>", total_h


def _team_rotation_grid_html(
    games_data: list[dict],
    players: list[dict],
    team_abbr: str,
) -> tuple[str, int]:
    """
    Multi-game rotation overview.
    Rows = players (sorted by minutes), columns = minute cells across last N games.
    """
    tbg    = _team_bg(team_abbr)
    CELL_W = 9
    CELL_H = 20
    LABEL_W = 95
    TOT_W   = 34
    GAP_W   = 5   # divider between games

    n_games = len(games_data)
    if n_games == 0 or not players:
        return "<html><body></body></html>", 50

    # Compute total on-court minutes per player across all games
    all_pids = {p["player_id"] for p in players}
    pid_total: dict[int, int] = {pid: 0 for pid in all_pids}
    for g in games_data:
        for pid, pres in g["presence"].items():
            if pid in pid_total:
                pid_total[pid] = pid_total[pid] + sum(pres)

    # Order players: most total minutes first
    ordered = sorted(players, key=lambda p: -pid_total.get(p["player_id"], 0))
    # Only show players with any presence
    ordered = [p for p in ordered if pid_total.get(p["player_id"], 0) > 0]

    # ── Header: game dates ──
    hdr_cells = f'<td style="width:{LABEL_W}px"></td>'
    for g_i, game in enumerate(games_data):
        if g_i > 0:
            hdr_cells += f'<td style="width:{GAP_W}px"></td>'
        gdate = game["game_date"][5:]
        opp   = game["opponent"]
        hdr_cells += (
            f'<td colspan="40" style="font-size:9px;color:#444;text-align:center;'
            f'border-left:2px solid #aaa;padding:1px 0;white-space:nowrap">'
            f'<b>{opp}</b>&nbsp;{gdate}</td>'
        )
    hdr_cells += f'<td style="width:{TOT_W}px;font-size:9px;color:#aaa;text-align:center">min</td>'
    header_row = f'<tr>{hdr_cells}</tr>'

    # ── Quarter sub-header ──
    q_cells = f'<td style="width:{LABEL_W}px"></td>'
    for g_i in range(n_games):
        if g_i > 0:
            q_cells += f'<td style="width:{GAP_W}px"></td>'
        for q in range(4):
            bl = "border-left:2px solid #aaa;" if q == 0 else "border-left:1px solid #ddd;"
            q_cells += (
                f'<td colspan="10" style="font-size:8px;color:#bbb;text-align:center;{bl}padding:0">Q{q+1}</td>'
            )
    q_cells += f'<td style="width:{TOT_W}px"></td>'
    q_row = f'<tr>{q_cells}</tr>'

    # ── Player rows ──
    rows_html = []
    for i, p in enumerate(ordered):
        pid        = p["player_id"]
        short_name = p["player_name"].split()[-1]
        net        = p["orapm"] + p["drapm"]
        net_color  = tbg if net >= 0 else "#e76f51"
        row_bg     = "#fafafa" if i % 2 == 0 else "#f2f2f2"
        total_on   = pid_total.get(pid, 0)

        label_cell = (
            f'<td style="width:{LABEL_W}px;text-align:right;padding:1px 6px;'
            f'white-space:nowrap;background:{row_bg}">'
            f'<span style="font-size:11px;font-weight:600;color:#222">{short_name}</span>&nbsp;'
            f'<span style="font-size:9px;color:{net_color}">{net:+.1f}</span></td>'
        )

        game_cells = ""
        for g_i, game in enumerate(games_data):
            if g_i > 0:
                game_cells += f'<td style="width:{GAP_W}px;background:#fff"></td>'
            presence = game["presence"].get(pid, [False] * 40)
            for m in range(40):
                on  = presence[m]
                bl  = "border-left:2px solid #ccc;" if m % 10 == 0 else "border-left:1px solid #e8e8e8;"
                bg  = tbg if on else "#ececec"
                game_cells += (
                    f'<td style="width:{CELL_W}px;height:{CELL_H}px;{bl}background:{bg};padding:0"></td>'
                )

        tot_cell = (
            f'<td style="width:{TOT_W}px;text-align:left;font-size:10px;color:#777;'
            f'padding:0 2px 0 5px;background:{row_bg}">{total_on}m</td>'
        )
        rows_html.append(f'<tr>{label_cell}{game_cells}{tot_cell}</tr>')

    total_h = (len(ordered) + 2) * (CELL_H + 2) + 36
    table = (
        f'<div style="overflow-x:auto">'
        f'<table style="border-collapse:collapse;font-family:system-ui,sans-serif">'
        f'<thead>{header_row}{q_row}</thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table></div>'
    )
    return f"<!DOCTYPE html><html><body style='margin:0;padding:4px'>{table}</body></html>", total_h


def rotation_editor(
    team_abbr: str,
    side: str,
    store: pd.DataFrame,
) -> tuple[dict[int, float], dict[int, float]]:
    """
    Render the drag-and-drop rotation chart + per-player minute controls + game log grids.
    Returns (lineup_minutes, rapm_overrides).
    """
    import streamlit.components.v1 as _stc

    players = _players_for_chart(team_abbr, store)
    if not players:
        st.warning(f"No players found for {team_abbr}. Add them in the Roster tab.")
        return {}, {}

    # Collect DNP player IDs from session state (set by previous render)
    dnp_pids = [
        p["player_id"] for p in players
        if st.session_state.get(f"{side}_dnp_{p['player_id']}", False)
    ]

    # Drag-and-drop stint chart — returns {player_id: total_minutes}
    lineup_raw = rotation_chart(
        players=players,
        label="",
        team_color=_team_bg(team_abbr),
        text_color=_team_fg(team_abbr),
        team_key=team_abbr,
        forced_zeros=dnp_pids,
        key=f"chart_{side}",
    )
    lineup: dict[int, float] = {pid: mins for pid, mins in lineup_raw.items() if mins > 0}

    # Load recent game presence once for the whole team
    games_data = _recent_game_presence(team_abbr, n=5)

    tbg = _team_bg(team_abbr)
    # Order by descending chart minutes (matches React sort)
    all_pids_ordered = sorted(players, key=lambda p: -lineup.get(p["player_id"], 0))

    adjusted_lineup: dict[int, float] = {}

    for p in all_pids_ordered:
        pid        = p["player_id"]
        chart_mins = lineup.get(pid, 0.0)
        short_name = p["player_name"].split()[-1]
        net        = p["orapm"] + p["drapm"]
        net_color  = tbg if net >= 0 else "#e76f51"

        # Row: [name+RAPM] [min input] [OUT checkbox]
        col_name, col_mins, col_dnp = st.columns([2.5, 1.2, 0.8])

        col_name.markdown(
            f"<div style='padding-top:6px;font-size:12px'>"
            f"<b>{short_name}</b>&nbsp;"
            f"<span style='color:{net_color};font-size:10px'>{net:+.1f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Read DNP state FIRST (from previous render's session state)
        _dnp_key = f"{side}_dnp_{pid}"
        _inp_key = f"{side}_min_{pid}"
        _ck      = f"_cv_{side}_{pid}"
        _prev_dnp = st.session_state.get(_dnp_key, False)

        # Sync chart → input when chart changed (skip if DNP)
        _prev = st.session_state.get(_ck)
        if _prev_dnp:
            st.session_state[_inp_key] = 0.0   # DNP forces 0
        elif _prev is not None and abs(_prev - chart_mins) > 0.01:
            st.session_state[_inp_key] = chart_mins
        st.session_state[_ck] = chart_mins

        new_mins = col_mins.number_input(
            "min",
            min_value=0.0, max_value=40.0,
            value=float(chart_mins),
            step=0.5, format="%.1f",
            label_visibility="collapsed",
            key=_inp_key,
        )

        is_dnp = col_dnp.checkbox(
            "OUT", value=False,
            key=_dnp_key,
            help="Mark as DNP — zeroed from projection",
        )

        if not is_dnp and new_mins > 0:
            adjusted_lineup[pid] = new_mins

        # Per-player game log expander (minute grid)
        player_games = [g for g in games_data if pid in g["presence"]]
        if player_games:
            with st.expander(
                f"📊 {short_name} — last {len(player_games)} games",
                expanded=False,
            ):
                html_doc, grid_h = _presence_grid_html(player_games, pid, team_abbr)
                _stc.html(html_doc, height=grid_h, scrolling=False)

    lineup = adjusted_lineup

    # Total minutes indicator
    total_mins = sum(lineup.values())
    color_ind = "green" if 195 <= total_mins <= 205 else "orange" if 180 <= total_mins <= 220 else "red"
    st.markdown(
        f"<small>Total on floor: <b style='color:{color_ind}'>{total_mins:.1f} min</b></small>",
        unsafe_allow_html=True,
    )

    # Inline RAPM overrides — collapsed
    rapm_overrides: dict[int, tuple[float, float]] = {}
    with st.expander("RAPM overrides (session only)", expanded=False):
        active_players = [p for p in players if lineup.get(p["player_id"], 0) > 0]
        if active_players:
            active_players = sorted(active_players, key=lambda p: p["orapm"] + p["drapm"], reverse=True)
            hdr = st.columns([3, 1, 1, 1])
            hdr[0].markdown("**Player**"); hdr[1].markdown("**oRAPM**")
            hdr[2].markdown("**dRAPM**");  hdr[3].markdown("**net**")
            for p in active_players:
                pid = p["player_id"]
                rc0, rc1, rc2, rc3 = st.columns([3, 1, 1, 1])
                rc0.markdown(f"<small>{p['player_name']}</small>", unsafe_allow_html=True)
                o = rc1.number_input("o", value=p["orapm"], step=0.1, format="%.2f",
                                     label_visibility="collapsed", key=f"{side}_o_{pid}")
                d = rc2.number_input("d", value=p["drapm"], step=0.1, format="%.2f",
                                     label_visibility="collapsed", key=f"{side}_d_{pid}")
                net = o + d
                net_color = "green" if net >= 0 else "red"
                rc3.markdown(f"<small style='color:{net_color}'><b>{net:+.2f}</b></small>",
                             unsafe_allow_html=True)
                if o != p["orapm"] or d != p["drapm"]:
                    rapm_overrides[pid] = (o, d)

    return lineup, rapm_overrides


def _apply_session_rapm(
    store: pd.DataFrame,
    home_overrides: dict[int, tuple[float, float]],
    away_overrides: dict[int, tuple[float, float]],
) -> pd.DataFrame:
    """Apply session-only RAPM edits to a copy of the store for this projection."""
    store = store.copy()
    for pid, (orapm, drapm) in {**home_overrides, **away_overrides}.items():
        mask = store["player_id"] == pid
        if mask.any():
            store.loc[mask, "orapm"] = orapm
            store.loc[mask, "drapm"] = drapm
    return store


def tab_game(store: pd.DataFrame, pace_cache: pd.DataFrame) -> None:
    st.header("Game Projector")

    top = st.columns([2, 1, 2])
    with top[0]:
        home_team = st.selectbox("Home Team", TEAMS, index=TEAMS.index("LVA"), key="home_team")
    with top[2]:
        away_team = st.selectbox("Away Team", TEAMS, index=TEAMS.index("NYL"), key="away_team")
    with top[1]:
        st.markdown("<br>", unsafe_allow_html=True)
        neutral = st.checkbox("Neutral site", value=False, key="neutral")
        pace_lock = st.checkbox("Lock pace", value=False, key="pace_lock",
                                help="Override dynamic pace prediction")
        locked_pace = None
        if pace_lock:
            locked_pace = st.number_input("Pace", min_value=60, max_value=95,
                                          value=mx.PACE_DEFAULT, step=1, key="locked_pace")

    st.divider()

    lcol, rcol = st.columns(2)
    with lcol:
        st.markdown(team_badge(home_team, "1.4rem"), unsafe_allow_html=True)
        st.markdown("")
        home_lineup, home_rapm_overrides = rotation_editor(home_team, "home", store)
    with rcol:
        st.markdown(team_badge(away_team, "1.4rem"), unsafe_allow_html=True)
        st.markdown("")
        away_lineup, away_rapm_overrides = rotation_editor(away_team, "away", store)

    # Project
    if not home_lineup or not away_lineup:
        st.info("Both lineups need at least one player with minutes > 0.")
        return

    hca = 0.0 if neutral else mx.HCA
    session_store = _apply_session_rapm(store, home_rapm_overrides, away_rapm_overrides)

    home_team_id = ABBR_TO_TEAM_ID.get(home_team)
    away_team_id = ABBR_TO_TEAM_ID.get(away_team)

    result = mx.project_matchup(
        home_lineup, away_lineup,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        pace=locked_pace,
        hca=hca,
        store=session_store,
        pace_cache=pace_cache if not pace_cache.empty else None,
    )

    st.divider()

    r1, r2, r3, r4, r5, r6 = st.columns(6)
    home_ml, away_ml = ml_from_prob(result["win_prob_home"])
    spread_str = spread_display(result["spread"], home_team, away_team)

    home_ml_str = f"{home_ml:+d}"
    away_ml_str = f"{away_ml:+d}"

    r1.metric("Spread", spread_str)
    r2.metric("Total", f"{result['total']:.1f}")
    with r3:
        st.markdown(team_badge(home_team, "0.75rem"), unsafe_allow_html=True)
        st.metric("", f"{result['home_pts']:.1f}")
    with r4:
        st.markdown(team_badge(away_team, "0.75rem"), unsafe_allow_html=True)
        st.metric("", f"{result['away_pts']:.1f}")
    r5.metric("Pace", f"{result['pace']:.0f}")
    with r6:
        st.markdown(team_badge(home_team, "0.75rem"), unsafe_allow_html=True)
        st.metric("Win%", f"{result['win_prob_home']:.1%}")

    with st.expander("ML / RAPM Details"):
        d1, d2, d3, d4 = st.columns(4)
        d1.metric(f"{home_team} ML", home_ml_str)
        d2.metric(f"{away_team} ML", away_ml_str)
        d3.metric(f"{home_team} oRAPM", f"{result['home_orapm']:+.2f}")
        d3.metric(f"{home_team} dRAPM", f"{result['home_drapm']:+.2f}")
        d4.metric(f"{away_team} oRAPM", f"{result['away_orapm']:+.2f}")
        d4.metric(f"{away_team} dRAPM", f"{result['away_drapm']:+.2f}")

    # Save rotation button
    scol1, scol2, _ = st.columns([1, 1, 4])
    if scol1.button("Save home rotation"):
        _save_rotation_from_editor(home_team, home_lineup, store)
        st.success(f"Saved {home_team} rotation")
    if scol2.button("Save away rotation"):
        _save_rotation_from_editor(away_team, away_lineup, store)
        st.success(f"Saved {away_team} rotation")


def _save_rotation_from_editor(team_abbr: str, lineup: dict[int, float], store: pd.DataFrame) -> None:
    name_map = dict(zip(store["player_id"], store["player_name"]))
    rows = [{"player_id": pid, "player_name": name_map.get(pid, str(pid)),
             "projected_minutes": mins} for pid, mins in lineup.items()]
    df = pd.DataFrame(rows)
    roster_mod.set_rotation(team_abbr, df)


# ── Season Tab ───────────────────────────────────────────────────────────────

def tab_season() -> None:
    st.header("Season Win Totals")
    df = get_win_totals()
    if df.empty:
        st.warning("No team data available.")
        return

    st.markdown("**Enter sportsbook win total lines:**")
    line_cols = st.columns(5)
    lines: dict[str, float] = {}
    for i, row in df.iterrows():
        col = line_cols[i % 5]
        lines[row["team"]] = col.number_input(
            row["team"], min_value=0.0, max_value=44.0,
            value=float(row["proj_wins"]), step=0.5,
            key=f"line_{row['team']}",
        )

    st.divider()

    display = df[["team", "orapm", "drapm", "net_rapm", "proj_wins",
                  "p_ge_30", "p_ge_32", "p_ge_34", "p_ge_36",
                  "p_lt_15", "p_lt_10"]].copy()

    pricing_rows = []
    for _, row in df.iterrows():
        line = lines.get(row["team"], row["proj_wins"])
        pricing = wt.fair_ml_win_total(row["proj_wins"], line)
        pricing_rows.append({
            "team": row["team"],
            "O/U Line": line,
            "Proj W": row["proj_wins"],
            "P(Over)": f"{pricing['p_over']:.1%}",
            "Over ML": f"{pricing['over_ml']:+d}",
            "Under ML": f"{pricing['under_ml']:+d}",
        })

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.dataframe(
            display.style.format({
                "orapm": "{:+.2f}", "drapm": "{:+.2f}", "net_rapm": "{:+.2f}",
                "proj_wins": "{:.1f}",
                "p_ge_30": "{:.1%}", "p_ge_32": "{:.1%}",
                "p_ge_34": "{:.1%}", "p_ge_36": "{:.1%}",
                "p_lt_15": "{:.1%}", "p_lt_10": "{:.1%}",
            }),
            use_container_width=True, hide_index=True,
        )
    with col_b:
        st.dataframe(pd.DataFrame(pricing_rows), use_container_width=True, hide_index=True)


# ── Roster Tab ───────────────────────────────────────────────────────────────

def tab_roster(store: pd.DataFrame) -> None:
    st.header("Roster Management")
    st.caption("Changes here persist across sessions and sit on top of the daily player store refresh.")

    subtab1, subtab2, subtab3 = st.tabs(["Player Overrides", "Team Rosters", "CSV Upload"])

    # ── Player Overrides ──
    with subtab1:
        st.markdown("Edit team assignment or RAPM for any player. Saves to `roster_overrides.csv`.")

        overrides = roster_mod.load_roster_overrides()

        search = st.text_input("Search player", key="roster_search")
        view = store.copy()
        if search:
            view = view[view["player_name"].str.contains(search, case=False, na=False)]
        view = view.head(50)

        st.markdown("**Current player store** (showing top 50 / search results):")
        st.dataframe(
            view.assign(net=view["orapm"] + view["drapm"])[["player_id", "player_name", "team_abbr", "orapm", "drapm", "net", "minutes"]],
            use_container_width=True, hide_index=True,
        )

        st.divider()
        st.markdown("**Add / edit override:**")

        oc1, oc2 = st.columns(2)
        with oc1:
            sel_name = st.selectbox(
                "Select player",
                options=store["player_name"].sort_values().tolist(),
                key="override_player_sel",
            )
            sel_row = store[store["player_name"] == sel_name].iloc[0] if sel_name else None

        if sel_row is not None:
            pid = int(sel_row["player_id"])
            existing = overrides[overrides["player_id"] == pid]

            with oc1:
                new_team = st.selectbox(
                    "Team", ["(keep current)"] + TEAMS,
                    index=0, key="override_team",
                )
            with oc2:
                cur_orapm = float(existing["orapm_override"].iloc[0]) if not existing.empty and pd.notna(existing["orapm_override"].iloc[0]) else float(sel_row["orapm"])
                cur_drapm = float(existing["drapm_override"].iloc[0]) if not existing.empty and pd.notna(existing["drapm_override"].iloc[0]) else float(sel_row["drapm"])
                new_orapm = st.number_input("oRAPM override", value=cur_orapm, step=0.1, format="%.2f", key="override_orapm")
                new_drapm = st.number_input("dRAPM override", value=cur_drapm, step=0.1, format="%.2f", key="override_drapm")
                notes = st.text_input("Notes (e.g. injured, traded)", key="override_notes")

            bc1, bc2 = st.columns(2)
            if bc1.button("Save override"):
                roster_mod.upsert_player(
                    player_id=pid,
                    player_name=str(sel_row["player_name"]),
                    team_abbr=new_team if new_team != "(keep current)" else None,
                    orapm_override=new_orapm if new_orapm != float(sel_row["orapm"]) else None,
                    drapm_override=new_drapm if new_drapm != float(sel_row["drapm"]) else None,
                    notes=notes,
                )
                st.cache_data.clear()
                st.success(f"Saved override for {sel_name}")
                st.rerun()

            if not existing.empty:
                if bc2.button("Remove override"):
                    roster_mod.remove_player(pid)
                    st.cache_data.clear()
                    st.success(f"Removed override for {sel_name}")
                    st.rerun()

        if not overrides.empty:
            st.divider()
            st.markdown("**Active overrides:**")
            st.dataframe(overrides, use_container_width=True, hide_index=True)

    # ── Team Rosters ──
    with subtab2:
        st.markdown("View or clear saved rotations per team.")
        team_sel = st.selectbox("Team", TEAMS, key="roster_team_sel")
        saved = roster_mod.get_rotation(team_sel)

        if saved is not None:
            st.markdown(f"Saved rotation for &nbsp;{team_badge(team_sel)}", unsafe_allow_html=True)
            store_idx = store.set_index("player_id")
            def _enrich(row):
                pid = row["player_id"]
                if pd.notna(pid) and int(pid) in store_idx.index:
                    r = store_idx.loc[int(pid)]
                    return pd.Series({"orapm": round(r["orapm"], 2), "drapm": round(r["drapm"], 2), "net": round(r["orapm"] + r["drapm"], 2)})
                return pd.Series({"orapm": 0.0, "drapm": 0.0, "net": 0.0})
            enriched = saved.join(saved.apply(_enrich, axis=1))
            st.dataframe(enriched[["player_name", "projected_minutes", "orapm", "drapm", "net"]],
                         use_container_width=True, hide_index=True)
            if st.button("Clear rotation"):
                roster_mod.clear_rotation(team_sel)
                st.success(f"Cleared {team_sel} rotation")
                st.rerun()
        else:
            st.markdown(
                f"No saved rotation for &nbsp;{team_badge(team_sel)}&nbsp; — build one in the Game tab and hit Save.",
                unsafe_allow_html=True,
            )

        # Build from scratch (for POR/TOR expansion teams)
        st.divider()
        st.markdown("**Build rotation manually** (for expansion teams or full custom rosters):")
        with st.expander("Manual rotation builder"):
            n_players = st.number_input("Number of players", min_value=1, max_value=15, value=8, step=1)
            manual_rows = []
            for i in range(int(n_players)):
                mc1, mc2, mc3 = st.columns([3, 1, 1])
                pname = mc1.text_input(f"Player {i+1} name", key=f"manual_name_{i}")
                pmins = mc2.number_input(f"Minutes", min_value=0.0, max_value=40.0,
                                         value=20.0, step=0.5, key=f"manual_mins_{i}",
                                         label_visibility="collapsed")
                pid_in = mc3.number_input(f"ID (opt)", min_value=0, value=0, step=1,
                                          key=f"manual_pid_{i}", label_visibility="collapsed")
                if pname:
                    manual_rows.append({
                        "player_id": int(pid_in) if pid_in > 0 else np.nan,
                        "player_name": pname,
                        "projected_minutes": pmins,
                    })

            if manual_rows and st.button("Save manual rotation"):
                df = pd.DataFrame(manual_rows)
                roster_mod.set_rotation(team_sel, df)
                st.success(f"Saved {len(manual_rows)}-player rotation for {team_sel}")
                st.rerun()

    # ── CSV Upload ──
    with subtab3:
        st.markdown("Upload a CSV with projected minutes for a team.")
        st.code("Expected columns: player_name, minutes (+ optional player_id)")

        team_up = st.selectbox("Team", TEAMS, key="upload_team")
        st.markdown(team_badge(team_up), unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload rotation CSV", type=["csv"], key="rotation_upload")

        if uploaded is not None:
            try:
                df = roster_mod.parse_rotation_csv(uploaded.read(), team_up)
                st.markdown(f"**Preview — {len(df)} players for** &nbsp;{team_badge(team_up)}", unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True, hide_index=True)
                if st.button("Save uploaded rotation"):
                    roster_mod.set_rotation(team_up, df)
                    st.cache_data.clear()
                    st.success(f"Saved rotation for {team_up}")
                    st.rerun()
            except Exception as e:
                st.error(f"Parse error: {e}")


# ── Performance Tab ──────────────────────────────────────────────────────────

def tab_performance() -> None:
    st.header("Model vs. Market Performance")
    results = get_performance()
    if results.empty:
        st.info(
            "No BigDataBall data loaded.\n\n"
            "Drop CSVs named `2024.csv`, `2025.csv`, `2026.csv` into:\n\n"
            "`wnba_origination/data/bigdataball/`"
        )
        return

    stats = perf.summary_stats(results)
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Games", stats.get("n_games", "—"))
    s2.metric("Spread RMSE", stats.get("model_spread_rmse", "—"))
    s3.metric("Total RMSE", stats.get("model_total_rmse", "—"))
    s4.metric("Model ATS%", f"{stats.get('model_ats_pct', 0):.1f}%")
    s5.metric("Model O/U%", f"{stats.get('model_ou_pct', 0):.1f}%")
    s6.metric("Actual Over%", f"{stats.get('over_hit_pct', 0):.1f}%")

    st.divider()
    season_filter = st.selectbox(
        "Season",
        options=["All"] + sorted(results["season"].unique().tolist(), reverse=True),
    )
    if season_filter != "All":
        results = results[results["season"] == int(season_filter)]

    cols_to_show = [c for c in [
        "date", "teams", "actual_margin", "model_spread", "model_spread_err",
        "actual_total", "model_total", "model_total_err",
        "open_spread", "close_spread", "open_ou", "close_ou",
    ] if c in results.columns]
    st.dataframe(results[cols_to_show].sort_values("date", ascending=False),
                 use_container_width=True, hide_index=True)

    try:
        import plotly.express as px
        valid = results.dropna(subset=["model_spread", "actual_margin"])
        fig = px.scatter(
            valid, x="model_spread", y="actual_margin",
            hover_data=["date", "teams"],
            labels={"model_spread": "Model Spread (home)", "actual_margin": "Actual Margin (home)"},
            title="Model Spread vs. Actual Margin",
        )
        fig.add_shape(type="line", x0=-40, y0=-40, x1=40, y1=40,
                      line=dict(dash="dash", color="gray"))
        fig.add_hline(y=0, line_dash="dot", line_color="lightgray")
        fig.add_vline(x=0, line_dash="dot", line_color="lightgray")
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        pass


# ── Game Log helpers ─────────────────────────────────────────────────────────

# h/a = home/away column name for the stat; flip_off/flip_def = lower raw is better
METRIC_CONFIG: dict[str, dict] = {
    "eFG%":   {"h": "h_efg",    "a": "a_efg",    "flip_off": False, "flip_def": True,  "pct": True},
    "TOV%":   {"h": "h_tov",    "a": "a_tov",    "flip_off": True,  "flip_def": False, "pct": True},
    "OREB%":  {"h": "h_oreb",   "a": "a_oreb",   "flip_off": False, "flip_def": True,  "pct": True},
    "FTR":    {"h": "h_ftr",    "a": "a_ftr",    "flip_off": False, "flip_def": True,  "pct": True},
    "Pace":   {"h": "pace",     "a": "pace",     "flip_off": False, "flip_def": False, "pct": False},
    "Points": {"h": "home_pts", "a": "away_pts", "flip_off": False, "flip_def": True,  "pct": False},
    "Margin": {"h": "margin",   "a": None,       "flip_off": False, "flip_def": False, "pct": False},
}


def _team_season_chart(gl: pd.DataFrame, team: str, season: int, metric: str) -> None:
    """Render game-by-game percentile chart with opponent-coloured rolling average dots."""
    import plotly.graph_objects as go

    cfg = METRIC_CONFIG[metric]

    # Filter season, split matchup
    gls = gl[gl["season"] == season].copy()
    split = gls["matchup"].str.split(r"\s+vs\.?\s+", expand=True)
    gls["_home"] = split[0].str.strip()
    gls["_away"] = split[1].str.strip() if 1 in split.columns else ""

    records = []
    for _, row in gls.iterrows():
        is_home = row["_home"] == team
        is_away = row["_away"] == team
        if not is_home and not is_away:
            continue

        h_col, a_col = cfg["h"], cfg["a"]
        if is_home:
            off_val = row[h_col]
            def_val = row[a_col] if a_col and a_col != h_col else None
            opp = row["_away"]
        else:
            off_val = row[a_col] if a_col else None
            def_val = row[h_col] if a_col and a_col != h_col else None
            # margin: negate for away team
            if metric == "Margin" and off_val is not None:
                off_val = -off_val
            opp = row["_home"]

        records.append({"date": row["game_date"], "off_val": off_val,
                         "def_val": def_val, "opp": opp})

    if not records:
        st.info(f"No {season} games found for {team}.")
        return

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    df["game_num"] = df.index + 1

    # Percentile vs all season values for that metric
    all_vals = pd.concat([gls[cfg["h"]], gls[cfg["a"] or cfg["h"]]]).dropna().values

    def _pct(v, flip):
        if pd.isna(v):
            return np.nan
        p = float((all_vals < v).sum()) / len(all_vals)
        return 1.0 - p if flip else p

    df["off_pct"]  = df["off_val"].apply(lambda v: _pct(v, cfg["flip_off"]))
    df["def_pct"]  = df["def_val"].apply(lambda v: _pct(v, cfg["flip_def"])) if cfg["a"] and cfg["a"] != cfg["h"] else np.nan
    df["off_roll"] = df["off_pct"].rolling(5, min_periods=1).mean()
    df["def_roll"] = df["def_pct"].rolling(5, min_periods=1).mean() if isinstance(df["def_pct"], pd.Series) else None
    df["opp_col"]  = df["opp"].apply(_team_bg)
    df["opp_fg"]   = df["opp"].apply(_team_fg)

    tbg = _team_bg(team)
    fig = go.Figure()

    # Off percentile line
    fig.add_trace(go.Scatter(
        x=df["game_num"], y=df["off_pct"],
        mode="lines", name=f"Off {metric} %ile",
        line=dict(color=tbg, width=1.5),
        hovertemplate="G%{x} vs %{customdata}<br>Off: %{y:.2f}<extra></extra>",
        customdata=df["opp"],
    ))

    # Off rolling with opp-coloured dots
    fig.add_trace(go.Scatter(
        x=df["game_num"], y=df["off_roll"],
        mode="lines+markers", name="Rolling Off (5g)",
        line=dict(color=tbg, dash="dot", width=1),
        marker=dict(color=df["opp_col"].tolist(), size=11,
                    line=dict(color=df["opp_fg"].tolist(), width=1.5)),
        hovertemplate="G%{x} vs %{customdata}<br>Roll Off: %{y:.2f}<extra></extra>",
        customdata=df["opp"],
    ))

    # Def percentile line + rolling (if metric has separate def side)
    if df["def_pct"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["game_num"], y=df["def_pct"],
            mode="lines", name=f"Def {metric} %ile",
            line=dict(color="#e76f51", width=1.5),
            hovertemplate="G%{x} vs %{customdata}<br>Def: %{y:.2f}<extra></extra>",
            customdata=df["opp"],
        ))
        fig.add_trace(go.Scatter(
            x=df["game_num"], y=df["def_roll"],
            mode="lines+markers", name="Rolling Def (5g)",
            line=dict(color="#e76f51", dash="dot", width=1),
            marker=dict(color=df["opp_col"].tolist(), size=11,
                        line=dict(color=df["opp_fg"].tolist(), width=1.5)),
            hovertemplate="G%{x} vs %{customdata}<br>Roll Def: %{y:.2f}<extra></extra>",
            customdata=df["opp"],
        ))

    fig.update_layout(
        xaxis_title="Game #", yaxis_title="Percentile (higher = better)",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=40, r=20, t=40, b=40),
        height=360,
        plot_bgcolor="#fafafa",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Game Log Tab ─────────────────────────────────────────────────────────────

def tab_game_log() -> None:
    st.header("Game Log — Five Factors")
    st.caption("eFG%, TOV%, OREB%, FT Rate, Pace — computed from raw PBP for every game.")

    gl = get_game_log(_mtime=_game_log_mtime())
    if gl.empty:
        st.warning("No game log data. Run: python game_log.py")
        return

    # ── Filters ──
    fc1, fc2, fc3, fc4 = st.columns([1, 1, 1, 2])
    seasons = sorted(gl["season"].unique(), reverse=True)
    sel_season = fc1.selectbox("Season", seasons)

    view = gl[gl["season"] == sel_season].copy()

    # Split matchup "CON vs. ATL" into Home / Away columns
    split = view["matchup"].str.split(r"\s+vs\.?\s+", expand=True)
    view["home"] = split[0].str.strip()
    view["away"] = split[1].str.strip() if 1 in split.columns else ""

    # Team filter — pull unique teams from home/away columns
    all_teams = sorted(set(view["home"].dropna()) | set(view["away"].dropna()))
    sel_team = fc2.selectbox("Team", ["All"] + all_teams)
    if sel_team != "All":
        view = view[(view["home"] == sel_team) | (view["away"] == sel_team)]

    # Home/Away toggle
    perspective = fc3.radio("View as", ["Home", "Away", "Both"], horizontal=True)
    search = fc4.text_input("Search team")
    if search:
        view = view[
            view["home"].str.contains(search, case=False, na=False) |
            view["away"].str.contains(search, case=False, na=False)
        ]

    # Team badge label when filtered
    if sel_team != "All":
        st.markdown(team_badge(sel_team), unsafe_allow_html=True)
    st.caption(f"{len(view)} games")

    # Helper: scale fraction cols to whole-number percentages for display
    PCT_SCALE = ["h_efg", "h_tov", "h_oreb", "h_ftr",
                 "a_efg", "a_tov", "a_oreb", "a_ftr"]
    view = view.copy()
    for c in PCT_SCALE:
        if c in view.columns:
            view[c] = (view[c] * 100).round(1)

    if perspective == "Home":
        disp = view[["game_date", "home", "away", "home_pts", "away_pts", "margin", "pace",
                      "h_efg", "h_tov", "h_oreb", "h_ftr",
                      "h_fga", "h_tpa", "h_fta", "h_oreb_n", "h_tov_n"]].rename(columns={
            "home": "Home", "away": "Away",
            "home_pts": "Pts", "away_pts": "Opp",
            "h_efg": "eFG%", "h_tov": "TOV%", "h_oreb": "OREB%", "h_ftr": "FTR",
            "h_fga": "FGA", "h_tpa": "3PA", "h_fta": "FTA",
            "h_oreb_n": "OREB", "h_tov_n": "TOV",
        })
    elif perspective == "Away":
        disp = view[["game_date", "home", "away", "away_pts", "home_pts", "pace",
                      "a_efg", "a_tov", "a_oreb", "a_ftr",
                      "a_fga", "a_tpa", "a_fta", "a_oreb_n", "a_tov_n"]].rename(columns={
            "home": "Home", "away": "Away",
            "away_pts": "Pts", "home_pts": "Opp",
            "a_efg": "eFG%", "a_tov": "TOV%", "a_oreb": "OREB%", "a_ftr": "FTR",
            "a_fga": "FGA", "a_tpa": "3PA", "a_fta": "FTA",
            "a_oreb_n": "OREB", "a_tov_n": "TOV",
        })
        disp["margin"] = disp["Pts"] - disp["Opp"]
    else:
        disp = view[["game_date", "home", "away", "home_pts", "away_pts", "margin", "pace",
                      "h_efg", "h_tov", "h_oreb", "h_ftr",
                      "a_efg", "a_tov", "a_oreb", "a_ftr"]].rename(columns={
            "home": "Home", "away": "Away",
            "h_efg": "H eFG%", "h_tov": "H TOV%", "h_oreb": "H OREB%", "h_ftr": "H FTR",
            "a_efg": "A eFG%", "a_tov": "A TOV%", "a_oreb": "A OREB%", "a_ftr": "A FTR",
        })

    disp = disp.sort_values("game_date", ascending=False)
    import streamlit.components.v1 as _stc
    _stc.html(_html_table(disp), height=610, scrolling=False)

    # League averages — restored as table
    with st.expander("League averages (selected filter)"):
        raw = gl[gl["season"] == sel_season]  # unfiltered season for true league avg
        avg_df = pd.DataFrame({
            "Stat":     ["eFG%", "TOV%", "OREB%", "FT Rate", "Pace"],
            "Home avg": [
                f"{raw['h_efg'].mean()*100:.1f}%",
                f"{raw['h_tov'].mean()*100:.1f}%",
                f"{raw['h_oreb'].mean()*100:.1f}%",
                f"{raw['h_ftr'].mean()*100:.1f}%",
                f"{raw['pace'].mean():.1f}",
            ],
            "Away avg": [
                f"{raw['a_efg'].mean()*100:.1f}%",
                f"{raw['a_tov'].mean()*100:.1f}%",
                f"{raw['a_oreb'].mean()*100:.1f}%",
                f"{raw['a_ftr'].mean()*100:.1f}%",
                "—",
            ],
        })
        st.dataframe(avg_df, use_container_width=True, hide_index=True)

    # ── Team Season Chart ──
    st.divider()
    st.subheader("Team Season Chart")
    st.caption("Percentile vs. league each game. Rolling 5-game avg dots coloured by opponent.")

    cc1, cc2, cc3 = st.columns([1, 1, 1])
    chart_team   = cc1.selectbox("Team", TEAMS, key="gl_chart_team")
    chart_season = cc2.selectbox("Season", seasons, key="gl_chart_season")
    chart_metric = cc3.selectbox("Metric", list(METRIC_CONFIG.keys()), key="gl_chart_metric")

    _team_season_chart(gl, chart_team, int(chart_season), chart_metric)

    # ── Team Rotation History ──
    st.divider()
    st.subheader("Team Rotation History")
    st.caption("Minute-by-minute presence across last 5 games. Darker = on court.")

    import streamlit.components.v1 as _stc2
    rc1, rc2 = st.columns([1, 1])
    rot_team   = rc1.selectbox("Team", TEAMS, key="gl_rot_team")
    rot_season_opts = [s for s in seasons if s == int(chart_season) or True]
    rot_season = rc2.selectbox("Season", seasons, key="gl_rot_season")

    rot_store = get_store()
    rot_players = _players_for_chart(rot_team, rot_store)
    rot_games   = _recent_game_presence(rot_team, n=5, year=int(rot_season))

    if not rot_games:
        st.info(f"No stints data found for {rot_team} in {rot_season}.")
    else:
        st.markdown(team_badge(rot_team), unsafe_allow_html=True)
        grid_html, grid_h = _team_rotation_grid_html(rot_games, rot_players, rot_team)
        _stc2.html(grid_html, height=grid_h + 20, scrolling=True)

        # Box scores for each game
        st.markdown("**Box scores:**")
        for game in rot_games:
            gid   = game["game_id"]
            gdate = game["game_date"]
            opp   = game["opponent"]
            with st.expander(f"📋 {gdate} vs {opp}", expanded=False):
                box = _game_box_score(gid)
                if box.empty:
                    st.caption("No PBP data found for this game.")
                else:
                    # Split by team, show each side
                    team_ids = box["team_id"].unique()
                    for tid in sorted(team_ids):
                        side_box = (
                            box[box["team_id"] == tid]
                            .sort_values("pts", ascending=False)
                            [["name", "pts", "fg", "3p", "ft", "reb", "ast", "tov"]]
                            .rename(columns={"name": "Player", "pts": "PTS",
                                            "fg": "FG", "3p": "3P", "ft": "FT",
                                            "reb": "REB", "ast": "AST", "tov": "TOV"})
                        )
                        # Team badge header
                        # Find abbr by team_id
                        abbr = next((a for a, t in ABBR_TO_TEAM_ID.items() if t == tid), str(tid))
                        st.markdown(team_badge(abbr, "0.9rem"), unsafe_allow_html=True)
                        st.dataframe(side_box, use_container_width=True, hide_index=True,
                                     height=min(35 * len(side_box) + 38, 400))


# ── EC Historical Tab ────────────────────────────────────────────────────────

def tab_ec_historical(store: pd.DataFrame) -> None:
    st.header("EC Historical")
    st.caption("Season-level Estimated Contribution (positiveresidual.com) + RAPM / 2025 minutes where available.")

    ec = get_ec_historical(store)
    if ec.empty:
        st.warning(f"EC file not found. Expected: wnba_ec_all_seasons.csv on Desktop.")
        return

    # ── Filters ──
    fc1, fc2, fc3 = st.columns([1, 1, 2])
    seasons = sorted(ec["season"].unique(), reverse=True)
    sel_seasons = fc1.multiselect("Season", seasons, default=[seasons[0]] if seasons else [])
    teams = sorted(ec["team"].dropna().unique())
    sel_teams = fc2.multiselect("Team", teams)
    search = fc3.text_input("Player search")

    view = ec.copy()
    if sel_seasons:
        view = view[view["season"].isin(sel_seasons)]
    if sel_teams:
        view = view[view["team"].isin(sel_teams)]
    if search:
        view = view[view["player"].str.contains(search, case=False, na=False)]

    # ── Display cols ──
    base_cols = ["season", "player", "team", "mins", "oec", "dec", "ec", "war"]
    rapm_cols = ["orapm", "drapm", "net_rapm", "curr_mins_2025"]
    show_cols = base_cols + [c for c in rapm_cols if c in view.columns]
    view = view[show_cols].sort_values("ec", ascending=False)

    has_rapm = view["orapm"].notna().sum()
    st.caption(f"{len(view):,} player-seasons shown · {has_rapm:,} with RAPM data")

    st.dataframe(
        view.style.format({
            "oec": "{:.2f}", "dec": "{:.2f}", "ec": "{:.2f}", "war": "{:.2f}",
            "orapm": "{:+.2f}", "drapm": "{:+.2f}", "net_rapm": "{:+.2f}",
            "curr_mins_2025": "{:.0f}",
        }, na_rep="—"),
        use_container_width=True,
        hide_index=True,
        height=600,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("WNBA Line Origination")

    store = get_store()
    pace_cache = get_pace_cache()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Game", "Season", "Roster", "Performance", "Game Log", "EC Historical"])
    with tab1:
        tab_game(store, pace_cache)
    with tab2:
        tab_season()
    with tab3:
        tab_roster(store)
    with tab4:
        tab_performance()
    with tab5:
        tab_game_log()
    with tab6:
        tab_ec_historical(store)

    with st.sidebar:
        st.markdown("## Data")
        st.markdown(f"**Players:** {len(store)}")
        st.markdown(f"**Teams:** {store['team_abbr'].nunique()}")
        overrides = roster_mod.load_roster_overrides()
        st.markdown(f"**Active overrides:** {len(overrides)}")
        if st.button("Rebuild player store"):
            import player_store as ps
            ps.build()
            st.cache_data.clear()
            st.rerun()
        if st.button("Rebuild pace cache"):
            pace_module.build_pace_cache()
            st.cache_data.clear()
            st.rerun()
        if st.button("Run daily ingest"):
            import ingest
            with st.spinner("Running ingest..."):
                ingest.run(skip_ec=True)
            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    main()
