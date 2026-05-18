"""
League Dashboard tab — live league baselines and team profiles, driven by
the analysis CSVs synced from WNBA_RAPM/analysis.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import league_stats as ls


# ── Caching wrappers (so the page doesn't re-read CSVs on every interaction) ──

@st.cache_data(ttl=600)
def _baselines(season: str) -> dict:
    return ls.league_baselines(season)


@st.cache_data(ttl=600)
def _last_n_baselines(n: int) -> dict:
    return ls.last_n_baselines(n)


@st.cache_data(ttl=600)
def _team_profiles(season: str) -> pd.DataFrame:
    return ls.team_profiles(season)


@st.cache_data(ttl=600)
def _recent_games(n: int, season: str) -> pd.DataFrame:
    return ls.recent_games(n, season)


@st.cache_data(ttl=600)
def _bonus_summary(season: str) -> pd.DataFrame:
    return ls.bonus_summary(season)


@st.cache_data(ttl=600)
def _foul_rates(season: str) -> pd.DataFrame:
    return ls.foul_rates(season)


# ── Helpers ────────────────────────────────────────────────────────────────


def _team_badge(abbr: str) -> str:
    """Inline team badge (importable to avoid circular import with app.py)."""
    try:
        from app import team_badge as _tb  # type: ignore
        return _tb(abbr, size="0.85rem")
    except Exception:
        return f'<span style="font-weight:700">{abbr}</span>'


def _fmt_delta(a: float, b: float, suffix: str = "", precision: int = 1) -> str:
    if pd.isna(a) or pd.isna(b):
        return "—"
    d = b - a
    arrow = "▲" if d > 0 else ("▼" if d < 0 else "—")
    color = "#22c55e" if d > 0 else ("#ef4444" if d < 0 else "#6b7280")
    return f'<span style="color:{color}">{arrow} {abs(d):.{precision}f}{suffix}</span>'


# ── Sub-views ─────────────────────────────────────────────────────────────


def _view_season_context() -> None:
    st.subheader("League baselines")

    bl25 = _baselines("2025")
    bl26 = _baselines("2026")
    bl_last30 = _last_n_baselines(30)
    if not bl25 or not bl26:
        st.warning("Baseline data not yet available — hit Refresh in the sidebar.")
        return

    # (metric_label, key, unit, is_count_only)
    SPEC = [
        ("Games (team-games)", "n_team_games", "",  True),
        ("Pace (poss/team-game)", "pace",      "",  False),
        ("ORTG (pts/100 poss)", "ortg",        "",  False),
        ("PTS / team-game",     "pts",         "",  False),
        ("FGA / team-game",     "fga",         "",  False),
        ("FTA / team-game",     "fta",         "",  False),
        ("TOV / team-game",     "tov",         "",  False),
        ("OREB / team-game",    "oreb",        "",  False),
        ("FT/FGA",              "ft_per_fga",  "",  False),
        ("Transition %",        "trans_pct",   "%", False),
        ("Bonus-quarter %",     "bonus_q_pct", "%", False),
    ]

    def fmt_val(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        if isinstance(v, int):
            return f"{v:,}"
        return f"{v:.2f}"

    rows = []
    for label, key, unit, count_only in SPEC:
        v25 = bl25.get(key)
        v26 = bl26.get(key)
        v_last = bl_last30.get(key)
        delta_html = "" if count_only else _fmt_delta(v25, v26, unit)
        rows.append({
            "Metric":    label,
            "2025":      fmt_val(v25),
            "2026":      fmt_val(v26),
            "Last 30":   fmt_val(v_last),
            "Δ '25→'26": delta_html,
        })
    df = pd.DataFrame(rows)
    st.markdown(
        df.to_html(escape=False, index=False, classes="dataframe", border=0),
        unsafe_allow_html=True,
    )
    st.caption(
        f"Pace = counted possessions from PBP walker. "
        f"Last-30 = most recent {bl_last30.get('n_games', 0)} games (covers ~{bl_last30.get('n_team_games', 0)} team-games)."
    )


def _view_game_log() -> None:
    st.subheader("Recent games")
    col1, col2 = st.columns([1, 3])
    with col1:
        season = st.selectbox("Season", ["2026", "2025"], key="ld_gl_season")
        n = st.slider("How many", 5, 100, 25, key="ld_gl_n")

    df = _recent_games(n, season)
    if df.empty:
        st.info("No games available for this season yet.")
        return

    # Display table
    show = df.copy()
    show["Date"] = pd.to_datetime(show["date"]).dt.strftime("%-m/%d") if False else \
        pd.to_datetime(show["date"]).dt.strftime("%m/%d").str.lstrip("0").str.replace("/0", "/")
    show["Matchup"] = show["matchup"]
    show["Total"] = show["total"]
    show["Pace"] = show["pace"]
    show["FTA A/B"] = show["fta_a"].astype(str) + " / " + show["fta_b"].astype(str)

    st.dataframe(
        show[["Date", "Matchup", "Total", "Pace", "FTA A/B"]],
        use_container_width=True,
        hide_index=True,
        height=min(40 + 35 * len(show), 600),
    )

    st.caption(f"Median total: {df['total'].median():.0f}   Mean: {df['total'].mean():.1f}   "
               f"Median pace: {df['pace'].median():.1f}")


def _view_team_profiles() -> None:
    st.subheader("Team profiles")
    season = st.selectbox("Season", ["2026", "2025"], key="ld_tp_season")
    tp = _team_profiles(season)
    if tp.empty:
        st.info("No team profiles for this season yet.")
        return

    # Style: render team as badge + numeric cols
    rows_html = ['<table style="width:100%;border-collapse:collapse">']
    rows_html.append(
        '<thead><tr>'
        + ''.join(
            f'<th style="text-align:left;padding:6px 10px;font-size:12px;'
            f'color:#555;border-bottom:1px solid #ddd">{c}</th>'
            for c in ["Team", "G", "Pace", "ORTG", "DRTG", "Net",
                       "PTS", "FGA", "FTA", "TOV", "OREB", "FT/FGA"]
        )
        + '</tr></thead><tbody>'
    )
    for _, r in tp.iterrows():
        cells = [
            _team_badge(r["team"]),
            f'{int(r["n"])}',
            f'{r["pace"]:.1f}',
            f'{r["ortg"]:.1f}',
            f'{r["drtg"]:.1f}' if pd.notna(r["drtg"]) else "—",
            f'{r["net"]:+.1f}' if pd.notna(r["net"]) else "—",
            f'{r["pts"]:.1f}',
            f'{r["fga"]:.1f}',
            f'{r["fta"]:.1f}',
            f'{r["tov"]:.1f}',
            f'{r["oreb"]:.1f}',
            f'{r["fta_per_fga"]:.3f}',
        ]
        rows_html.append(
            '<tr>'
            + ''.join(
                f'<td style="padding:5px 10px;font-size:12px;'
                f'border-bottom:1px solid #f0f0f0">{c}</td>'
                for c in cells
            )
            + '</tr>'
        )
    rows_html.append('</tbody></table>')
    st.markdown('\n'.join(rows_html), unsafe_allow_html=True)
    st.caption(f"Per-team-game means. {len(tp)} teams in {season}.")


def _view_bonus_fts() -> None:
    st.subheader("Bonus reaching by quarter")
    bs25 = _bonus_summary("2025")
    bs26 = _bonus_summary("2026")

    # Combine for comparison
    if not bs25.empty and not bs26.empty:
        merged = bs25[["period", "pct_reached"]].merge(
            bs26[["period", "pct_reached"]],
            on="period", suffixes=("_2025", "_2026"),
        )
        merged["Δpp"] = (merged["pct_reached_2026"] - merged["pct_reached_2025"]).round(1)
        merged["Quarter"] = "Q" + merged["period"].astype(int).astype(str)
        merged = merged[["Quarter", "pct_reached_2025", "pct_reached_2026", "Δpp"]]
        merged.columns = ["Quarter", "2025 %", "2026 %", "Δpp"]
        st.dataframe(
            merged, use_container_width=True, hide_index=True,
        )
        st.caption("% of team-quarters where the team reached the 5-foul bonus threshold.")
    else:
        st.info("Bonus data not yet available.")

    st.divider()
    st.subheader("Foul & violation rates per 100 possessions")
    fr25 = _foul_rates("2025")
    fr26 = _foul_rates("2026")
    if not fr25.empty and not fr26.empty:
        cols = ["category", "sub_type"]
        m25 = fr25.set_index(cols)[["total", "per_100_poss"]].rename(
            columns={"total": "N 2025", "per_100_poss": "2025 /100"}
        )
        m26 = fr26.set_index(cols)[["total", "per_100_poss"]].rename(
            columns={"total": "N 2026", "per_100_poss": "2026 /100"}
        )
        merged = pd.concat([m25, m26], axis=1).reset_index()
        merged["Δ /100"] = (merged["2026 /100"] - merged["2025 /100"]).round(2)
        merged["Δ %"] = (merged["Δ /100"] / merged["2025 /100"] * 100).round(1)
        merged = merged.sort_values("Δ /100", ascending=False, na_position="last")
        st.dataframe(
            merged[["category", "sub_type", "N 2025", "2025 /100",
                    "N 2026", "2026 /100", "Δ /100", "Δ %"]].round(2),
            use_container_width=True, hide_index=True, height=400,
        )


# ── Main render ───────────────────────────────────────────────────────────


def render() -> None:
    st.markdown("### League Dashboard")
    st.caption("Live league baselines and team profiles. Hit **Refresh data** in the sidebar to pull the latest games.")

    sub1, sub2, sub3, sub4 = st.tabs(["Season Context", "Game Log", "Team Profiles", "Bonus & FTs"])
    with sub1:
        _view_season_context()
    with sub2:
        _view_game_log()
    with sub3:
        _view_team_profiles()
    with sub4:
        _view_bonus_fts()
