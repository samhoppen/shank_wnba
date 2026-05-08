"""
Python wrapper for the rotation chart Streamlit component.

In release mode (default), loads from the built static bundle.
Set ROTATION_CHART_DEV=1 to load from the local dev server (npm start on port 3001).
"""
import os
import streamlit.components.v1 as components
from pathlib import Path

_DEV = os.getenv("ROTATION_CHART_DEV", "0") == "1"
_FRONTEND = Path(__file__).parent / "frontend" / "build"

if _DEV:
    _func = components.declare_component("rotation_chart", url="http://localhost:3001")
else:
    _func = components.declare_component("rotation_chart", path=str(_FRONTEND))


def rotation_chart(
    players: list[dict],
    label: str = "",
    team_color: str = "#1e3a5f",
    text_color: str = "#FFFFFF",
    team_key: str = "default",
    forced_zeros: list[int] | None = None,
    key: str | None = None,
) -> dict[int, float]:
    """
    Render an interactive drag-and-drop rotation chart.

    Parameters
    ----------
    players : list of dicts with keys:
        player_id, player_name, default_minutes, orapm, drapm
    label : header label shown above chart
    team_color : hex background color for stint blocks (primary)
    text_color  : hex text/icon color inside stint blocks (secondary)
    team_key : changes this value to reset all stint positions (e.g. on team switch)
    key : Streamlit component key

    Returns
    -------
    dict mapping player_id -> total projected minutes (float)
    """
    result = _func(
        players=players,
        label=label,
        team_color=team_color,
        text_color=text_color,
        team_key=team_key,
        forced_zeros=forced_zeros or [],
        key=key,
        default={},
    )
    # Coerce keys to int (JSON round-trips them as strings)
    if result:
        return {int(k): float(v) for k, v in result.items()}
    return {}
