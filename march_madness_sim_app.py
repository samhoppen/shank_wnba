"""
NCAA March Madness Bracket Simulator
=====================================
Monte Carlo bracket simulation for the full 68-team NCAA tournament.

Bracket structure:
  First Four (4 play-in games)  →  4 regions × 16 seeds  →  Final Four  →  Championship

Win probability model:
  spread = (net_A - net_B) × possessions / 100
  P(A wins) = norm.cdf(spread / 11.0)
  possessions = round(((tempo_A + tempo_B) / 2) × 0.75 + NAT_TEMPO × 0.25)

Ratings source: team_ratings_cache.csv (adj_ortg, adj_drtg, tempo for all D1 teams)
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import io
import os
import sys
from collections import defaultdict

# ─── Constants ────────────────────────────────────────────────────────────────
NAT_TEMPO  = 69.1
SIGMA      = 11.0
DEFAULT_OE = 108.6
DEFAULT_DE = 108.6
REGIONS    = ["East", "West", "South", "Midwest"]

# Standard NCAA bracket pod order within each region.
# Adjacent pairs give the correct R64 matchups:
#   1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
BRACKET_SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

# Final Four pairing options: (Pair A, Pair B) where each pair is (region1, region2)
FF_PAIRING_OPTIONS = {
    "East vs West / South vs Midwest": [("East", "West"), ("South", "Midwest")],
    "East vs South / West vs Midwest": [("East", "South"), ("West", "Midwest")],
    "East vs Midwest / West vs South": [("East", "Midwest"), ("West", "South")],
}

# Round tracking keys → display info (key, col_header, tooltip)
ROUND_DISPLAY = [
    ("ff_adv", "FF%",    "Advance past First Four"),
    ("r32",    "R32%",   "Make Round of 32 (win R64)"),
    ("s16",    "S16%",   "Make Sweet 16 (win R32)"),
    ("e8",     "E8%",    "Make Elite 8 (win S16)"),
    ("f4",     "F4%",    "Make Final Four (win E8 / Regional Champ)"),
    ("final",  "Final%", "Make Championship game (win F4 semifinal)"),
    ("champ",  "Champ%", "Win Championship"),
]
ALL_ROUND_KEYS = [r[0] for r in ROUND_DISPLAY]

# ─── Resource helpers ─────────────────────────────────────────────────────────

def resource_path(rel):
    """Works both in dev and PyInstaller bundle."""
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, rel)


def find_cache():
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
    else:
        exe_dir = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(exe_dir, "team_ratings_cache.csv")
    if os.path.exists(local):
        return local
    bundled = resource_path("team_ratings_cache.csv")
    if os.path.exists(bundled):
        return bundled
    return r"C:\Users\shank.subramani_betf\Desktop\ShotsDashboard\team_ratings_cache.csv"


@st.cache_data(show_spinner="Loading ratings cache…")
def load_cache():
    path = find_cache()
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.warning(f"Could not load ratings cache: {e}")
        return pd.DataFrame(columns=["team", "adj_ortg", "adj_drtg", "adj_net", "tempo"])


# ─── Win probability ──────────────────────────────────────────────────────────

def win_prob_game(oe_a, de_a, tempo_a, oe_b, de_b, tempo_b) -> float:
    poss   = round(((tempo_a + tempo_b) / 2) * 0.75 + NAT_TEMPO * 0.25)
    spread = (oe_a - de_a - (oe_b - de_b)) * poss / 100
    return float(norm.cdf(spread / SIGMA))


def _sg(ta: dict, tb: dict, rng) -> dict:
    """Simulate one game between two team dicts; return winner."""
    p = win_prob_game(ta["oe"], ta["de"], ta["tempo"], tb["oe"], tb["de"], tb["tempo"])
    return ta if rng.random() < p else tb


# ─── Simulation engine ────────────────────────────────────────────────────────

def simulate_march_madness(region_teams: dict, ff_games: list, final_four_pairs: list, n_sims: int) -> dict:
    """
    Parameters
    ----------
    region_teams : {(region, seed): {team:str, oe:float, de:float, tempo:float}}
        All 64 non-FF seed slots. FF-seed slots are omitted; they'll be filled
        by FF game winners at simulation time.
    ff_games : list of dicts
        [{region:str, seed:int, is_bye:bool, team_a:dict, team_b:dict}]
        4 play-in slots. If is_bye=True or team_b is empty, team_a auto-advances.
    final_four_pairs : [("East","West"), ("South","Midwest")]
        Which regional champions face each other in the F4 semis.
    n_sims : int

    Returns
    -------
    {team_name: {round_key: float (0–1)}}
    """
    # Collect all unique team names for counting
    all_names: set = set()
    ff_eligible: set = set()

    for t in region_teams.values():
        if t.get("team"):
            all_names.add(t["team"])

    for g in ff_games:
        ta_nm = g["team_a"].get("team", "")
        tb_nm = g["team_b"].get("team", "") if g["team_b"] else ""
        if ta_nm:
            all_names.add(ta_nm)
        if not g["is_bye"] and tb_nm:
            all_names.add(tb_nm)
            ff_eligible.add(ta_nm)
            ff_eligible.add(tb_nm)

    counts  = {nm: defaultdict(int) for nm in all_names}
    ff_map  = {(g["region"], int(g["seed"])): g for g in ff_games}
    rng     = np.random.default_rng()

    # Pre-build per-region slot list in bracket order (saves repeated dict lookups)
    _empty = lambda r, s: {"team": f"__empty_{r}_{s}", "oe": DEFAULT_OE, "de": DEFAULT_DE, "tempo": NAT_TEMPO}

    for _ in range(n_sims):
        # ── Step 1: Resolve First Four ─────────────────────────────────────────
        sim_slots = dict(region_teams)

        for (region, seed), g in ff_map.items():
            ta, tb = g["team_a"], g["team_b"] or {}
            if g["is_bye"] or not tb.get("team"):
                winner = ta
            else:
                winner = _sg(ta, tb, rng)
                if winner["team"] in counts:
                    counts[winner["team"]]["ff_adv"] += 1
            sim_slots[(region, seed)] = winner

        # ── Step 2: Simulate each region (4 rounds) ───────────────────────────
        region_winners: dict = {}
        for region in REGIONS:
            bracket = [
                sim_slots.get((region, seed), _empty(region, seed))
                for seed in BRACKET_SEED_ORDER
            ]
            # Rounds: R64 winners "made R32", R32 winners "made S16", etc.
            for rnd_key in ("r32", "s16", "e8", "f4"):
                nxt = []
                for i in range(0, len(bracket), 2):
                    w = _sg(bracket[i], bracket[i + 1], rng)
                    if w["team"] in counts:
                        counts[w["team"]][rnd_key] += 1
                    nxt.append(w)
                bracket = nxt
            region_winners[region] = bracket[0]

        # ── Step 3: Final Four semis ───────────────────────────────────────────
        final_two = []
        for r1, r2 in final_four_pairs:
            w = _sg(region_winners[r1], region_winners[r2], rng)
            if w["team"] in counts:
                counts[w["team"]]["final"] += 1
            final_two.append(w)

        # ── Step 4: Championship ───────────────────────────────────────────────
        champ = _sg(final_two[0], final_two[1], rng)
        if champ["team"] in counts:
            counts[champ["team"]]["champ"] += 1

    return {
        nm: {rk: counts[nm][rk] / n_sims for rk in ALL_ROUND_KEYS}
        for nm in all_names
    }


# ─── Session-state helpers ────────────────────────────────────────────────────

def _default_ff_configs():
    return [
        {"region": "South",   "seed": 16, "is_bye": False, "team_a": "", "team_b": ""},
        {"region": "Midwest", "seed": 16, "is_bye": False, "team_a": "", "team_b": ""},
        {"region": "East",    "seed": 11, "is_bye": False, "team_a": "", "team_b": ""},
        {"region": "West",    "seed": 11, "is_bye": False, "team_a": "", "team_b": ""},
    ]


def _default_region_df():
    return pd.DataFrame({
        "Seed":  list(range(1, 17)),
        "Team":  [""] * 16,
        "OE":    [DEFAULT_OE] * 16,
        "DE":    [DEFAULT_DE] * 16,
        "Tempo": [NAT_TEMPO] * 16,
    })


def _init_state():
    if "ff_configs" not in st.session_state:
        st.session_state.ff_configs = _default_ff_configs()
    if "region_dfs" not in st.session_state:
        st.session_state.region_dfs = {r: _default_region_df() for r in REGIONS}
    if "sim_results" not in st.session_state:
        st.session_state.sim_results = None
    if "ff_pairing_key" not in st.session_state:
        st.session_state.ff_pairing_key = list(FF_PAIRING_OPTIONS.keys())[0]
    if "n_sims" not in st.session_state:
        st.session_state.n_sims = 10_000


# ─── Build simulation inputs ──────────────────────────────────────────────────

def build_sim_inputs(team_lkup: dict):
    """
    Read session state and return (region_teams, ff_games) ready for the sim engine.
    """
    ff_map = {
        (cfg["region"], cfg["seed"]): cfg
        for cfg in st.session_state.ff_configs
    }

    def ratings_for(name: str) -> dict:
        if name and name in team_lkup:
            r = team_lkup[name]
            return {"team": name, "oe": float(r["adj_ortg"]), "de": float(r["adj_drtg"]), "tempo": float(r["tempo"])}
        return {"team": name or "", "oe": DEFAULT_OE, "de": DEFAULT_DE, "tempo": NAT_TEMPO}

    region_teams = {}
    for region in REGIONS:
        df = st.session_state.region_dfs[region]
        for _, row in df.iterrows():
            seed = int(row["Seed"])
            if (region, seed) in ff_map:
                continue  # FF game handles this slot
            nm = str(row["Team"] or "").strip()
            if not nm:
                continue
            region_teams[(region, seed)] = {
                "team":  nm,
                "oe":    float(row["OE"]),
                "de":    float(row["DE"]),
                "tempo": float(row["Tempo"]),
            }

    ff_games = []
    for cfg in st.session_state.ff_configs:
        ff_games.append({
            "region":  cfg["region"],
            "seed":    cfg["seed"],
            "is_bye":  cfg["is_bye"],
            "team_a":  ratings_for(cfg["team_a"]),
            "team_b":  ratings_for(cfg["team_b"]),
        })

    return region_teams, ff_games


# ─── Build results table ──────────────────────────────────────────────────────

def build_results_df(sim_results: dict, region_teams: dict, ff_games: list) -> pd.DataFrame:
    # Build metadata lookup: team_name → {region, seed, oe, de, is_ff}
    meta: dict = {}
    for (region, seed), t in region_teams.items():
        if t["team"]:
            meta[t["team"]] = {
                "region": region, "seed": seed,
                "oe": t["oe"], "de": t["de"], "is_ff": False,
            }
    for g in ff_games:
        region, seed = g["region"], g["seed"]
        is_ff = not g["is_bye"] and bool(g["team_b"].get("team"))
        for t in (g["team_a"], g["team_b"] or {}):
            nm = t.get("team", "")
            if nm:
                meta[nm] = {
                    "region": region, "seed": seed,
                    "oe": t.get("oe", DEFAULT_OE), "de": t.get("de", DEFAULT_DE),
                    "is_ff": is_ff,
                }

    rows = []
    for nm, probs in sim_results.items():
        m = meta.get(nm, {})
        rows.append({
            "Team":   nm,
            "Region": m.get("region", "?"),
            "Seed":   m.get("seed", 0),
            "OE":     round(m.get("oe", DEFAULT_OE), 1),
            "DE":     round(m.get("de", DEFAULT_DE), 1),
            "Net":    round(m.get("oe", DEFAULT_OE) - m.get("de", DEFAULT_DE), 1),
            "is_ff":  m.get("is_ff", False),
            **{rk: probs[rk] for rk in ALL_ROUND_KEYS},
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        # Sort by Champ% desc, then seed asc
        df = df.sort_values(["champ", "seed"], ascending=[False, True]).reset_index(drop=True)
        df.index += 1
    return df


# ─── Formatting helpers ───────────────────────────────────────────────────────

def fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.1%}"


def fmt_pct_or_dash(x, show: bool) -> str:
    return fmt_pct(x) if show else "—"


# ─── UI ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="March Madness Simulator",
        page_icon="🏀",
        layout="wide",
    )

    _init_state()

    st.title("🏀 NCAA March Madness Bracket Simulator")
    st.caption(
        "68-team bracket · Possession-scaled win probability · σ = 11.0 pts · "
        "25% regression to national tempo · Ratings from team_ratings_cache.csv"
    )

    # ── Load ratings cache ──────────────────────────────────────────────────
    ratings   = load_cache()
    all_teams = sorted(ratings["team"].tolist()) if not ratings.empty else []
    team_lkup = {row["team"]: row for _, row in ratings.iterrows()}

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        n_sims = st.select_slider(
            "Simulations",
            options=[1_000, 5_000, 10_000, 25_000],
            value=st.session_state.n_sims,
        )
        st.session_state.n_sims = n_sims

        pairing_key = st.selectbox(
            "Final Four pairing",
            options=list(FF_PAIRING_OPTIONS.keys()),
            index=list(FF_PAIRING_OPTIONS.keys()).index(st.session_state.ff_pairing_key),
            help="Which region winners face each other in the Final Four semis.",
        )
        st.session_state.ff_pairing_key = pairing_key

        st.divider()

        # Team lookup
        st.subheader("🔍 Team lookup")
        search = st.selectbox("Search team", [""] + all_teams, key="sidebar_search")
        if search and search in team_lkup:
            r = team_lkup[search]
            c1, c2, c3 = st.columns(3)
            c1.metric("OE",    f"{r['adj_ortg']:.1f}")
            c2.metric("DE",    f"{r['adj_drtg']:.1f}")
            c3.metric("Net",   f"{r['adj_ortg'] - r['adj_drtg']:+.1f}")
            st.caption(f"Tempo: {r['tempo']:.1f} | {r.get('conference','')}")

        st.divider()

        if st.button("🔄 Auto-fill all regions from cache", use_container_width=True):
            filled = 0
            for region in REGIONS:
                df = st.session_state.region_dfs[region].copy()
                for i, row in df.iterrows():
                    nm = str(row["Team"] or "").strip()
                    if nm in team_lkup:
                        df.at[i, "OE"]    = round(float(team_lkup[nm]["adj_ortg"]), 1)
                        df.at[i, "DE"]    = round(float(team_lkup[nm]["adj_drtg"]), 1)
                        df.at[i, "Tempo"] = round(float(team_lkup[nm]["tempo"]), 1)
                        filled += 1
                st.session_state.region_dfs[region] = df
            st.session_state.sim_results = None
            st.success(f"Filled ratings for {filled} team(s).") if filled else st.warning("No team names matched the cache.")

        st.divider()

        run_btn = st.button("▶️ Run Simulation", type="primary", use_container_width=True)

        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.sim_results = None
            st.rerun()

        if st.button("♻️ Reset Bracket", use_container_width=True):
            st.session_state.ff_configs   = _default_ff_configs()
            st.session_state.region_dfs   = {r: _default_region_df() for r in REGIONS}
            st.session_state.sim_results  = None
            st.rerun()

        st.divider()
        st.caption(
            "**Tip:** Auto-fill populates OE/DE/Tempo from cache after entering team names. "
            "You can edit ratings manually for injury adjustments."
        )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tabs = st.tabs(["🎯 First Four", "🏟️ East", "🏟️ West", "🏟️ South", "🏟️ Midwest", "📊 Results"])

    # ╔═══════════════════════════════════════════════════════════╗
    # ║  FIRST FOUR TAB                                           ║
    # ╚═══════════════════════════════════════════════════════════╝
    with tabs[0]:
        st.subheader("🎯 First Four")
        st.caption(
            "Configure the 4 play-in games. Check **Bye** if a team has a direct entry "
            "(no play-in game needed — only Team A is used)."
        )

        for i in range(4):
            cfg = st.session_state.ff_configs[i]
            if cfg["is_bye"]:
                _slot_label = f"Slot {i+1} — {cfg['region']} #{cfg['seed']}  ✅ Bye"
            else:
                _a = cfg["team_a"] or "?"
                _b = cfg["team_b"] or "?"
                _slot_label = f"Slot {i+1} — {cfg['region']} #{cfg['seed']}  {_a} vs {_b}"
            with st.expander(_slot_label, expanded=True):
                col1, col2, col3 = st.columns([3, 2, 2])
                with col1:
                    new_region = st.selectbox(
                        "Region", REGIONS,
                        index=REGIONS.index(cfg["region"]),
                        key=f"ff_region_{i}",
                    )
                with col2:
                    new_seed = st.number_input(
                        "Seed", min_value=1, max_value=16, value=cfg["seed"],
                        step=1, key=f"ff_seed_{i}",
                    )
                with col3:
                    new_bye = st.checkbox("Bye (direct entry)", value=cfg["is_bye"], key=f"ff_bye_{i}")

                if new_bye:
                    c1, _ = st.columns(2)
                    with c1:
                        new_team_a = st.selectbox(
                            "Team (direct entry)", [""] + all_teams,
                            index=(all_teams.index(cfg["team_a"]) + 1) if cfg["team_a"] in all_teams else 0,
                            key=f"ff_team_a_{i}",
                        )
                    new_team_b = ""
                    if new_team_a and new_team_a in team_lkup:
                        r = team_lkup[new_team_a]
                        st.caption(f"OE {r['adj_ortg']:.1f} · DE {r['adj_drtg']:.1f} · Net {r['adj_ortg']-r['adj_drtg']:+.1f} · Tempo {r['tempo']:.1f}")
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        new_team_a = st.selectbox(
                            "Team A", [""] + all_teams,
                            index=(all_teams.index(cfg["team_a"]) + 1) if cfg["team_a"] in all_teams else 0,
                            key=f"ff_team_a_{i}",
                        )
                        if new_team_a and new_team_a in team_lkup:
                            r = team_lkup[new_team_a]
                            st.caption(f"OE {r['adj_ortg']:.1f} · DE {r['adj_drtg']:.1f} · Net {r['adj_ortg']-r['adj_drtg']:+.1f}")
                    with c2:
                        new_team_b = st.selectbox(
                            "Team B", [""] + all_teams,
                            index=(all_teams.index(cfg["team_b"]) + 1) if cfg["team_b"] in all_teams else 0,
                            key=f"ff_team_b_{i}",
                        )
                        if new_team_b and new_team_b in team_lkup:
                            r = team_lkup[new_team_b]
                            st.caption(f"OE {r['adj_ortg']:.1f} · DE {r['adj_drtg']:.1f} · Net {r['adj_ortg']-r['adj_drtg']:+.1f}")

                    if new_team_a and new_team_b and new_team_a in team_lkup and new_team_b in team_lkup:
                        ta, tb = team_lkup[new_team_a], team_lkup[new_team_b]
                        p = win_prob_game(
                            float(ta["adj_ortg"]), float(ta["adj_drtg"]), float(ta["tempo"]),
                            float(tb["adj_ortg"]), float(tb["adj_drtg"]), float(tb["tempo"]),
                        )
                        st.info(f"P({new_team_a} wins FF game) = **{p:.1%}**   |   P({new_team_b}) = {1-p:.1%}")

                # Update session state
                st.session_state.ff_configs[i] = {
                    "region":  new_region,
                    "seed":    int(new_seed),
                    "is_bye":  new_bye,
                    "team_a":  new_team_a,
                    "team_b":  new_team_b,
                }

    # ╔═══════════════════════════════════════════════════════════╗
    # ║  REGION TABS                                              ║
    # ╚═══════════════════════════════════════════════════════════╝
    for tab_idx, region in enumerate(REGIONS):
        with tabs[tab_idx + 1]:
            st.subheader(f"🏟️ {region} Region")

            # Identify which seeds in this region are First Four slots
            ff_seeds = {
                cfg["seed"]
                for cfg in st.session_state.ff_configs
                if cfg["region"] == region
            }

            if ff_seeds:
                ff_info = []
                for cfg in st.session_state.ff_configs:
                    if cfg["region"] == region:
                        if cfg["is_bye"]:
                            ff_info.append(f"Seed {cfg['seed']}: Bye → {cfg['team_a'] or '(unnamed)'}")
                        else:
                            a = cfg["team_a"] or "(TBD)"
                            b = cfg["team_b"] or "(TBD)"
                            ff_info.append(f"Seed {cfg['seed']}: First Four → {a} vs {b}")
                st.info("**First Four slots** (set in First Four tab):\n\n" + "\n\n".join(ff_info))

            st.caption(
                "Enter team names — OE/DE/Tempo auto-fill when you hit **Auto-fill** in the sidebar. "
                + (f"Seeds {sorted(ff_seeds)} are First Four games; entries in those rows are ignored in simulation." if ff_seeds else "")
            )

            prev_df = st.session_state.region_dfs[region]

            # Mark FF rows with a note
            display_df = prev_df.copy()
            for seed in ff_seeds:
                mask = display_df["Seed"] == seed
                cfg = next((c for c in st.session_state.ff_configs if c["region"] == region and c["seed"] == seed), None)
                if cfg:
                    if cfg["is_bye"]:
                        display_df.loc[mask, "Team"] = f"→ FF Bye: {cfg['team_a'] or 'TBD'}"
                    else:
                        a = cfg["team_a"] or "TBD"
                        b = cfg["team_b"] or "TBD"
                        display_df.loc[mask, "Team"] = f"→ FF: {a} vs {b}"

            edited_df = st.data_editor(
                display_df,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "Seed":  st.column_config.NumberColumn("Seed",  min_value=1, max_value=16, step=1, width="small", disabled=True),
                    "Team":  st.column_config.SelectboxColumn("Team", options=[""] + all_teams, width="large"),
                    "OE":    st.column_config.NumberColumn("Adj OE",    format="%.1f", width="small", min_value=80.0, max_value=160.0),
                    "DE":    st.column_config.NumberColumn("Adj DE",    format="%.1f", width="small", min_value=80.0, max_value=160.0),
                    "Tempo": st.column_config.NumberColumn("Tempo",     format="%.1f", width="small", min_value=50.0, max_value=90.0),
                },
                key=f"editor_{region}",
            )

            # Auto-fill OE/DE/Tempo when a team name changes
            needs_rerun = False
            for idx in edited_df.index:
                seed_val = int(edited_df.at[idx, "Seed"])
                if seed_val in ff_seeds:
                    # Don't overwrite FF row back with the display text
                    edited_df.at[idx, "Team"]  = prev_df.at[idx, "Team"]
                    edited_df.at[idx, "OE"]    = prev_df.at[idx, "OE"]
                    edited_df.at[idx, "DE"]    = prev_df.at[idx, "DE"]
                    edited_df.at[idx, "Tempo"] = prev_df.at[idx, "Tempo"]
                    continue
                new_nm = str(edited_df.at[idx, "Team"] or "").strip()
                old_nm = str(prev_df.at[idx, "Team"] or "").strip()
                if new_nm != old_nm and new_nm in team_lkup:
                    r = team_lkup[new_nm]
                    edited_df.at[idx, "OE"]    = round(float(r["adj_ortg"]), 1)
                    edited_df.at[idx, "DE"]    = round(float(r["adj_drtg"]), 1)
                    edited_df.at[idx, "Tempo"] = round(float(r["tempo"]),    1)
                    needs_rerun = True

            st.session_state.region_dfs[region] = edited_df
            if needs_rerun:
                st.rerun()

            # Region summary
            filled = sum(
                1 for _, row in edited_df.iterrows()
                if str(row["Team"] or "").strip() and int(row["Seed"]) not in ff_seeds
            )
            non_ff_count = 16 - len(ff_seeds)
            st.caption(f"Filled: {filled}/{non_ff_count} non-FF seeds")

    # ╔═══════════════════════════════════════════════════════════╗
    # ║  SIMULATION TRIGGER                                       ║
    # ╚═══════════════════════════════════════════════════════════╝
    if run_btn:
        region_teams, ff_games = build_sim_inputs(team_lkup)
        final_four_pairs = FF_PAIRING_OPTIONS[st.session_state.ff_pairing_key]

        # Validation
        total_teams = len(region_teams)
        ff_team_count = sum(
            (1 if g["team_a"].get("team") else 0) + (0 if g["is_bye"] else (1 if g["team_b"].get("team") else 0))
            for g in ff_games
        )
        if total_teams + ff_team_count < 4:
            st.error("Please enter at least 4 teams before simulating.")
        else:
            empty_slots = 64 - total_teams
            if empty_slots > 0:
                st.warning(f"{empty_slots} non-FF seed(s) are empty — those slots will use average ratings (OE/DE = {DEFAULT_OE}). Fill them for accurate results.")

            with st.spinner(f"Running {n_sims:,} simulations…"):
                results = simulate_march_madness(region_teams, ff_games, final_four_pairs, n_sims)

            st.session_state.sim_results = {
                "results":      results,
                "region_teams": region_teams,
                "ff_games":     ff_games,
                "n_sims":       n_sims,
                "pairing":      st.session_state.ff_pairing_key,
            }
            st.rerun()

    # ╔═══════════════════════════════════════════════════════════╗
    # ║  RESULTS TAB                                              ║
    # ╚═══════════════════════════════════════════════════════════╝
    with tabs[5]:
        if not st.session_state.sim_results:
            st.info("Fill in the bracket and click **▶️ Run Simulation** in the sidebar to see results.")
        else:
            res          = st.session_state.sim_results
            results      = res["results"]
            region_teams = res["region_teams"]
            ff_games     = res["ff_games"]
            n_sims_ran   = res["n_sims"]
            pairing_used = res["pairing"]

            st.subheader(f"📊 Results — {n_sims_ran:,} simulations")
            st.caption(f"Final Four pairing: {pairing_used}")

            result_df = build_results_df(results, region_teams, ff_games)

            if result_df.empty:
                st.warning("No results to display.")
            else:
                # Determine which teams are FF-eligible
                ff_eligible = {
                    nm for g in ff_games if not g["is_bye"]
                    for nm in [g["team_a"].get("team",""), g["team_b"].get("team","")]
                    if nm
                }

                # Build display DataFrame
                disp = result_df[["Team", "Region", "Seed", "OE", "DE", "Net"]].copy()
                disp["OE"]  = disp["OE"].map(lambda x: f"{x:.1f}")
                disp["DE"]  = disp["DE"].map(lambda x: f"{x:.1f}")
                disp["Net"] = disp["Net"].map(lambda x: f"{x:+.1f}")

                for key, col_label, _ in ROUND_DISPLAY:
                    if key == "ff_adv":
                        disp[col_label] = result_df.apply(
                            lambda row: fmt_pct(row[key]) if row["Team"] in ff_eligible else "—",
                            axis=1,
                        )
                    else:
                        disp[col_label] = result_df[key].map(fmt_pct)

                st.dataframe(disp.drop(columns=["is_ff"], errors="ignore"), use_container_width=True, hide_index=False)

                # ── Championship odds summary ──────────────────────────────────
                st.divider()
                st.subheader("🏆 Championship odds (top 20)")
                champ_top = result_df.head(20)[["Team", "Region", "Seed", "Net", "f4", "final", "champ"]].copy()
                champ_top["Net"]    = champ_top["Net"].map(lambda x: f"{x:+.1f}")
                champ_top.rename(columns={"f4": "F4%", "final": "Final%", "champ": "Champ%"}, inplace=True)

                st.dataframe(
                    champ_top,
                    use_container_width=True,
                    hide_index=False,
                    column_config={
                        "Champ%": st.column_config.ProgressColumn("Champ%", format="%.1%", min_value=0, max_value=1),
                        "Final%": st.column_config.ProgressColumn("Final%", format="%.1%", min_value=0, max_value=1),
                        "F4%":    st.column_config.ProgressColumn("F4%",    format="%.1%", min_value=0, max_value=1),
                    },
                )

                # ── Probability sanity checks ──────────────────────────────────
                st.divider()
                with st.expander("🔬 Probability sanity checks"):
                    champ_sum = result_df["champ"].sum()
                    f4_sum    = result_df["f4"].sum()
                    final_sum = result_df["final"].sum()
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Sum of Champ%",  f"{champ_sum:.3f}", help="Should be ≈ 1.000")
                    col2.metric("Sum of Final%",  f"{final_sum:.3f}", help="Should be ≈ 2.000 (2 teams in final)")
                    col3.metric("Sum of F4%",     f"{f4_sum:.3f}",    help="Should be ≈ 4.000 (4 F4 teams)")

                # ── Export ─────────────────────────────────────────────────────
                st.divider()
                st.subheader("📥 Export")

                export_rows = []
                for _, row in result_df.iterrows():
                    ff_show = row["Team"] in ff_eligible
                    export_rows.append([
                        row["Region"],
                        row["Seed"],
                        row["Team"],
                        f"{row['OE']:.1f}",
                        f"{row['DE']:.1f}",
                        f"{row['Net']:+.1f}",
                        fmt_pct(row["ff_adv"]) if ff_show else "—",
                        fmt_pct(row["r32"]),
                        fmt_pct(row["s16"]),
                        fmt_pct(row["e8"]),
                        fmt_pct(row["f4"]),
                        fmt_pct(row["final"]),
                        fmt_pct(row["champ"]),
                    ])

                export_df = pd.DataFrame(
                    export_rows,
                    columns=["Region", "Seed", "Team", "OE", "DE", "Net",
                             "FF%", "R32%", "S16%", "E8%", "F4%", "Final%", "Champ%"],
                )

                buf = io.StringIO()
                export_df.to_csv(buf, index=False)

                col_dl, col_tsv = st.columns(2)
                with col_dl:
                    st.download_button(
                        "⬇️ Download CSV",
                        data=buf.getvalue(),
                        file_name="march_madness_sim.csv",
                        mime="text/csv",
                        use_container_width=True,
                        type="primary",
                    )
                with col_tsv:
                    tsv_data = export_df.to_csv(index=False, sep="\t")
                    st.download_button(
                        "📋 Tab-separated (paste to spreadsheet)",
                        data=tsv_data,
                        file_name="march_madness_sim.tsv",
                        mime="text/plain",
                        use_container_width=True,
                    )


if __name__ == "__main__":
    main()
