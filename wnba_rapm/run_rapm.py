"""WNBA RAPM + 4-Factor RAPM — script version of rapm_reproducible.ipynb.

Loads possession-level stints, fits ridge-regression RAPM (offense/defense),
computes per-player four-factor on-court rates, fits an OLS decomposition,
prints the top-25 by net RAPM, and writes the combined output CSV.

Usage:
    python run_rapm.py
    python run_rapm.py --seasons 2023 2024 2025 --lambda-ridge 2000 --min-poss 200
    python run_rapm.py --data-dir wnba_data --out wnba_data/rapm_and_4f_output.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression, Ridge


OFF_COLS = [f"off_p{i}" for i in range(1, 6)]
DEF_COLS = [f"def_p{i}" for i in range(1, 6)]
FACTOR_COLS = ["ots", "otov", "oreb", "otrans", "dts", "dtov", "dreb", "dtrans"]


def load_stints(data_dir: Path, seasons: list[int], rich: bool) -> pd.DataFrame:
    subdir = "stints_rich" if rich else "stints"
    prefix = "stints_rich" if rich else "stints"
    frames = []
    for yr in seasons:
        path = data_dir / subdir / f"{prefix}_{yr}_RS.csv"
        if not path.exists():
            print(f"  missing: {path}")
            continue
        df = pd.read_csv(path)
        df["season"] = yr
        frames.append(df)
        print(f"  loaded {yr}: {len(df):,} possessions")
    if not frames:
        raise FileNotFoundError(f"No stints files found in {data_dir / subdir}")
    return pd.concat(frames, ignore_index=True)


def fit_rapm(df: pd.DataFrame, lam: float) -> pd.DataFrame:
    all_ids = sorted(
        set(df[OFF_COLS].values.flatten().tolist())
        | set(df[DEF_COLS].values.flatten().tolist())
    )
    all_ids = [int(p) for p in all_ids if not np.isnan(p)]
    pid_to_col = {pid: i for i, pid in enumerate(all_ids)}
    n_players = len(all_ids)
    n_poss = len(df)

    off_r, off_c, def_r, def_c = [], [], [], []
    for row_idx, row in enumerate(df.itertuples(index=False)):
        for col in OFF_COLS:
            pid = int(getattr(row, col))
            if pid in pid_to_col:
                off_r.append(row_idx)
                off_c.append(pid_to_col[pid])
        for col in DEF_COLS:
            pid = int(getattr(row, col))
            if pid in pid_to_col:
                def_r.append(row_idx)
                def_c.append(pid_to_col[pid])

    X_off = csr_matrix(([1.0] * len(off_r), (off_r, off_c)), shape=(n_poss, n_players))
    X_def = csr_matrix(([1.0] * len(def_r), (def_r, def_c)), shape=(n_poss, n_players))
    y = df["points"].values.astype(float)

    ridge = Ridge(alpha=lam, fit_intercept=True)
    ridge.fit(X_off, y)
    orapm = ridge.coef_ * 100

    ridge.fit(X_def, y)
    drapm = -ridge.coef_ * 100

    poss_map: dict[int, int] = {}
    for col in OFF_COLS + DEF_COLS:
        for pid, cnt in df[col].value_counts().items():
            pid = int(pid)
            poss_map[pid] = poss_map.get(pid, 0) + int(cnt)

    return pd.DataFrame({
        "player_id": all_ids,
        "orapm": np.round(orapm, 3),
        "drapm": np.round(drapm, 3),
        "net_rapm": np.round(orapm + drapm, 3),
        "poss": [poss_map.get(pid, 0) for pid in all_ids],
    })


def compute_factor_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pts"] = df["fgm"] * 2 + df["fg3m"] + df["ftm"]
    df["shot_att"] = df["fga"] * 2 + df["fg3a"] + df["fta"] * 0.44
    df["poss_denom"] = df["fga"] + df["fta"] * 0.44 + df["tov_flag"]

    aggs: dict[str, pd.DataFrame] = {}
    for side, cols in [("off", OFF_COLS), ("def", DEF_COLS)]:
        melted = (
            df[[
                "pts", "shot_att", "poss_denom", "tov_flag",
                "oreb", "oreb_chance", "trans_flag",
            ] + cols]
            .melt(
                id_vars=[
                    "pts", "shot_att", "poss_denom", "tov_flag",
                    "oreb", "oreb_chance", "trans_flag",
                ],
                value_vars=cols,
                value_name="player_id",
            )
            .dropna(subset=["player_id"])
        )
        melted["player_id"] = melted["player_id"].astype(int)

        g = (
            melted.groupby("player_id")
            .agg(
                poss=("pts", "count"),
                sum_pts=("pts", "sum"),
                sum_shot=("shot_att", "sum"),
                sum_poss_d=("poss_denom", "sum"),
                sum_tov=("tov_flag", "sum"),
                sum_oreb=("oreb", "sum"),
                sum_oreb_c=("oreb_chance", "sum"),
                sum_trans=("trans_flag", "sum"),
            )
            .reset_index()
        )
        prefix = "o" if side == "off" else "d"
        g[f"raw_{prefix}ts"] = g["sum_pts"] / g["sum_shot"].clip(lower=1e-9)
        g[f"raw_{prefix}tov"] = g["sum_tov"] / g["sum_poss_d"].clip(lower=1e-9)
        g[f"raw_{prefix}reb"] = g["sum_oreb"] / g["sum_oreb_c"].clip(lower=1e-9)
        g[f"raw_{prefix}trans"] = g["sum_trans"] / g["poss"].clip(lower=1e-9)
        g = g.rename(columns={"poss": f"poss_{side}"})
        aggs[side] = g[[
            "player_id", f"poss_{side}",
            f"raw_{prefix}ts", f"raw_{prefix}tov",
            f"raw_{prefix}reb", f"raw_{prefix}trans",
        ]]

    return aggs["off"].merge(aggs["def"], on="player_id", how="outer")


def league_averages(stints_rich: pd.DataFrame) -> dict[str, float]:
    avg = {
        "ots": (stints_rich["fgm"] * 2 + stints_rich["fg3m"] + stints_rich["ftm"]).sum()
        / (stints_rich["fga"] * 2 + stints_rich["fg3a"] + stints_rich["fta"] * 0.44).clip(lower=1e-9).sum(),
        "otov": stints_rich["tov_flag"].sum()
        / (stints_rich["fga"] + stints_rich["fta"] * 0.44 + stints_rich["tov_flag"]).clip(lower=1e-9).sum(),
        "oreb": stints_rich["oreb"].sum() / stints_rich["oreb_chance"].clip(lower=1e-9).sum(),
        "otrans": stints_rich["trans_flag"].sum() / len(stints_rich),
    }
    avg.update({"dts": avg["ots"], "dtov": avg["otov"], "dreb": avg["oreb"], "dtrans": avg["otrans"]})
    return avg


def demean_factors(ff: pd.DataFrame, avg: dict[str, float]) -> pd.DataFrame:
    ff = ff.copy()
    ff["ots"] = ff["raw_ots"] - avg["ots"]
    ff["otov"] = ff["raw_otov"] - avg["otov"]
    ff["oreb"] = ff["raw_oreb"] - avg["oreb"]
    ff["otrans"] = ff["raw_otrans"] - avg["otrans"]
    # Sign-flipped so positive = good defense.
    ff["dts"] = -(ff["raw_dts"] - avg["dts"])
    ff["dtov"] = ff["raw_dtov"] - avg["dtov"]
    ff["dreb"] = -(ff["raw_dreb"] - avg["dreb"])
    ff["dtrans"] = -(ff["raw_dtrans"] - avg["dtrans"])
    return ff


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WNBA RAPM + 4-Factor RAPM")
    p.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "wnba_data")
    p.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025])
    p.add_argument("--lambda-ridge", type=float, default=2000.0, dest="lam")
    p.add_argument("--min-poss", type=int, default=200)
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV path (default: <data-dir>/rapm_and_4f_output.csv)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir
    out_path: Path = args.out or (data_dir / "rapm_and_4f_output.csv")

    print(f"DATA_DIR   : {data_dir.resolve()}")
    print(f"RAPM years : {args.seasons}")
    print(f"LAMBDA     : {args.lam:,}")
    print(f"MIN_POSS   : {args.min_poss}")

    print("\nLoading standard stints...")
    stints = load_stints(data_dir, args.seasons, rich=False)
    print("\nLoading stints_rich...")
    stints_rich = load_stints(data_dir, args.seasons, rich=True)

    names_path = data_dir / "player_names.csv"
    player_names: dict[int, str] = {}
    if names_path.exists():
        n = pd.read_csv(names_path)
        player_names = dict(zip(n["player_id"].astype(int), n["player_name"]))
        print(f"\nLoaded {len(player_names):,} player names")
    else:
        print(f"\n  player_names.csv not found — IDs will be used")

    print(f"\nTotal possessions: {len(stints):,}")

    print(f"\nFitting RAPM on {len(stints):,} possessions...")
    rapm = fit_rapm(stints, lam=args.lam)
    rapm = rapm[rapm["poss"] >= args.min_poss].copy()
    rapm["player_name"] = rapm["player_id"].map(player_names).fillna(rapm["player_id"].astype(str))
    print(f"Players qualifying (>={args.min_poss} poss): {len(rapm)}")

    print("\nComputing per-player four-factor rates...")
    ff = compute_factor_rates(stints_rich)
    print(f"Players with factor data: {len(ff):,}")

    avg = league_averages(stints_rich)
    print("\nLeague averages:")
    for k, v in avg.items():
        print(f"  {k}: {v:.4f}")
    ff = demean_factors(ff, avg)

    merged = rapm.merge(
        ff[["player_id"] + FACTOR_COLS], on="player_id", how="inner"
    ).dropna(subset=FACTOR_COLS)

    ols = LinearRegression(fit_intercept=True)
    ols.fit(merged[FACTOR_COLS].values, merged["net_rapm"].values)
    merged["rapm_reconstructed"] = np.round(ols.predict(merged[FACTOR_COLS].values), 3)
    merged["residual"] = np.round(merged["net_rapm"] - merged["rapm_reconstructed"], 3)
    r2 = ols.score(merged[FACTOR_COLS].values, merged["net_rapm"].values)
    print(f"\n4F OLS fit — R^2 = {r2:.4f}  (factors explain {r2*100:.1f}% of RAPM variance)")

    display_cols = [
        "player_name", "poss",
        "net_rapm", "orapm", "drapm",
        "rapm_reconstructed", "residual",
        "ots", "otov", "oreb", "otrans",
        "dts", "dtov", "dreb", "dtrans",
    ]
    out = (
        merged[display_cols]
        .sort_values("net_rapm", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\nTop 25 by net RAPM (seasons: {', '.join(str(y) for y in args.seasons)}, "
          f"lambda={args.lam:,.0f}, min {args.min_poss} poss):")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", lambda v: f"{v:+.2f}"):
        print(out[["player_name", "poss", "net_rapm", "orapm", "drapm"]].head(25).to_string(index=False))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
