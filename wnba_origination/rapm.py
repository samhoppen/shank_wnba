"""
Incremental ridge-regression RAPM for the current WNBA season.

Reads stints files, builds the design matrix, fits orapm + drapm separately,
and writes rapm_2026_RS.csv to data/.

Usage:
  python rapm.py              # fit current season (auto-detect latest year)
  python rapm.py --year 2025  # explicit year
"""
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
from pathlib import Path
from paths import RAPM_DIR, DATA, stints, rapm_season

LAMBDA = 2000   # ridge penalty (possession-scale)
MIN_POSS_OUTPUT = 100


def load_stints_for_year(year: int) -> pd.DataFrame:
    path = stints(year)
    if not path.exists():
        raise FileNotFoundError(f"Stints not found: {path}")
    return pd.read_csv(path)


def fit_rapm(df: pd.DataFrame, player_ids: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit oRAPM and dRAPM from possession-level stints via ridge regression.

    Design matrix: offensive players get +1, defensive players get -1.
    Single regression predicts points scored per possession; intercept = league average.
    Coefficients are player effects (positive = helps offense or hurts defense).

    We split into two separate regressions so oRAPM and dRAPM are interpretable separately:
      - oRAPM regression: only offensive player columns (+1), predict points scored
      - dRAPM regression: only defensive player columns (+1), predict points allowed
        then negate so positive = good defense
    """
    pid_to_col = {pid: i for i, pid in enumerate(player_ids)}
    n = len(player_ids)
    n_stints = len(df)

    off_cols = [f"off_p{i}" for i in range(1, 6)]
    def_cols = [f"def_p{i}" for i in range(1, 6)]

    off_rows, off_cols_idx = [], []
    def_rows, def_cols_idx = [], []

    for stint_idx, row in enumerate(df.itertuples()):
        for col_name in off_cols:
            pid = int(getattr(row, col_name))
            if pid in pid_to_col:
                off_rows.append(stint_idx)
                off_cols_idx.append(pid_to_col[pid])
        for col_name in def_cols:
            pid = int(getattr(row, col_name))
            if pid in pid_to_col:
                def_rows.append(stint_idx)
                def_cols_idx.append(pid_to_col[pid])

    off_vals = [1.0] * len(off_rows)
    def_vals = [1.0] * len(def_rows)

    X_off = csr_matrix((off_vals, (off_rows, off_cols_idx)), shape=(n_stints, n))
    X_def = csr_matrix((def_vals, (def_rows, def_cols_idx)), shape=(n_stints, n))

    y = df["points"].values.astype(float)

    ridge = Ridge(alpha=LAMBDA, fit_intercept=True)
    ridge.fit(X_off, y)
    orapm = ridge.coef_ * 100   # scale to per-100 possessions

    ridge_d = Ridge(alpha=LAMBDA, fit_intercept=True)
    ridge_d.fit(X_def, y)
    drapm = -ridge_d.coef_ * 100   # negate: positive = good defense

    return orapm, drapm


def run(year: int | None = None, save: bool = True) -> pd.DataFrame:
    if year is None:
        # Auto-detect: check for 2026 stints, fall back to 2025
        for yr in [2026, 2025]:
            if stints(yr).exists():
                year = yr
                break
        if year is None:
            raise RuntimeError("No stints file found for 2025 or 2026")

    print(f"Fitting RAPM for {year}...")
    df = load_stints_for_year(year)
    print(f"  {len(df):,} possessions loaded")

    off_cols = [f"off_p{i}" for i in range(1, 6)]
    def_cols = [f"def_p{i}" for i in range(1, 6)]
    all_player_ids = sorted(
        set(df[off_cols].values.flatten()) | set(df[def_cols].values.flatten())
    )
    all_player_ids = [p for p in all_player_ids if not np.isnan(p)]
    all_player_ids = [int(p) for p in all_player_ids]

    orapm, drapm = fit_rapm(df, all_player_ids)
    net = orapm + drapm

    # Count possessions per player
    poss_off = df[off_cols].apply(pd.Series.value_counts).sum(axis=1)
    poss_map = {}
    for col in off_cols + def_cols:
        for pid, cnt in df[col].value_counts().items():
            poss_map[int(pid)] = poss_map.get(int(pid), 0) + cnt

    result = pd.DataFrame({
        "player_id": all_player_ids,
        "orapm": orapm,
        "drapm": drapm,
        "net_rapm": net,
        "poss": [poss_map.get(pid, 0) for pid in all_player_ids],
        "season": year,
        "season_type": "RS",
    })
    result = result[result["poss"] >= MIN_POSS_OUTPUT].sort_values("net_rapm", ascending=False)

    if save:
        out_path = DATA / f"rapm_{year}_RS.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        print(f"  Saved {len(result)} players -> {out_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None)
    args = parser.parse_args()
    df = run(year=args.year)
    print(df[["player_id", "orapm", "drapm", "net_rapm", "poss"]].head(20).to_string())
