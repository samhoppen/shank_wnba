"""
Refresh all app-side caches after new game data has been fetched by the notebook.

Run this AFTER running wnba_rapm.ipynb cell 0 with:
    RUN_SEASON = 2025
    RUN_REBUILD_STINTS = True

Usage:
    python refresh.py              # rebuild everything for 2025
    python refresh.py --year 2025  # explicit year
    python refresh.py --all        # rebuild game_log for all years
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(HERE))
    if result.returncode != 0:
        print(f"ERROR: command failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--all", action="store_true", help="Rebuild game_log for all years")
    args = parser.parse_args()

    py = sys.executable

    # 1. Rebuild player store (picks up new RAPM CSVs from notebook)
    print("\n=== Step 1: Rebuild player store ===")
    run([py, "player_store.py"])

    # 2. Rebuild game log (reads raw_pbp JSONs — new games auto-included)
    print("\n=== Step 2: Rebuild game log ===")
    if args.all:
        run([py, "game_log.py"])
    else:
        run([py, "game_log.py", "--year", str(args.year)])

    print("\n✓ Done. Restart Streamlit (Ctrl+C then `streamlit run app.py`) to pick up changes.")
    print("  Or just hit R in the Streamlit browser window to rerun.")


if __name__ == "__main__":
    main()
