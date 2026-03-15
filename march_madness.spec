# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for MarchMadnessSimulator.exe
Build: pyinstaller march_madness.spec --clean
"""
import sys
import os
from pathlib import Path
import streamlit
from PyInstaller.utils.hooks import copy_metadata

STREAMLIT_DIR = Path(streamlit.__file__).parent
WORK_DIR = Path(r"C:\Users\shank.subramani_betf\Desktop\ShotsDashboard\.claude\worktrees\eloquent-blackwell")

a = Analysis(
    [str(WORK_DIR / "run_march_madness.py")],
    pathex=[str(WORK_DIR)],
    binaries=[],
    datas=[
        # Package metadata (fixes PackageNotFoundError for streamlit)
        *copy_metadata("streamlit"),
        # Main app script & data
        (str(WORK_DIR / "march_madness_sim_app.py"),                         "."),
        (str(WORK_DIR.parent.parent.parent / "team_ratings_cache.csv"),      "."),
        # Streamlit frontend assets
        (str(STREAMLIT_DIR / "static"),  "streamlit/static"),
        (str(STREAMLIT_DIR / "runtime"), "streamlit/runtime"),
    ],
    hiddenimports=[
        "streamlit",
        "streamlit.web.cli",
        "streamlit.web.server",
        "streamlit.web.server.server",
        "streamlit.runtime",
        "streamlit.runtime.scriptrunner",
        "streamlit.runtime.state",
        "streamlit.components.v1",
        "scipy",
        "scipy.stats",
        "scipy.stats._continuous_distns",
        "scipy.stats._distn_infrastructure",
        "scipy.special",
        "scipy.special._ufuncs",
        "numpy",
        "pandas",
        "altair",
        "pydeck",
        "click",
        "tornado",
        "packaging",
        "importlib_metadata",
        "pkg_resources",
        "pyarrow",
        "tzlocal",
        "rich",
        "gitpython",
        "watchdog",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="MarchMadnessSimulator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
