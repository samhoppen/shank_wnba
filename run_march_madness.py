"""
Launcher for the NCAA March Madness Bracket Simulator.
Opens the Streamlit app in a browser window automatically.

Port: 8507
"""
import sys
import os
import threading
import webbrowser
import time


def resource_path(relative_path):
    """Get path to bundled resource; works in dev and PyInstaller."""
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)


def open_browser():
    time.sleep(3)
    webbrowser.open("http://localhost:8507")


def main():
    script = resource_path("march_madness_sim_app.py")

    sys.argv = [
        "streamlit", "run", script,
        "--global.developmentMode=false",
        "--server.port=8507",
        "--server.headless=true",
        "--server.address=localhost",
        "--browser.gatherUsageStats=false",
    ]

    t = threading.Thread(target=open_browser, daemon=True)
    t.start()

    from streamlit.web import cli as stcli
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
