"""
Master Launcher for NSE Stocks AI Native Desktop Application.
1. Checks for GitHub updates and applies them automatically.
2. Starts background Streamlit trading server (headless).
3. Launches dedicated native Windows desktop window.
"""
import os
import sys

# Ensure root is in path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from desktop.updater import GitHubUpdater
from desktop.app_window import DesktopWindowManager


def main():
    print("\n" + "=" * 60)
    print("   [+] NSE STOCKS AI - PHYSICAL DESKTOP APPLICATION")
    print("   Algorithmic Trading, Market Breadth & Real-Time Screener")
    print("=" * 60 + "\n")

    # Step 1: Auto-update from GitHub
    print("[1/2] Checking for updates from GitHub...")
    try:
        updater = GitHubUpdater(ROOT_DIR)
        status = updater.check_and_update(auto_apply=True)
        print(f"      {status.get('message', 'Checked.')}")
    except Exception as e:
        print(f"      [Notice] Offline or GitHub check skipped: {e}")

    # Step 2: Ensure iOS / Android PWA capabilities are enabled
    try:
        from desktop.patch_pwa import patch_streamlit_pwa
        patch_streamlit_pwa()
    except Exception as e:
        pass

    # Step 3: Launch Native Desktop Window
    print("\n[2/2] Launching Native Windows Desktop Window...")
    try:
        manager = DesktopWindowManager(port=8501)
        manager.run()
    except KeyboardInterrupt:
        print("\nExiting Desktop Application.")
    except Exception as e:
        print(f"\n[Error] Window launcher exception: {e}")


if __name__ == "__main__":
    main()
