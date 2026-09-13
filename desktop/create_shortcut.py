"""
Creates a Windows Desktop Shortcut (.lnk) for NSE Stocks AI.
Places the shortcut directly onto the user's Windows Desktop so they can
launch the physical app with a single click.
"""
import os
import sys
import subprocess
import logging

logger = logging.getLogger("CreateShortcut")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAT_PATH = os.path.join(PROJECT_ROOT, "START_DESKTOP_APP.bat")


def create_desktop_shortcut():
    """Create a Windows desktop shortcut pointing to START_DESKTOP_APP.bat."""
    # Find Windows desktop directory (accounting for OneDrive redirection if present)
    possible_desktops = [
        os.path.join(os.environ.get("USERPROFILE", ""), "OneDrive", "Desktop"),
        os.path.join(os.environ.get("USERPROFILE", ""), "Desktop"),
    ]
    desktop_dir = next((d for d in possible_desktops if os.path.exists(d)), None)

    if not desktop_dir:
        logger.error("Could not locate Windows Desktop directory.")
        return False

    shortcut_path = os.path.join(desktop_dir, "NSE Stocks AI.lnk")

    icon_path = os.path.join(PROJECT_ROOT, "desktop", "app_icon.ico")
    ps_script = f"""
    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut('{shortcut_path}')
    $Shortcut.TargetPath = '{BAT_PATH}'
    $Shortcut.WorkingDirectory = '{PROJECT_ROOT}'
    $Shortcut.IconLocation = '{icon_path}'
    $Shortcut.Description = 'NSE Stocks AI - Real-Time Algorithmic Trading Terminal'
    $Shortcut.Save()
    """

    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=10
        )
        if res.returncode == 0:
            print(f"[SUCCESS] Desktop shortcut successfully created at:\n   {shortcut_path}")
            return True
        else:
            print(f"Failed to create shortcut: {res.stderr}")
            return False
    except Exception as e:
        print(f"Error creating shortcut: {e}")
        return False


if __name__ == "__main__":
    create_desktop_shortcut()
