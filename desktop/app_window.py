"""
Native Windows Desktop Window Controller for NSE Stocks AI.
Hosts the Streamlit application in a dedicated, frameless/native desktop window
using pywebview (Edge WebView2) with fallback to Microsoft Edge App Mode.
"""
import os
import sys
import time
import socket
import logging
import subprocess
import urllib.request
from typing import Optional

logger = logging.getLogger("DesktopWindow")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_ENTRY = os.path.join(PROJECT_ROOT, "frontend", "app.py")
DEFAULT_PORT = 8501


class DesktopWindowManager:
    """Manages the lifecycle of the Streamlit server and the Native Desktop Window."""

    def __init__(self, port: int = DEFAULT_PORT):
        self.port = port
        self.url = f"http://localhost:{self.port}"
        self.server_process: Optional[subprocess.Popen] = None

    def is_server_listening(self) -> bool:
        """Check if Streamlit server is answering HTTP requests on the port."""
        try:
            with urllib.request.urlopen(self.url, timeout=1.5) as response:
                return response.status == 200
        except Exception:
            return False

    def start_streamlit_server(self) -> bool:
        """Starts Streamlit in a hidden background process if not already running."""
        if self.is_server_listening():
            logger.info(f"Streamlit server already active at {self.url}.")
            return True

        logger.info(f"Starting Streamlit backend on port {self.port}...")
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW

        cmd = [
            sys.executable, "-m", "streamlit", "run",
            APP_ENTRY,
            "--server.port", str(self.port),
            "--server.headless", "true",
            "--browser.serverAddress", "localhost",
            "--server.runOnSave", "false",
            "--browser.gatherUsageStats", "false"
        ]

        self.server_process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags
        )

        # Wait up to 15 seconds for the server to become ready
        for i in range(30):
            time.sleep(0.5)
            if self.is_server_listening():
                logger.info(f"Streamlit server initialized successfully in {(i+1)*0.5:.1f}s.")
                return True

        logger.error("Timed out waiting for Streamlit server to start.")
        return False

    def launch_pywebview_window(self) -> bool:
        """Launches native desktop window via pywebview (Edge WebView2)."""
        try:
            import webview

            logger.info("Opening Native Desktop Window via pywebview (Edge WebView2)...")
            window = webview.create_window(
                title="NSE Stocks AI — Algorithmic Trading & Market Terminal",
                url=self.url,
                width=1480,
                height=920,
                resizable=True,
                min_size=(1024, 700),
                confirm_close=False
            )
            webview.start(gui="edgechromium")
            return True
        except Exception as e:
            logger.warning(f"pywebview window could not start ({e}). Falling back to Native App Mode.")
            return False

    def launch_edge_app_mode(self) -> bool:
        """Fallback: Launches standalone native App Mode window using Edge or Chrome."""
        candidates = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        ]
        browser_exe = next((p for p in candidates if os.path.exists(p)), None)
        if not browser_exe:
            logger.error("No compatible browser found for native app window.")
            return False

        logger.info(f"Launching Native Standalone App Window with {os.path.basename(browser_exe)}...")
        cmd = [
            browser_exe,
            f"--app={self.url}",
            "--window-size=1480,920",
            "--window-position=50,50"
        ]
        proc = subprocess.Popen(cmd)
        proc.wait()  # Block until user closes desktop window
        return True

    def run(self):
        """Main lifecycle entrypoint: ensures server is up, launches window, tears down on close."""
        server_ok = self.start_streamlit_server()
        if not server_ok:
            print(f"Error: Could not start local engine. You can manually access {self.url}")
            return

        print(f"[INFO] Physical App Active: Connecting to {self.url}...")
        # Prioritize Native App Mode for zero-freeze, instant startup
        launched = self.launch_edge_app_mode()
        if not launched:
            launched = self.launch_pywebview_window()

        self.cleanup()

    def cleanup(self):
        """Terminates background server process when window is closed."""
        if self.server_process:
            logger.info("Closing Streamlit backend server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None


if __name__ == "__main__":
    manager = DesktopWindowManager()
    manager.run()
