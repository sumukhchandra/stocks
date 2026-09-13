"""
GitHub Auto-Updater for NSE Stocks Desktop Application.
Checks remote repository (https://github.com/sumukhchandra/stocks.git),
fetches latest commits from origin/main, applies updates safely,
and verifies dependencies.
"""
import os
import sys
import subprocess
import logging
from typing import Dict, Any, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DesktopUpdater")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GitHubUpdater:
    """Manages automatic updates from GitHub repository."""

    def __init__(self, root_dir: str = PROJECT_ROOT):
        self.root_dir = root_dir
        self.repo_url = "https://github.com/sumukhchandra/stocks.git"

    def _run_git(self, args: list, timeout: int = 15) -> Tuple[int, str, str]:
        """Execute a git command within the project root."""
        try:
            res = subprocess.run(
                ["git"] + args,
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return res.returncode, res.stdout.strip(), res.stderr.strip()
        except subprocess.TimeoutExpired:
            return -1, "", "Git command timed out."
        except Exception as e:
            return -1, "", str(e)

    def get_local_commit(self) -> str:
        """Get the current local HEAD commit hash."""
        code, out, _ = self._run_git(["rev-parse", "HEAD"])
        return out if code == 0 else "unknown"

    def get_remote_commit(self) -> str:
        """Fetch remote refs and get origin/main commit hash."""
        code, _, err = self._run_git(["fetch", "origin", "main", "--quiet"], timeout=10)
        if code != 0:
            logger.warning(f"Failed to fetch remote repository: {err}")
            return "unknown"
        code, out, _ = self._run_git(["rev-parse", "origin/main"])
        return out if code == 0 else "unknown"

    def has_uncommitted_changes(self) -> bool:
        """Check if working tree has unstaged or staged changes."""
        code, out, _ = self._run_git(["status", "--porcelain"])
        return code == 0 and len(out) > 0

    def check_and_update(self, auto_apply: bool = True) -> Dict[str, Any]:
        """
        Check GitHub for updates and automatically apply if available.
        
        Returns:
            Dict containing update status and details.
        """
        local_hash = self.get_local_commit()
        logger.info(f"Local commit: {local_hash[:7] if len(local_hash) >= 7 else local_hash}")

        remote_hash = self.get_remote_commit()
        if remote_hash == "unknown":
            return {
                "success": False,
                "updated": False,
                "local_commit": local_hash,
                "remote_commit": "unknown",
                "message": "Offline or cannot reach GitHub origin. Launching local version."
            }

        logger.info(f"Remote commit: {remote_hash[:7]}")

        if local_hash == remote_hash:
            return {
                "success": True,
                "updated": False,
                "local_commit": local_hash,
                "remote_commit": remote_hash,
                "message": "Application is up to date with GitHub main."
            }

        if not auto_apply:
            return {
                "success": True,
                "updated": False,
                "local_commit": local_hash,
                "remote_commit": remote_hash,
                "message": f"Update available: {remote_hash[:7]} (Local: {local_hash[:7]})."
            }

        # Apply update
        logger.info(f"Applying GitHub update: {local_hash[:7]} -> {remote_hash[:7]}...")
        stashed = False
        if self.has_uncommitted_changes():
            logger.info("Uncommitted changes detected. Stashing before pull...")
            s_code, _, _ = self._run_git(["stash", "save", "AutoUpdater-pre-update"])
            stashed = (s_code == 0)

        # Pull latest changes
        pull_code, pull_out, pull_err = self._run_git(["pull", "origin", "main"], timeout=30)

        # Restore stash if needed
        if stashed:
            self._run_git(["stash", "pop"])

        if pull_code != 0:
            logger.error(f"Git pull failed: {pull_err}")
            return {
                "success": False,
                "updated": False,
                "local_commit": local_hash,
                "remote_commit": remote_hash,
                "message": f"Pull failed: {pull_err}. Continuing with local version."
            }

        # Check if requirements.txt changed
        diff_code, diff_files, _ = self._run_git(["diff", "--name-only", local_hash, remote_hash])
        if diff_code == 0 and "requirements.txt" in diff_files:
            logger.info("requirements.txt changed in remote. Installing new packages...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"],
                    cwd=self.root_dir,
                    timeout=120
                )
            except Exception as e:
                logger.warning(f"Package installation warning: {e}")

        new_local = self.get_local_commit()
        return {
            "success": True,
            "updated": True,
            "local_commit": new_local,
            "remote_commit": remote_hash,
            "message": f"Successfully updated to commit {new_local[:7]} from GitHub."
        }


if __name__ == "__main__":
    updater = GitHubUpdater()
    status = updater.check_and_update(auto_apply=False)
    print("Update status:", status)
