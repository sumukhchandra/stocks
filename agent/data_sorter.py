"""
agent.data_sorter: Automated Data Cleaner and Storage Optimizer.
Identifies temporary simulation caches, stale dumps, and optimizes disk space
while safeguarding critical master datasets and active models.
"""
import os
import sys
import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class DataSorter:
    """Manages the data lifecycle and cleans disposable items safely."""

    def __init__(self, root_dir=None):
        self.root_dir = root_dir or PROJECT_ROOT
        self.data_dir = os.path.join(self.root_dir, "data")
        # Critical files that must NEVER be deleted
        self.protected_files = {
            "master_labeled_dataset.parquet",
            "fetch_stock_data.py",
        }

    def inspect_storage(self):
        """Analyze disk usage across the data directory."""
        report = {
            "total_files": 0,
            "total_bytes": 0,
            "disposable_files": [],
            "protected_files": [],
        }

        if not os.path.exists(self.data_dir):
            return report

        for root, _, files in os.walk(self.data_dir):
            for f in files:
                filepath = os.path.join(root, f)
                try:
                    size = os.path.getsize(filepath)
                    report["total_files"] += 1
                    report["total_bytes"] += size

                    if f in self.protected_files:
                        report["protected_files"].append((filepath, size))
                    elif f.endswith(".tmp") or f.endswith(".bak") or f.startswith("temp_") or "cache" in root.lower():
                        report["disposable_files"].append((filepath, size))
                except Exception:
                    pass

        return report

    def cleanup_disposable_data(self, dry_run=False):
        """Remove only temporary / disposable files that are no longer needed."""
        inspection = self.inspect_storage()
        disposable = inspection["disposable_files"]
        reclaimed_bytes = 0
        cleaned_files = []

        for filepath, size in disposable:
            if not dry_run:
                try:
                    os.remove(filepath)
                    reclaimed_bytes += size
                    cleaned_files.append(filepath)
                except Exception:
                    pass
            else:
                reclaimed_bytes += size
                cleaned_files.append(filepath)

        return {
            "cleaned_count": len(cleaned_files),
            "reclaimed_mb": round(reclaimed_bytes / (1024 * 1024), 2),
            "files": cleaned_files,
            "dry_run": dry_run
        }
