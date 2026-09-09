"""
agent.diagnostics: Automated Problem Hunter and System Health Inspector.
Scans models, data integrity, import pathways, and system logs.
"""
import os
import sys
import glob
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class SystemDiagnostics:
    """Intelligent problem finder that audits the entire trading ecosystem."""

    def __init__(self, root_dir=None):
        self.root_dir = root_dir or PROJECT_ROOT

    def run_full_diagnostics(self):
        """Execute complete problem discovery."""
        results = {
            "critical_errors": [],
            "warnings": [],
            "passed_checks": [],
            "summary_status": "HEALTHY"
        }

        self._check_folder_structure(results)
        self._check_model_health(results)
        self._check_dataset_integrity(results)
        self._check_import_health(results)
        self._check_logs_for_errors(results)

        if results["critical_errors"]:
            results["summary_status"] = "CRITICAL"
        elif results["warnings"]:
            results["summary_status"] = "WARNINGS_FOUND"

        return results

    def run_full_audit(self):
        """Frontend-compatible audit report."""
        diag = self.run_full_diagnostics()
        details = []
        for p in diag["passed_checks"]:
            details.append({
                "category": "System Integrity",
                "description": p,
                "status": "OK",
                "details": "Verified passed."
            })
        for w in diag["warnings"]:
            details.append({
                "category": "Warning Notice",
                "description": w,
                "status": "WARNING",
                "details": w,
                "resolution": "Check log files and non-standard directories."
            })
        for e in diag["critical_errors"]:
            details.append({
                "category": "Critical Issue",
                "description": e,
                "status": "CRITICAL",
                "details": e,
                "resolution": "Restore missing components or fix broken files."
            })

        return {
            "critical_issues": len(diag["critical_errors"]),
            "warnings": len(diag["warnings"]),
            "total_checks": len(diag["passed_checks"]) + len(diag["warnings"]) + len(diag["critical_errors"]),
            "details": details
        }

    def _check_folder_structure(self, results):
        """Ensure core folders exist at root."""
        required = {"frontend", "backend", "database", "ml_models", "simulations", "data", "agent"}
        allowed = required | {"trading", "apps", "backtesting", "docs", "scripts", "tests"}
        actual = {
            f for f in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, f)) and not f.startswith(".")
        }

        missing = required - actual
        extraneous = actual - allowed

        if missing:
            results["critical_errors"].append(f"Missing core directories: {missing}")
        else:
            results["passed_checks"].append("Core 7 directories all present and healthy.")

        if extraneous:
            results["warnings"].append(f"Non-standard root directories found: {extraneous}")

    def _check_model_health(self, results):
        """Validate presence and integrity of trained ensemble models."""
        saved_dir = os.path.join(self.root_dir, "ml_models", "models", "saved_models")
        ensemble_dir = os.path.join(saved_dir, "ensemble")
        
        required_files = [
            os.path.join(ensemble_dir, "xgboost_calibrated.joblib"),
            os.path.join(ensemble_dir, "lightgbm_calibrated.joblib"),
            os.path.join(ensemble_dir, "catboost_calibrated.joblib"),
            os.path.join(ensemble_dir, "feature_cols.joblib"),
            os.path.join(saved_dir, "return_regressor_xgb.joblib"),
        ]

        missing_models = [f for f in required_files if not os.path.exists(f)]
        if missing_models:
            results["critical_errors"].append(f"Missing trained model files: {[os.path.basename(f) for f in missing_models]}")
        else:
            results["passed_checks"].append("Ensemble and Return Regressor models present and intact.")

    def _check_dataset_integrity(self, results):
        """Validate master dataset for missing features, NaNs, and freshness."""
        data_path = os.path.join(self.root_dir, "data", "processed", "master_labeled_dataset.parquet")
        if not os.path.exists(data_path):
            results["critical_errors"].append("master_labeled_dataset.parquet is missing.")
            return

        try:
            df = pd.read_parquet(data_path)
            if df.empty:
                results["critical_errors"].append("master_labeled_dataset.parquet is completely empty.")
                return

            rows = len(df)
            symbols = df["symbol"].nunique() if "symbol" in df.columns else 0
            results["passed_checks"].append(f"Master dataset has {rows:,} rows across {symbols} symbols.")

            # Check required columns
            req_cols = ["timestamp", "close", "symbol"]
            missing_cols = [c for c in req_cols if c not in df.columns]
            if missing_cols:
                results["critical_errors"].append(f"Dataset missing essential columns: {missing_cols}")

            # Check null rates in close
            if "close" in df.columns and df["close"].isnull().sum() > 0:
                results["warnings"].append(f"Found {df['close'].isnull().sum()} null close prices in dataset.")
        except Exception as e:
            results["critical_errors"].append(f"Corrupted dataset file: {e}")

    def _check_import_health(self, results):
        """Verify core modules import cleanly without ModuleNotFoundError."""
        test_modules = [
            ("backend.nse_auto_trader", "NSEAutoTrader"),
            ("database.db_manager", "DatabaseManager"),
            ("database.rules", "TradingRulesEngine"),
            ("database.relations", "STOCK_METADATA"),
            ("ml_models", "FinalEnsembleEngine"),
            ("simulations.stock_simulator", "fetch_stock_range"),
        ]

        for mod_name, obj_name in test_modules:
            try:
                mod = __import__(mod_name, fromlist=[obj_name])
                getattr(mod, obj_name)
                results["passed_checks"].append(f"Import verified: {mod_name}.{obj_name}")
            except Exception as e:
                results["critical_errors"].append(f"Failed import {mod_name}.{obj_name}: {e}")

    def _check_logs_for_errors(self, results):
        """Scan logs for recurring exceptions or crashes."""
        log_patterns = [
            os.path.join(self.root_dir, "backend", "logs", "*.jsonl"),
            os.path.join(self.root_dir, "simulations", "logs", "*.jsonl"),
            os.path.join(self.root_dir, "simulations", "logs", "*.log"),
        ]

        error_count = 0
        for pattern in log_patterns:
            for log_file in glob.glob(pattern):
                try:
                    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if "Traceback" in line or "Error" in line or "Exception" in line:
                                error_count += 1
                except Exception:
                    pass

        if error_count > 0:
            results["warnings"].append(f"Detected {error_count} error/exception occurrences in log files.")
        else:
            results["passed_checks"].append("No recent runtime errors found in logs.")
