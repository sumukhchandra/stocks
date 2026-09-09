"""
agent.trader_agent: Master Autonomous AI Agent for the NSE Trading System.
Coordinates:
1. Problem Discovery (SystemDiagnostics)
2. Risk & Account Management (AccountManager)
3. Data Lifecycle & Sorter (DataSorter)
4. Strategy & Dynamic Thresholds (StrategyAdvisor)
"""
import os
import sys
import argparse
from datetime import datetime

# Configure UTF-8 encoding for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.diagnostics import SystemDiagnostics
from agent.account_manager import AccountManager
from agent.data_sorter import DataSorter
from agent.strategy_advisor import StrategyAdvisor
from database.db_manager import DatabaseManager

class TraderAIAgent:
    """
    Autonomous AI Supervisor for the Trading System.
    Operates 100% locally with zero external API dependencies.
    """

    def __init__(self):
        self.diagnostics = SystemDiagnostics(PROJECT_ROOT)
        self.account = AccountManager()
        self.sorter = DataSorter(PROJECT_ROOT)
        self.advisor = StrategyAdvisor()
        self.db = DatabaseManager()

    def diagnose_system(self):
        """Find problems across code, models, data, and logs."""
        print("\n🔍 [AI AGENT] Scanning System for Problems & Anomalies...")
        report = self.diagnostics.run_full_diagnostics()

        print(f"  Overall Status: [{report['summary_status']}]")
        if report["critical_errors"]:
            print("  ❌ Critical Errors Found:")
            for err in report["critical_errors"]:
                print(f"     • {err}")
        else:
            print("  ✅ Zero Critical Errors.")

        if report["warnings"]:
            print("  ⚠️  Warnings / Non-Critical Observations:")
            for w in report["warnings"]:
                print(f"     • {w}")

        print(f"  ✅ Passed Checks: {len(report['passed_checks'])} system components verified.")

        self.db.log_event("DIAGNOSTIC", "TraderAIAgent", f"Diagnostics complete: {report['summary_status']}", report)
        return report

    def audit_account(self):
        """Monitor account health, drawdown, and sector limits."""
        print("\n💼 [AI AGENT] Auditing Portfolio & Account Risk...")
        report = self.account.get_account_report()

        print(f"  Total Capital:     ₹{report['total_capital']:,.2f}")
        print(f"  Available Cash:    ₹{report['available_capital']:,.2f}")
        print(f"  Invested Capital:  ₹{report['invested_capital']:,.2f} ({report['invested_pct']}%)")
        print(f"  Net Realized P&L:  ₹{report['net_pnl']:+,.2f} ({report['net_pnl_pct']:+.2f}%)")
        print(f"  Open Positions:    {report['open_positions_count']}")

        if report["sector_allocation_pct"]:
            print("  Sector Allocation:")
            for sec, pct in report["sector_allocation_pct"].items():
                print(f"     • {sec}: {pct:.1f}%")

        if report["risk_alerts"]:
            print("  ⚠️  Active Risk Alerts:")
            for alert in report["risk_alerts"]:
                print(f"     • {alert}")
        else:
            print("  ✅ Risk Profile: STABLE (All exposure limits respected)")

        self.db.log_event("ACCOUNT_AUDIT", "TraderAIAgent", f"Account Health: {report['health']}", report)
        return report

    def clean_temporary_data(self, dry_run=False):
        """Identify and clean disposable data files to reclaim storage."""
        print(f"\n🧹 [AI AGENT] Sorting Data & Cleaning Temporary Files (DryRun={dry_run})...")
        report = self.sorter.cleanup_disposable_data(dry_run=dry_run)
        print(f"  Cleaned {report['cleaned_count']} disposable files. Reclaimed {report['reclaimed_mb']} MB.")
        self.db.log_event("DATA_CLEANUP", "TraderAIAgent", f"Reclaimed {report['reclaimed_mb']} MB", report)
        return report

    def evaluate_strategy(self):
        """Compute performance metrics and recommend threshold adjustments."""
        print("\n📈 [AI AGENT] Evaluating Trading Strategy & Calibrating Thresholds...")
        report = self.advisor.evaluate_performance()
        if report["status"] == "ANALYZED":
            print(f"  Total Trades:     {report['total_trades']}")
            print(f"  Win Rate:         {report['win_rate_pct']:.1f}%")
            print(f"  Profit Factor:    {report['profit_factor']:.2f}")
            print(f"  Sharpe Ratio:     {report['sharpe_ratio']:.2f}")
            print(f"  Recommended Gate: {report['recommended_threshold']*100:.0f}% Confidence")
            print(f"  AI Advice:        {report['advice']}")
        else:
            print(f"  Status: {report['message']}")

        self.db.log_event("STRATEGY_ADVICE", "TraderAIAgent", report.get("advice", "Evaluation done"), report)
        return report

    def run_full_autonomous_cycle(self):
        """Run all agent workflows sequentially and generate an executive report."""
        print("=" * 60)
        print("  🤖 NSE AUTO-TRADER: AUTONOMOUS AI AGENT SUPERVISOR")
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        diag = self.diagnose_system()
        acc = self.audit_account()
        clean = self.clean_temporary_data(dry_run=False)
        strat = self.evaluate_strategy()

        print("\n" + "=" * 60)
        print("  ✨ AI AGENT SUPERVISORY SUMMARY: ALL OPERATIONS COMPLETE")
        print("=" * 60 + "\n")

        return {
            "diagnostics": diag,
            "account": acc,
            "cleanup": clean,
            "strategy": strat
        }

def main():
    parser = argparse.ArgumentParser(description="NSE Trading System AI Agent")
    parser.add_argument(
        "--action",
        choices=["diagnose", "account", "clean", "optimize", "all"],
        default="all",
        help="Action to execute"
    )
    args = parser.parse_args()

    agent = TraderAIAgent()
    if args.action == "diagnose":
        agent.diagnose_system()
    elif args.action == "account":
        agent.audit_account()
    elif args.action == "clean":
        agent.clean_temporary_data(dry_run=False)
    elif args.action == "optimize":
        agent.evaluate_strategy()
    else:
        agent.run_full_autonomous_cycle()

if __name__ == "__main__":
    main()
