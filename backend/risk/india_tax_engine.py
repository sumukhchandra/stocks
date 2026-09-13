"""
Real Zerodha intraday equity cost model for Indian stock markets (2025-2026 rates).

Includes: brokerage (Rs.20 flat or 0.03%), STT, exchange charges, SEBI fees,
GST, stamp duty, slippage estimation, and short-term capital gains tax.
"""

import numpy as np


class IndiaTaxEngine:
    """Calculates all trading frictions for NSE intraday equity trades."""

    def __init__(self, profit_tax_pct=0.25):
        # -- Brokerage: Rs.20 flat or 0.03%, whichever is lower (per side) --
        self.brokerage_pct = 0.0003         # 0.03%
        self.brokerage_flat_cap = 20.0      # Rs.20 cap per order

        # -- STT: 0.025% on sell side only (intraday equity) --
        self.stt_sell_pct = 0.00025

        # -- Exchange transaction charges (NSE) --
        self.exchange_pct = 0.0000307       # 0.00307% per side

        # -- SEBI turnover fee --
        self.sebi_pct = 0.000001            # Rs.10 per crore = 0.0001%

        # -- GST: 18% on (brokerage + exchange + SEBI) --
        self.gst_pct = 0.18

        # -- Stamp duty: ~0.003% on buy side (Maharashtra rate) --
        self.stamp_duty_buy_pct = 0.00003

        # -- Slippage estimate --
        self.default_slippage_bps = 2.5     # 0.025%

        # -- Short-term capital gains tax (user set 25% as safety buffer) --
        self.profit_tax_pct = profit_tax_pct

    def _brokerage_per_side(self, trade_value):
        """Brokerage = min(0.03% of trade value, Rs.20)."""
        return min(trade_value * self.brokerage_pct, self.brokerage_flat_cap)

    def calculate_total_cost(self, capital, gross_return, slippage_bps=None):
        """
        Returns a detailed INR cost breakdown for one complete round-trip trade.

        Parameters
        ----------
        capital : float
            Amount invested in the buy side (INR).
        gross_return : float
            Gross price movement as a decimal (e.g. 0.01 = 1%).
        slippage_bps : float, optional
            Override slippage in basis points. Defaults to self.default_slippage_bps.

        Returns
        -------
        dict with all cost components and net profit.
        """
        if slippage_bps is None:
            slippage_bps = self.default_slippage_bps

        buy_value = capital
        sell_value = capital * (1 + gross_return)

        # 1. Brokerage (both sides, each capped at Rs.20)
        buy_brokerage = self._brokerage_per_side(buy_value)
        sell_brokerage = self._brokerage_per_side(sell_value)
        total_brokerage = buy_brokerage + sell_brokerage

        # 2. STT (sell side only for intraday)
        stt = sell_value * self.stt_sell_pct

        # 3. Exchange transaction charges (both sides)
        exchange = (buy_value + sell_value) * self.exchange_pct

        # 4. SEBI turnover fees (both sides)
        sebi = (buy_value + sell_value) * self.sebi_pct

        # 5. GST: 18% on (brokerage + exchange charges + SEBI fees)
        gst_base = total_brokerage + exchange + sebi
        gst = gst_base * self.gst_pct

        # 6. Stamp duty (buy side only)
        stamp = buy_value * self.stamp_duty_buy_pct

        # 7. Slippage (both sides)
        slippage_pct = slippage_bps / 10000
        slippage = (buy_value + sell_value) * slippage_pct

        # -- Totals --
        total_cost = total_brokerage + stt + exchange + sebi + gst + stamp + slippage
        gross_profit = sell_value - buy_value
        profit_after_costs = gross_profit - total_cost
        tax = max(0.0, profit_after_costs * self.profit_tax_pct)
        net_profit = profit_after_costs - tax
        net_return_pct = (net_profit / capital) * 100 if capital > 0 else 0.0

        return {
            "capital": capital,
            "gross_return_pct": gross_return * 100,
            "buy_value": buy_value,
            "sell_value": sell_value,
            "brokerage": round(total_brokerage, 2),
            "stt": round(stt, 2),
            "exchange_charges": round(exchange, 2),
            "sebi_fees": round(sebi, 2),
            "gst": round(gst, 2),
            "stamp_duty": round(stamp, 2),
            "slippage": round(slippage, 2),
            "total_cost": round(total_cost, 2),
            "gross_profit": round(gross_profit, 2),
            "profit_after_costs": round(profit_after_costs, 2),
            "tax": round(tax, 2),
            "net_profit": round(net_profit, 2),
            "net_return_pct": round(net_return_pct, 4),
        }

    def calculate_net_return(self, gross_return, slippage_bps=None):
        """
        Quick helper: returns the net return as a decimal after all costs.
        Uses a Rs.1,00,000 reference capital for percentage calculation.
        """
        breakdown = self.calculate_total_cost(100000, gross_return, slippage_bps)
        return breakdown["net_profit"] / 100000

    def is_trade_viable(self, capital, predicted_gross_return, min_net_pct=0.005):
        """
        Entry gate: returns (True/False, breakdown_dict).

        A trade is viable ONLY if the predicted net return after ALL costs
        (brokerage, STT, exchange, GST, stamp duty, slippage, tax) exceeds
        min_net_pct (default 0.5%).
        """
        breakdown = self.calculate_total_cost(capital, predicted_gross_return)
        is_viable = breakdown["net_profit"] > (capital * min_net_pct)
        return is_viable, breakdown

    def is_exit_profitable(self, capital, gross_return, min_net_pct=0.005):
        """
        Exit gate: returns True if the REALISED gross return produces a
        net profit > min_net_pct after all costs.
        """
        breakdown = self.calculate_total_cost(capital, gross_return)
        return breakdown["net_profit"] > (capital * min_net_pct), breakdown

    def get_required_gross_for_net(self, capital=100000, target_net_pct=0.005):
        """
        Binary search: find the minimum gross return needed to achieve
        target_net_pct net return after all costs.
        """
        lo, hi = 0.0, 0.05  # Search between 0% and 5%
        for _ in range(100):
            mid = (lo + hi) / 2
            breakdown = self.calculate_total_cost(capital, mid)
            net_pct = breakdown["net_profit"] / capital
            if net_pct < target_net_pct:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def get_min_gross_threshold(self, min_net_bps=50, slippage_bps=2.5):
        """
        Backwards-compatible helper: returns min gross return (as decimal)
        needed for a target net return in basis points.
        """
        target_net_pct = min_net_bps / 10000
        return self.get_required_gross_for_net(target_net_pct=target_net_pct)

    def calculate_scalp_viability(self, capital, target_net_pct=0.005):
        """
        Pre-compute the minimum gross return needed for a scalp trade to
        guarantee target_net_pct (default 0.5%) net return after ALL costs.
        Also returns the cost breakdown at that gross return.
        """
        min_gross = self.get_required_gross_for_net(capital=capital, target_net_pct=target_net_pct)
        breakdown = self.calculate_total_cost(capital, min_gross)
        return {
            "capital": capital,
            "target_net_pct": target_net_pct * 100,
            "required_gross_pct": min_gross * 100,
            "cost_at_threshold": breakdown["total_cost"],
            "tax_at_threshold": breakdown["tax"],
            "net_profit_at_threshold": breakdown["net_profit"],
            "net_return_at_threshold": breakdown["net_return_pct"],
            "viable": breakdown["net_return_pct"] >= target_net_pct * 100,
        }

    def calculate_chain_tax(self, starting_capital, gross_returns_list):
        """
        Compute cumulative taxes and P&L across a chain of N sequential
        compounding trades where capital + net_profit rolls into the next trade.

        Parameters
        ----------
        starting_capital : float
            Initial capital at the start of the chain.
        gross_returns_list : list[float]
            List of gross returns (decimals) for each trade in the chain.

        Returns
        -------
        dict with per-trade breakdowns, cumulative P&L, and final capital.
        """
        current_capital = starting_capital
        trades = []
        total_costs = 0.0
        total_taxes = 0.0
        total_gross_profit = 0.0
        total_net_profit = 0.0

        for i, gross_ret in enumerate(gross_returns_list):
            bd = self.calculate_total_cost(current_capital, gross_ret)
            trades.append({
                "trade_num": i + 1,
                "capital_in": round(current_capital, 2),
                "gross_return_pct": round(gross_ret * 100, 4),
                "gross_profit": bd["gross_profit"],
                "costs": bd["total_cost"],
                "tax": bd["tax"],
                "net_profit": bd["net_profit"],
                "net_return_pct": bd["net_return_pct"],
                "capital_out": round(current_capital + bd["net_profit"], 2),
            })
            total_costs += bd["total_cost"]
            total_taxes += bd["tax"]
            total_gross_profit += bd["gross_profit"]
            total_net_profit += bd["net_profit"]
            current_capital += bd["net_profit"]

        overall_net_return = ((current_capital - starting_capital) / starting_capital) * 100

        return {
            "starting_capital": starting_capital,
            "final_capital": round(current_capital, 2),
            "num_trades": len(gross_returns_list),
            "total_gross_profit": round(total_gross_profit, 2),
            "total_costs": round(total_costs, 2),
            "total_taxes": round(total_taxes, 2),
            "total_net_profit": round(total_net_profit, 2),
            "overall_net_return_pct": round(overall_net_return, 4),
            "meets_target": overall_net_return >= 0.5,
            "trades": trades,
        }


if __name__ == "__main__":
    engine = IndiaTaxEngine()

    # Example: What gross return do I need for 0.5% net?
    required = engine.get_required_gross_for_net(capital=100000, target_net_pct=0.005)
    print(f"To net 0.5%, required gross return: {required*100:.4f}%")

    # Example: Full cost breakdown for a 1% gross move on Rs.1,00,000
    breakdown = engine.calculate_total_cost(100000, 0.01)
    print(f"\n--- Cost Breakdown for 1% gross on Rs.1,00,000 ---")
    for k, v in breakdown.items():
        print(f"  {k}: {v}")

    # Example: Is a 0.8% predicted move viable?
    viable, details = engine.is_trade_viable(100000, 0.008)
    print(f"\nIs 0.8% gross viable? {viable} (net: {details['net_return_pct']:.4f}%)")

    # Scalp viability check
    print("\n--- Scalp Viability (Rs.10,000 capital) ---")
    sv = engine.calculate_scalp_viability(10000, target_net_pct=0.005)
    for k, v in sv.items():
        print(f"  {k}: {v}")

    # Chain tax: 10 trades at 1.0% gross each, starting with Rs.10,000
    print("\n--- Compound Chain: 10 trades @ 1.0% gross ---")
    chain = engine.calculate_chain_tax(10000, [0.01] * 10)
    print(f"  Starting: Rs.{chain['starting_capital']:,.2f}")
    print(f"  Final:    Rs.{chain['final_capital']:,.2f}")
    print(f"  Net P&L:  Rs.{chain['total_net_profit']:+,.2f}")
    print(f"  Overall:  {chain['overall_net_return_pct']:+.4f}%")
    print(f"  Target met: {chain['meets_target']}")

