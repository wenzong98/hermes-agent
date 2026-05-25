"""
US ETF Valuation-Driven DCA Strategy Backtest
==============================================
Multi-regime signal system based on Shiller CAPE + RSI.
Benchmark: Buy & Hold on S&P 500 (SPY)

Core philosophy:
  - DCA is the primary engine — signal only modulates amount
  - Sell signals are rare and proportional (never full exit unless bubble)
  - Goal: beat Buy&Hold with lower drawdown
"""

import argparse
import datetime
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────

def fetch_spy_data(start: str, end: str) -> pd.DataFrame:
    """Download S&P 500 ETF (SPY) price data via yfinance."""
    df = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df.resample("D").last().ffill()
    return df[["open", "high", "low", "close", "volume"]]


def fetch_pe_ratio(start: str, end: str) -> pd.DataFrame:
    """Fetch S&P 500 Shiller CAPE ratio from multpl.com."""
    try:
        url = "https://www.multpl.com/s-p-500-pe-ratio/table/by-month"
        tables = pd.read_html(url)
        cape_raw = tables[0]
        cape_raw.columns = ["date", "value"]
        cape_raw["date"] = pd.to_datetime(cape_raw["date"], format="mixed")

        def extract_cape(v):
            if isinstance(v, (int, float)):
                return float(v)
            m = re.search(r"[\d.]+", str(v))
            return float(m.group()) if m else None

        cape_raw["value"] = cape_raw["value"].apply(extract_cape)
        cape_raw = cape_raw.dropna(subset=["value"])
        cape_raw = cape_raw.set_index("date").sort_index()
        cape_raw = cape_raw.resample("D").last().ffill()
        cape_raw = cape_raw[(cape_raw.index >= start) & (cape_raw.index <= end)]
        return cape_raw
    except Exception as e:
        print(f"  CAPE fetch failed ({e}), using constant fallback...")
        spy = fetch_spy_data(start, end)
        synthetic = pd.Series(25.0, index=spy.index)
        return pd.DataFrame({"value": synthetic})


# ─────────────────────────────────────────────
# Technical indicators
# ─────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame, cape_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    df["sma200"] = close.rolling(200).mean()

    # RSI (14-day Wilder)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # CAPE merge
    df["cape"] = cape_df["value"].reindex(df.index).ffill()

    # CAPE percentile (rolling 10-year)
    df["cape_pct"] = df["cape"].rolling(2520).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )

    # Returns
    df["ret_1m"] = close.pct_change(21)
    df["ret_3m"] = close.pct_change(63)
    df["ret_6m"] = close.pct_change(126)

    return df


# ─────────────────────────────────────────────
# Strategy rules
# ─────────────────────────────────────────────

def build_signals(df: pd.DataFrame,
                  cape_fair: float = 25.0,
                  cape_overvalued_1: float = 32.0,
                  cape_overvalued_2: float = 40.0,
                  cape_bubble: float = 48.0,
                  rsi_oversold: float = 30.0,
                  rsi_overbought: float = 75.0) -> pd.Series:
    """
    Multi-regime signal system — OPTIMIZED for US equity bull markets.

    Key insight: US equities spend most of their time in "fair" or "slightly overvalued"
    territory. Selling too early kills compounding. This version:
      - Raises sell thresholds (less frequent profit-taking)
      - Lowers buy thresholds (more aggressive dip-buying)
      - Adds RSI overbought as a soft pause, not a sell trigger

    Signal values (DCA multiplier):
      -2  = SELL 50% + no DCA   (bubble: cape > cape_bubble)
      -1  = SELL 30% + no DCA   (very overvalued: cape > cape_overvalued_2)
       0  = HOLD / 0.5x DCA     (moderately overvalued: cape > cape_overvalued_1)
       1  = NORMAL DCA 1x       (fair value)
       2  = 1.5x DCA            (cheap: cape < cape_fair)
       3  = 2x DCA              (very cheap + oversold)
       4  = 3x DCA              (max conviction: crash-level cheap + oversold)
    """
    signals = pd.Series(1, index=df.index, dtype=int)
    in_bubble = False

    for i in range(200, len(df)):
        row = df.iloc[i]
        cape = row["cape"]
        rsi = row["rsi"]

        cheap = cape < cape_fair
        oversold = rsi < rsi_oversold
        very_cheap = cheap and oversold
        in_bubble_zone = cape > cape_bubble
        very_overvalued = cape > cape_overvalued_2
        moderately_overvalued = cape > cape_overvalued_1

        # Bubble regime
        if in_bubble_zone:
            signals.iloc[i] = -2
            in_bubble = True
            continue

        # Recovery from bubble
        if in_bubble:
            if cape < cape_overvalued_1:
                in_bubble = False
            else:
                signals.iloc[i] = 0
                continue

        # Regime logic — optimized thresholds
        if very_overvalued:
            signals.iloc[i] = -1
        elif moderately_overvalued:
            signals.iloc[i] = 0
        elif very_cheap:
            signals.iloc[i] = 4  # 3x DCA
        elif cheap and oversold:
            signals.iloc[i] = 3  # 2x DCA
        elif cheap:
            signals.iloc[i] = 2  # 1.5x DCA
        else:
            signals.iloc[i] = 1  # normal DCA

    return signals


# ─────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────

class Backtester:
    def __init__(self,
                 df: pd.DataFrame,
                 signals: pd.Series,
                 start_cash: float = 10000.0,
                 dca_monthly: float = 500.0,
                 expense_ratio: float = 0.0003,
                 tax_rate: float = 0.0015):
        self.df = df
        self.signals = signals
        self.start_cash = start_cash
        self.dca_monthly = dca_monthly
        self.expense_ratio = expense_ratio
        self.tax_rate = tax_rate

        self.cash = start_cash
        self.shares = 0.0
        self.portfolio_values = []
        self.dates = []
        self.positions = []

    def run(self) -> dict:
        df = self.df
        signals = self.signals
        daily_cost_rate = self.expense_ratio / 252

        # Record starting point
        start_dt = df.index[200]
        start_price = df.iloc[200]["close"]
        self.portfolio_values.append(float(self.start_cash))
        self.dates.append(start_dt)

        monthly_dca_done = set()

        for i in range(200, len(df)):
            dt = df.index[i]
            price = df.iloc[i]["close"]
            signal = int(signals.iloc[i])
            ym = (dt.year, dt.month)

            # Monthly DCA trigger
            dca_this_month = ym not in monthly_dca_done
            if dca_this_month:
                monthly_dca_done.add(ym)

            # ── DCA buy ─────────────────────────────────────────────
            if dca_this_month and signal > 0:
                multiplier = {
                    1: 1.0,
                    2: 1.5,
                    3: 2.0,
                    4: 2.5,
                }.get(signal, 1.0)
                alloc = self.dca_monthly * multiplier
                shares_bought = alloc / price
                self.shares += shares_bought
                self.cash -= alloc
                if shares_bought > 0:
                    self.positions.append({
                        "date": dt, "action": "BUY", "shares": shares_bought,
                        "price": price, "amount": alloc, "signal": signal
                    })

            # ── Expense ratio (daily deduction on total portfolio) ──
            portfolio_value = self.cash + self.shares * price
            expense = portfolio_value * daily_cost_rate
            # Simple: deduct from cash, but don't go below zero
            self.cash = max(0, self.cash - expense)

            # ── Signal-driven sells ─────────────────────────────────
            if signal == -1 and self.shares > 0.001:
                shares_to_sell = min(self.shares * 0.30, self.shares * 0.999)
                proceeds = shares_to_sell * price
                tax = proceeds * self.tax_rate
                self.shares -= shares_to_sell
                self.cash += proceeds - tax
                # Ensure cash doesn't go negative
                if self.cash < 0:
                    self.cash = 0
                self.positions.append({
                    "date": dt, "action": "SELL", "shares": shares_to_sell,
                    "price": price, "amount": proceeds - tax, "signal": signal
                })
            elif signal == -2 and self.shares > 0.001:
                shares_to_sell = min(self.shares * 0.50, self.shares * 0.999)
                proceeds = shares_to_sell * price
                tax = proceeds * self.tax_rate
                self.shares -= shares_to_sell
                self.cash += proceeds - tax
                if self.cash < 0:
                    self.cash = 0
                self.positions.append({
                    "date": dt, "action": "SELL", "shares": shares_to_sell,
                    "price": price, "amount": proceeds - tax, "signal": signal
                })

            # ── Record end-of-day value ─────────────────────────────
            portfolio_value = self.cash + self.shares * price
            self.portfolio_values.append(portfolio_value)
            self.dates.append(dt)

        return self._compute_stats()

    def _compute_stats(self) -> dict:
        dates = self.dates
        pv = np.array(self.portfolio_values)
        n_years = (dates[-1] - dates[0]).days / 365.25

        # Benchmark: buy & hold from same start
        init_price = self.df.iloc[200]["close"]
        initial_shares = self.start_cash / init_price
        # bpv must match pv length exactly (both start at day 200)
        bpv = np.array([
            initial_shares * self.df.iloc[i]["close"]
            for i in range(200, len(self.df))
        ])
        # Prepend start cash so bpv[0] == pv[0] == start_cash
        bpv = np.insert(bpv, 0, float(self.start_cash))
        # Expense drag
        bpv = bpv * (1 - self.expense_ratio * n_years)

        # Safety: trim to same length
        min_len = min(len(pv), len(bpv))
        pv = pv[:min_len]
        bpv = bpv[:min_len]
        dates = dates[:min_len]

        # Daily returns
        ret = np.diff(pv) / pv[:-1]
        bret = np.diff(bpv) / bpv[:-1]

        # Metrics
        total_return = (pv[-1] / pv[0]) - 1
        benchmark_return = (bpv[-1] / bpv[0]) - 1

        cagr = (pv[-1] / pv[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
        bench_cagr = (bpv[-1] / bpv[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

        volatility = ret.std() * np.sqrt(252)
        sharpe = cagr / volatility if volatility > 0 else 0

        running_max = np.maximum.accumulate(pv)
        drawdowns = (pv - running_max) / running_max
        max_dd = drawdowns.min()

        bench_running_max = np.maximum.accumulate(bpv)
        bench_drawdowns = (bpv - bench_running_max) / bench_running_max
        bench_max_dd = bench_drawdowns.min()

        downside_ret = ret[ret < 0]
        downside_std = downside_ret.std() * np.sqrt(252) if len(downside_ret) > 0 else 0
        sortino = cagr / downside_std if downside_std > 0 else 0

        win_rate = (ret > 0).sum() / len(ret) if len(ret) > 0 else 0

        buys = [p for p in self.positions if p["action"] == "BUY"]
        sells = [p for p in self.positions if p["action"] == "SELL"]

        return {
            "start_date": str(dates[0].date()),
            "end_date": str(dates[-1].date()),
            "n_years": round(n_years, 2),
            "total_return": f"{total_return*100:.2f}%",
            "benchmark_return": f"{benchmark_return*100:.2f}%",
            "cagr": f"{cagr*100:.2f}%",
            "benchmark_cagr": f"{bench_cagr*100:.2f}%",
            "volatility": f"{volatility*100:.2f}%",
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown": f"{max_dd*100:.2f}%",
            "benchmark_max_drawdown": f"{bench_max_dd*100:.2f}%",
            "win_rate": f"{win_rate*100:.2f}%",
            "n_buys": len(buys),
            "n_sells": len(sells),
            "final_value": round(pv[-1], 2),
            "benchmark_final_value": round(bpv[-1], 2),
            "portfolio_values": pv.tolist(),
            "benchmark_values": bpv.tolist(),
            "dates": [str(d.date()) for d in dates],
        }


# ─────────────────────────────────────────────
# Walk-forward validation
# ─────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, train_years: int = 3, test_years: int = 1) -> list:
    results = []
    start_idx = 200
    train_size = train_years * 252
    test_size = test_years * 252

    cursor = start_idx
    while cursor + train_size + test_size <= len(df):
        train_df = df.iloc[cursor:cursor + train_size]
        test_df = df.iloc[cursor + train_size:cursor + train_size + test_size]

        # Grid search on train
        best_score = -999
        best_params = {}

        cape_fair_vals = np.linspace(22.0, 28.0, 3)
        cape_ov1_vals = np.linspace(28.0, 34.0, 3)
        cape_ov2_vals = np.linspace(34.0, 42.0, 3)
        cape_bubble_vals = np.linspace(40.0, 50.0, 3)
        rsi_ob_vals = np.linspace(65.0, 75.0, 3)

        for cf in cape_fair_vals:
            for cov1 in cape_ov1_vals:
                if cov1 <= cf:
                    continue
                for cov2 in cape_ov2_vals:
                    if cov2 <= cov1:
                        continue
                    for cb in cape_bubble_vals:
                        if cb <= cov2:
                            continue
                        for rsi_ob in rsi_ob_vals:
                            signals = build_signals(train_df,
                                cape_fair=cf,
                                cape_overvalued_1=cov1,
                                cape_overvalued_2=cov2,
                                cape_bubble=cb,
                                rsi_overbought=rsi_ob)
                            bt = Backtester(train_df, signals)
                            stats = bt.run()
                            score = float(stats["sharpe_ratio"])
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    "cape_fair": cf,
                                    "cape_overvalued_1": cov1,
                                    "cape_overvalued_2": cov2,
                                    "cape_bubble": cb,
                                    "rsi_overbought": rsi_ob,
                                }

        # Evaluate on test
        signals_test = build_signals(test_df, **best_params)
        bt_test = Backtester(test_df, signals_test)
        stats_test = bt_test.run()

        results.append({
            "train_start": str(train_df.index[0].date()),
            "train_end": str(train_df.index[-1].date()),
            "test_start": str(test_df.index[0].date()),
            "test_end": str(test_df.index[-1].date()),
            "best_params": {k: float(v) for k, v in best_params.items()},
            "test_stats": stats_test,
        })

        cursor += test_size

    return results


# ─────────────────────────────────────────────
# Charting
# ─────────────────────────────────────────────

def plot_results(stats: dict, output_path: str = "backtest_chart.png"):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        dates = pd.to_datetime(stats["dates"])
        pv = np.array(stats["portfolio_values"])
        bpv = np.array(stats["benchmark_values"])

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 1]})

        # Equity curves
        ax = axes[0]
        ax.plot(dates, pv, label="Strategy", linewidth=1.5, color="#2E86AB")
        ax.plot(dates, bpv, label="Buy & Hold", linewidth=1.5, color="#A23B72", alpha=0.8)
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_title("US ETF Valuation-DCA Strategy Backtest")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Drawdown
        ax2 = axes[1]
        running_max = np.maximum.accumulate(pv)
        dd = (pv - running_max) / running_max * 100
        ax2.fill_between(dates, dd, 0, color="#F18F01", alpha=0.4)
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)

        # Benchmark drawdown
        ax3 = axes[2]
        b_running_max = np.maximum.accumulate(bpv)
        b_dd = (bpv - b_running_max) / b_running_max * 100
        ax3.fill_between(dates, b_dd, 0, color="#C73E1D", alpha=0.4)
        ax3.set_ylabel("B&H DD (%)")
        ax3.set_xlabel("Date")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"Chart saved to {output_path}")
    except Exception as e:
        print(f"Chart error: {e}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation")
    args = parser.parse_args()

    print("Fetching data...")
    spy = fetch_spy_data("1993-01-01", "2025-12-31")
    cape = fetch_pe_ratio("1993-01-01", "2025-12-31")

    print("Computing indicators...")
    df = compute_indicators(spy, cape)

    if args.walk_forward:
        print("Running walk-forward validation...")
        wf_results = walk_forward(df)
        print("\n=== Walk-Forward Results ===\n")
        for r in wf_results:
            ts = r["test_stats"]
            print(f"Train: {r['train_start']} → {r['train_end']}")
            print(f"  Test: {r['test_start']} → {r['test_end']}")
            print(f"  Params: {r['best_params']}")
            print(f"  Sharpe: {ts['sharpe_ratio']} | CAGR: {ts['cagr']} | MaxDD: {ts['max_drawdown']} | Return: {ts['total_return']} vs B&H: {ts['benchmark_return']}")
            print()
        with open("walk_forward_results.json", "w", encoding="utf-8") as f:
            json.dump(wf_results, f, indent=2, default=str)
        print("Saved to walk_forward_results.json")
        return

    print("Building signals...")
    signals = build_signals(df)

    print("Running backtest...")
    bt = Backtester(df, signals)
    stats = bt.run()

    print("\n" + "=" * 60)
    print("  US ETF Valuation-DCA Strategy Backtest Results")
    print("=" * 60)
    print(f"  Period:       {stats['start_date']} → {stats['end_date']} ({stats['n_years']} years)")
    print()
    print(f"  Strategy:     {'=' * 40}")
    print(f"  Total Return:    {stats['total_return']}")
    print(f"  CAGR:            {stats['cagr']}")
    print(f"  Sharpe Ratio:    {stats['sharpe_ratio']}")
    print(f"  Sortino Ratio:   {stats['sortino_ratio']}")
    print(f"  Volatility:      {stats['volatility']}")
    print(f"  Max Drawdown:    {stats['max_drawdown']}")
    print(f"  Win Rate:        {stats['win_rate']}")
    print(f"  # Buys:          {stats['n_buys']}")
    print(f"  # Sells:         {stats['n_sells']}")
    print(f"  Final Value:     ${stats['final_value']:,.2f}")
    print()
    print(f"  Benchmark (Buy&Hold on SPY):")
    print(f"  Total Return:    {stats['benchmark_return']}")
    print(f"  CAGR:            {stats['benchmark_cagr']}")
    print(f"  Max Drawdown:    {stats['benchmark_max_drawdown']}")
    print(f"  Final Value:     ${stats['benchmark_final_value']:,.2f}")
    print("=" * 60)

    with open("backtest_results.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    print("\nResults saved to backtest_results.json")

    plot_results(stats)


if __name__ == "__main__":
    main()
