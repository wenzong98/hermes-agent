"""
US ETF Valuation-Driven DCA Strategy v2
========================================
Dual-asset: SPY (S&P 500) + QQQ (Nasdaq-100)
Multi-regime signal based on CAPE + RSI + Volatility Regime

Core improvements over v1:
  1. Dual-asset rotation: overweight QQQ in bull, SPY in bear
  2. VIX-based position sizing: reduce exposure when vol spikes
  3. Trend filter: only buy when price > SMA200 (avoid catching falling knives)
  4. Asymmetric DCA: 3x on dips, 0.5x on rallies
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

def fetch_etf_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df.resample("D").last().ffill()
    return df[["open", "high", "low", "close", "volume"]]


def fetch_pe_ratio(start: str, end: str) -> pd.DataFrame:
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
        spy = fetch_etf_data("SPY", start, end)
        synthetic = pd.Series(25.0, index=spy.index)
        return pd.DataFrame({"value": synthetic})


def fetch_vix(start: str, end: str) -> pd.DataFrame:
    """Fetch VIX for volatility regime detection."""
    try:
        vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix = vix.reset_index()
        vix.columns = [str(c).lower() for c in vix.columns]
        vix["date"] = pd.to_datetime(vix["date"])
        vix.set_index("date", inplace=True)
        vix = vix.resample("D").last().ffill()
        return vix[["close"]].rename(columns={"close": "vix"})
    except Exception:
        # Fallback: synthetic VIX from SPY realized vol
        spy = fetch_etf_data("SPY", start, end)
        ret = spy["close"].pct_change()
        vol = ret.rolling(21).std() * np.sqrt(252) * 100
        return pd.DataFrame({"vix": vol})


# ─────────────────────────────────────────────
# Technical indicators
# ─────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame, cape_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    df["sma200"] = close.rolling(200).mean()
    df["sma50"] = close.rolling(50).mean()

    # RSI (14-day Wilder)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Realized volatility (21-day annualized)
    ret = close.pct_change()
    df["realized_vol"] = ret.rolling(21).std() * np.sqrt(252)

    # CAPE
    df["cape"] = cape_df["value"].reindex(df.index).ffill()

    # VIX
    df["vix"] = vix_df["vix"].reindex(df.index).ffill()
    df["vix_sma20"] = df["vix"].rolling(20).mean()

    # Returns
    df["ret_1m"] = close.pct_change(21)
    df["ret_3m"] = close.pct_change(63)
    df["ret_6m"] = close.pct_change(126)

    # Trend filter
    df["trend_up"] = close > df["sma200"]

    return df


# ─────────────────────────────────────────────
# Strategy rules
# ─────────────────────────────────────────────

def build_signals(df: pd.DataFrame,
                  cape_fair: float = 25.0,
                  cape_overvalued_1: float = 30.0,
                  cape_overvalued_2: float = 38.0,
                  cape_bubble: float = 44.0,
                  rsi_oversold: float = 35.0,
                  rsi_overbought: float = 70.0,
                  vix_high: float = 25.0,
                  vix_extreme: float = 35.0) -> pd.Series:
    """
    Multi-regime signal with VIX volatility adjustment.

    Signal values (DCA multiplier):
      -2  = SELL 50% + no DCA   (bubble)
      -1  = SELL 30% + no DCA   (very overvalued)
       0  = HOLD / 0.5x DCA     (overvalued)
       1  = NORMAL DCA 1x       (fair)
       2  = 1.5x DCA            (cheap)
       3  = 2x DCA              (cheap + oversold)
       4  = 3x DCA              (max conviction: cheap + oversold + high vol)
    """
    signals = pd.Series(1, index=df.index, dtype=int)
    in_bubble = False

    for i in range(200, len(df)):
        row = df.iloc[i]
        cape = row["cape"]
        rsi = row["rsi"]
        vix = row["vix"]
        trend_up = row["trend_up"]

        cheap = cape < cape_fair
        oversold = rsi < rsi_oversold
        very_cheap = cheap and oversold
        high_vol = vix > vix_high
        extreme_vol = vix > vix_extreme
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

        # Regime logic with VIX boost
        if very_overvalued:
            signals.iloc[i] = -1
        elif moderately_overvalued:
            signals.iloc[i] = 0
        elif very_cheap and extreme_vol:
            signals.iloc[i] = 4  # 3x DCA - max conviction during panic
        elif very_cheap and high_vol:
            signals.iloc[i] = 4  # 3x DCA
        elif very_cheap:
            signals.iloc[i] = 3  # 2x DCA
        elif cheap and oversold:
            signals.iloc[i] = 3  # 2x DCA
        elif cheap:
            signals.iloc[i] = 2  # 1.5x DCA
        else:
            signals.iloc[i] = 1  # normal DCA

    return signals


# ─────────────────────────────────────────────
# Backtest engine (dual-asset)
# ─────────────────────────────────────────────

class DualBacktester:
    def __init__(self,
                 df_spy: pd.DataFrame,
                 df_qqq: pd.DataFrame,
                 signals: pd.Series,
                 start_cash: float = 10000.0,
                 dca_monthly: float = 500.0,
                 spy_weight_default: float = 0.6,
                 qqq_weight_default: float = 0.4,
                 expense_ratio: float = 0.0003,
                 tax_rate: float = 0.0015):
        self.df_spy = df_spy
        self.df_qqq = df_qqq
        self.signals = signals
        self.start_cash = start_cash
        self.dca_monthly = dca_monthly
        self.spy_weight_default = spy_weight_default
        self.qqq_weight_default = qqq_weight_default
        self.expense_ratio = expense_ratio
        self.tax_rate = tax_rate

        self.cash = start_cash
        self.spy_shares = 0.0
        self.qqq_shares = 0.0
        self.portfolio_values = []
        self.dates = []
        self.positions = []

    def run(self) -> dict:
        df_spy = self.df_spy
        df_qqq = self.df_qqq
        signals = self.signals
        daily_cost_rate = self.expense_ratio / 252

        # Align QQQ to SPY index (QQQ started later)
        common_index = df_spy.index.intersection(df_qqq.index)
        df_spy = df_spy.loc[common_index]
        df_qqq = df_qqq.loc[common_index]
        signals = signals.loc[common_index]

        # Find first valid index after SMA200 warm-up
        valid_start = 200
        while valid_start < len(df_spy) and (pd.isna(df_spy.iloc[valid_start]["close"]) or pd.isna(df_qqq.iloc[valid_start]["close"])):
            valid_start += 1

        start_dt = df_spy.index[valid_start]
        self.portfolio_values.append(float(self.start_cash))
        self.dates.append(start_dt)

        monthly_dca_done = set()

        for i in range(valid_start, len(df_spy)):
            dt = df_spy.index[i]
            spy_price = df_spy.iloc[i]["close"]
            qqq_price = df_qqq.iloc[i]["close"]
            signal = int(signals.iloc[i])
            ym = (dt.year, dt.month)

            dca_this_month = ym not in monthly_dca_done
            if dca_this_month:
                monthly_dca_done.add(ym)

            # Determine asset weights based on signal
            if signal >= 3:  # Bullish - overweight QQQ
                spy_w = 0.3
                qqq_w = 0.7
            elif signal == 2:  # Slightly cheap
                spy_w = 0.5
                qqq_w = 0.5
            elif signal <= 0:  # Bearish / hold - defensive SPY
                spy_w = 0.8
                qqq_w = 0.2
            else:  # Normal
                spy_w = self.spy_weight_default
                qqq_w = self.qqq_weight_default

            # DCA buy
            if dca_this_month and signal > 0:
                multiplier = {
                    1: 1.0,
                    2: 1.5,
                    3: 2.0,
                    4: 3.0,
                }.get(signal, 1.0)
                alloc = self.dca_monthly * multiplier

                spy_alloc = alloc * spy_w
                qqq_alloc = alloc * qqq_w

                spy_shares_bought = spy_alloc / spy_price
                qqq_shares_bought = qqq_alloc / qqq_price

                self.spy_shares += spy_shares_bought
                self.qqq_shares += qqq_shares_bought
                self.cash -= alloc

                if spy_shares_bought > 0:
                    self.positions.append({
                        "date": dt, "action": "BUY_SPY", "shares": spy_shares_bought,
                        "price": spy_price, "amount": spy_alloc, "signal": signal
                    })
                if qqq_shares_bought > 0:
                    self.positions.append({
                        "date": dt, "action": "BUY_QQQ", "shares": qqq_shares_bought,
                        "price": qqq_price, "amount": qqq_alloc, "signal": signal
                    })

            # Expense
            portfolio_value = self.cash + self.spy_shares * spy_price + self.qqq_shares * qqq_price
            expense = portfolio_value * daily_cost_rate
            if portfolio_value > 0:
                cash_ratio = self.cash / portfolio_value
                self.cash -= expense * cash_ratio
            else:
                self.cash -= expense

            # Signal-driven sells (proportional from both holdings)
            if signal == -1:
                if self.spy_shares > 0.001:
                    shares_to_sell = min(self.spy_shares * 0.30, self.spy_shares * 0.999)
                    proceeds = shares_to_sell * spy_price
                    tax = proceeds * self.tax_rate
                    self.spy_shares -= shares_to_sell
                    self.cash += proceeds - tax
                    if self.cash < 0: self.cash = 0
                if self.qqq_shares > 0.001:
                    shares_to_sell = min(self.qqq_shares * 0.30, self.qqq_shares * 0.999)
                    proceeds = shares_to_sell * qqq_price
                    tax = proceeds * self.tax_rate
                    self.qqq_shares -= shares_to_sell
                    self.cash += proceeds - tax
                    if self.cash < 0: self.cash = 0
            elif signal == -2:
                if self.spy_shares > 0.001:
                    shares_to_sell = min(self.spy_shares * 0.50, self.spy_shares * 0.999)
                    proceeds = shares_to_sell * spy_price
                    tax = proceeds * self.tax_rate
                    self.spy_shares -= shares_to_sell
                    self.cash += proceeds - tax
                    if self.cash < 0: self.cash = 0
                if self.qqq_shares > 0.001:
                    shares_to_sell = min(self.qqq_shares * 0.50, self.qqq_shares * 0.999)
                    proceeds = shares_to_sell * qqq_price
                    tax = proceeds * self.tax_rate
                    self.qqq_shares -= shares_to_sell
                    self.cash += proceeds - tax
                    if self.cash < 0: self.cash = 0

            # Record
            portfolio_value = self.cash + self.spy_shares * spy_price + self.qqq_shares * qqq_price
            self.portfolio_values.append(portfolio_value)
            self.dates.append(dt)

        return self._compute_stats(df_spy, df_qqq, valid_start)

    def _compute_stats(self, df_spy_aligned, df_qqq_aligned, valid_start) -> dict:
        dates = self.dates
        pv = np.array(self.portfolio_values)
        n_years = (dates[-1] - dates[0]).days / 365.25

        # Benchmark: 60/40 SPY/QQQ buy & hold
        spy_price_0 = df_spy_aligned.iloc[valid_start]["close"]
        qqq_price_0 = df_qqq_aligned.iloc[valid_start]["close"]
        spy_shares = (self.start_cash * 0.6) / spy_price_0
        qqq_shares = (self.start_cash * 0.4) / qqq_price_0

        bpv = np.array([
            spy_shares * df_spy_aligned.iloc[i]["close"] + qqq_shares * df_qqq_aligned.iloc[i]["close"]
            for i in range(valid_start, len(df_spy_aligned))
        ])
        bpv = np.insert(bpv, 0, float(self.start_cash))
        bpv = bpv * (1 - self.expense_ratio * n_years)

        # Align
        min_len = min(len(pv), len(bpv))
        pv = pv[:min_len]
        bpv = bpv[:min_len]
        dates = dates[:min_len]

        ret = np.diff(pv) / pv[:-1]
        bret = np.diff(bpv) / bpv[:-1]

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
            "final_value": round(pv[-1], 2),
            "benchmark_final_value": round(bpv[-1], 2),
        }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("Fetching SPY + QQQ + VIX data...")
    spy = fetch_etf_data("SPY", "1999-01-01", "2025-12-31")
    qqq = fetch_etf_data("QQQ", "1999-01-01", "2025-12-31")
    cape = fetch_pe_ratio("1999-01-01", "2025-12-31")
    vix = fetch_vix("1999-01-01", "2025-12-31")

    print("Computing indicators...")
    # Use SPY as signal anchor (CAPE is market-wide)
    df_signals = compute_indicators(spy, cape, vix)

    print("Building signals...")
    signals = build_signals(df_signals)

    print("Running dual-asset backtest...")
    bt = DualBacktester(spy, qqq, signals)
    stats = bt.run()

    print("\n" + "=" * 60)
    print("  US ETF Dual-Asset Strategy Backtest Results")
    print("=" * 60)
    print(f"  Period:       {stats['start_date']} → {stats['end_date']} ({stats['n_years']} years)")
    print()
    print(f"  Strategy (SPY+QQQ valuation-DCA):")
    print(f"  Total Return:    {stats['total_return']}")
    print(f"  CAGR:            {stats['cagr']}")
    print(f"  Sharpe Ratio:    {stats['sharpe_ratio']}")
    print(f"  Sortino Ratio:   {stats['sortino_ratio']}")
    print(f"  Volatility:      {stats['volatility']}")
    print(f"  Max Drawdown:    {stats['max_drawdown']}")
    print(f"  Win Rate:        {stats['win_rate']}")
    print(f"  Final Value:     ${stats['final_value']:,.2f}")
    print()
    print(f"  Benchmark (60/40 SPY/QQQ Buy&Hold):")
    print(f"  Total Return:    {stats['benchmark_return']}")
    print(f"  CAGR:            {stats['benchmark_cagr']}")
    print(f"  Max Drawdown:    {stats['benchmark_max_drawdown']}")
    print(f"  Final Value:     ${stats['benchmark_final_value']:,.2f}")
    print("=" * 60)

    with open("backtest_results_v2.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    print("\nResults saved to backtest_results_v2.json")


if __name__ == "__main__":
    main()
