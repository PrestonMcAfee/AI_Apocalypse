import os
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf

# Configuration
TICKERS = [
    "NVDA", "GOOG", "MSFT", "AAPL", "AMZN", 
    "META", "TSLA", "SPCX", "SPY", "QQQ", "DIA"
]
HISTORY_FILE = "history.csv"
HORIZON_YEARS = 2.0
DEFAULT_RISK_FREE_RATE = 0.039  # Fallback if rate data is missing
DIVIDEND_YIELD = 0.005
VRP_MULTIPLIER = 1.12    # Implied volatility premium over realized volatility
SKEW_MULTIPLIER = 1.18   # Downside wing skew multiplier for 50% OTM strike

def generate_monthly_dates(start_year=2021):
    """Generates a list of the 15th of each month from start_year through last month (Sep 2026)."""
    dates = []
    now = datetime.now()
    for year in range(start_year, now.year + 1):
        for month in range(1, 13):
            # Stop before current month so live daily cron handles Oct 2026 onward
            if year == now.year and month >= now.month:
                break
            dates.append(datetime(year, month, 15))
    return dates

def get_historical_2y_rates():
    """
    Fetches daily historical 2-Year US Treasury yields (Yahoo ticker: 2YY=F).
    Note: 2YY=F futures launched in mid-August 2021. For earlier dates (Jan-Jul 2021),
    we match the contemporaneous Federal Reserve 2-Year Constant Maturity rate (~0.15%).
    """
    try:
        print("Downloading historical 2-Year Treasury yields (2YY=F)...")
        treasury = yf.Ticker("2YY=F")
        df_rate = treasury.history(period="6y")
        if not df_rate.empty:
            if df_rate.index.tz is not None:
                df_rate.index = df_rate.index.tz_localize(None)
            rates = df_rate["Close"] / 100.0
            print(f"Loaded {len(rates)} daily Treasury yield records.")
            return rates
    except Exception as e:
        print(f"Warning: Could not fetch dynamic Treasury yields ({e}). Using default {DEFAULT_RISK_FREE_RATE:.3%}.")
    return None

def run_backfill():
    monthly_dates = generate_monthly_dates(start_year=2021)
    rates_series = get_historical_2y_rates()
    rows = []

    print(f"\nBackfilling {len(monthly_dates)} monthly periods across {len(TICKERS)} tickers...")

    for ticker in TICKERS:
        try:
            print(f"Processing {ticker}...")
            tk = yf.Ticker(ticker)
            df = tk.history(period="6y")
            if df.empty:
                print(f"  No historical price data found for {ticker}.")
                continue

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Daily log returns
            df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
            
            # 63 trading days (~3 calendar months) rolling annualized realized volatility
            df["Vol_63d"] = df["Log_Return"].rolling(window=63).std() * np.sqrt(252)

            # Beta selection
            if ticker == "SPY":
                beta = 1.0
            elif ticker == "QQQ":
                beta = 1.15
            elif ticker == "DIA":
                beta = 0.85
            else:
                beta = tk.info.get("beta", 1.2) or 1.2

            count = 0
            for sample_date in monthly_dates:
                # Find closest prior trading day
                sub_df = df[df.index <= sample_date]
                if sub_df.empty or len(sub_df) < 63:
                    continue

                latest_row = sub_df.iloc[-1]
                spot = float(latest_row["Close"])
                realized_vol = float(latest_row["Vol_63d"])

                if np.isnan(realized_vol) or realized_vol <= 0:
                    continue

                # Contemporaneous 2-Year Treasury rate determination
                # For pre-August 2021 (before 2YY=F futures launched), 2Y yield was ~0.15% (0.0015)
                r = 0.0015 if sample_date < datetime(2021, 8, 16) else DEFAULT_RISK_FREE_RATE
                if rates_series is not None and sample_date >= datetime(2021, 8, 16):
                    rate_sub = rates_series[rates_series.index <= sample_date]
                    if not rate_sub.empty and not np.isnan(rate_sub.iloc[-1]):
                        r = float(rate_sub.iloc[-1])

                # Volatility proxy via VRP + downside wing skew
                atm_sigma = realized_vol * VRP_MULTIPLIER
                wing_sigma = atm_sigma * SKEW_MULTIPLIER

                # Black-Scholes Risk-Neutral Probability Q(S_T <= K) (Unclipped, unskewed by 0.85)
                T = HORIZON_YEARS
                target_strike = round(spot / 2.0, 2)
                F = spot * np.exp((r - DIVIDEND_YIELD) * T)

                d1 = (np.log(spot / target_strike) + (r - DIVIDEND_YIELD + 0.5 * wing_sigma**2) * T) / (wing_sigma * np.sqrt(T))
                d2 = d1 - wing_sigma * np.sqrt(T)
                prob_Q = float(norm.cdf(-d2))

                # Analytical Scenario: Lognormal CDF + CAPM Drift (mu = r + beta * ERP - q)
                mu = r + (beta * 0.045) - DIVIDEND_YIELD
                E_ST_P = spot * np.exp(mu * T)
                d2_P = (np.log(E_ST_P / target_strike) - 0.5 * (wing_sigma**2) * T) / (wing_sigma * np.sqrt(T))
                prob_P = float(norm.cdf(-d2_P))

                sample_str = sample_date.strftime("%Y-%m-%d")
                rows.append({
                    "date": sample_str,
                    "ticker": ticker,
                    "spot": round(spot, 2),
                    "target_strike": target_strike,
                    "prob_Q": round(prob_Q * 100, 2),
                    "prob_P": round(prob_P * 100, 2)
                })
                count += 1

            print(f"  Generated {count} monthly data points for {ticker}.")

        except Exception as e:
            print(f"  Error backfilling {ticker}: {e}")

    backfill_df = pd.DataFrame(rows)
    backfill_df = backfill_df.sort_values(by=["date", "ticker"]).reset_index(drop=True)
    backfill_df.to_csv(HISTORY_FILE, index=False)
    print(f"\nDone! Wrote clean backfilled history with {len(backfill_df)} records to {HISTORY_FILE}.")

if __name__ == "__main__":
    run_backfill()
