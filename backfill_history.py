"""
Historical backfill, calibrated to the live option-implied method.

Yahoo does not store historical option chains, so history is estimated from
realized volatility, then scaled per ticker so that it matches the live
option-based probabilities where the two overlap.

  1. Two-year volatility proxy. Two-year implied volatility moves much less
     than short-window realized volatility, because volatility mean-reverts.
     The proxy blends recent and long-run realized variance:
         sigma_proxy^2 = W_SHORT * RV_63d^2 + (1 - W_SHORT) * RV_504d^2
     (W_SHORT = 0.3 corresponds to a volatility half-life of roughly 6 months.)
  2. Calibration. Each live row stores sigma_eff: the single lognormal
     volatility that reproduces the option-implied probability (it absorbs
     the volatility risk premium AND the put skew at the 50% strike).
     For each ticker, multiplier = median(sigma_eff / sigma_proxy) over live days.
  3. History. Q = N(-d2) with sigma = multiplier * sigma_proxy, so the
     historical series lands on the same level as the live one at the seam.

Sampling: the 15th of each month through DAILY_FROM, then every trading day
up to the first live option-based row. Live rows (method == "options") are
never modified. Tickers with less than MIN_HISTORY_DAYS of trading history at
a date are skipped for that date (so recent IPOs start with live data).

Run this after build_site.py has produced some live rows; rerun occasionally
as live data accumulates to refine the calibration.
"""
import math
from datetime import datetime, date

import numpy as np
import pandas as pd
import yfinance as yf

from build_site import (
    TICKERS, HISTORY_FILE, HORIZON_YEARS, DEFAULT_RISK_FREE_RATE,
    EQUITY_RISK_PREMIUM, INDEX_BETAS, lognormal_prob_below,
)

START_YEAR = 2021
DAILY_FROM = date(2026, 10, 1)      # daily sampling from here until live data begins
PRICE_HISTORY_START = "2018-06-01"  # enough to fill the 504-day window by 2021
SHORT_WIN, LONG_WIN = 63, 504
W_SHORT = 0.30
MIN_HISTORY_DAYS = 252
DEFAULT_MULTIPLIER = 1.12 * 1.18    # old fixed factors; used only if a ticker has no live rows
PRE_FUTURES_RATE = 0.0015           # 2Y yield before 2YY=F futures launched (Aug 2021)
FUTURES_START = pd.Timestamp("2021-08-16")


def sample_dates(first_live):
    dates = []
    for y in range(START_YEAR, first_live.year + 1):
        for m in range(1, 13):
            d = date(y, m, 15)
            if d < min(DAILY_FROM, first_live):
                dates.append(pd.Timestamp(d))
    daily = pd.bdate_range(DAILY_FROM, first_live - pd.Timedelta(days=1))
    return dates, list(daily)


def get_rates():
    try:
        df = yf.Ticker("2YY=F").history(start="2021-01-01")
        if not df.empty:
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df["Close"] / 100.0
    except Exception as e:
        print(f"Warning: no 2Y yield history ({e}); using {DEFAULT_RISK_FREE_RATE:.2%}.")
    return None


def rate_on(rates, d):
    if d < FUTURES_START:
        return PRE_FUTURES_RATE
    if rates is not None:
        sub = rates[rates.index <= d].dropna()
        if not sub.empty:
            return float(sub.iloc[-1])
    return DEFAULT_RISK_FREE_RATE


def ticker_frame(tk):
    df = tk.history(start=PRICE_HISTORY_START, auto_adjust=False)
    if df.empty:
        return df
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    adj = df["Adj Close"] if "Adj Close" in df else df["Close"]
    lr = np.log(adj / adj.shift(1))
    rv_s = lr.rolling(SHORT_WIN).std() * np.sqrt(252)
    rv_l = lr.rolling(LONG_WIN, min_periods=MIN_HISTORY_DAYS).std() * np.sqrt(252)
    df["sigma_proxy"] = np.sqrt(W_SHORT * rv_s ** 2 + (1 - W_SHORT) * rv_l ** 2)
    df["n_days"] = np.arange(1, len(df) + 1)
    # Trailing 12-month dividend yield on the actual (unadjusted) close
    divs = df["Dividends"] if "Dividends" in df else pd.Series(0.0, index=df.index)
    df["div_12m"] = divs.rolling("365D").sum()
    df["q"] = (df["div_12m"] / df["Close"]).fillna(0.0)
    return df


def row_on(df, d):
    sub = df[df.index <= d]
    return None if sub.empty else sub.iloc[-1]


def run_backfill():
    hist = pd.read_csv(HISTORY_FILE)
    if "method" not in hist.columns:
        hist["method"] = np.nan
    live = hist[hist["method"] == "options"].copy()
    if live.empty:
        first_live = pd.Timestamp(datetime.now().date())
        print("WARNING: no option-based live rows yet; using default multiplier for every ticker.")
    else:
        first_live = pd.Timestamp(live["date"].min())
    monthly, daily = sample_dates(first_live.date())
    print(f"First live option-based date: {first_live.date()}  |  "
          f"{len(monthly)} monthly + {len(daily)} daily backfill dates")

    rates = get_rates()
    T = HORIZON_YEARS
    rows, calib = [], []

    for ticker in TICKERS:
        try:
            tk = yf.Ticker(ticker)
            df = ticker_frame(tk)
            if df.empty:
                print(f"  {ticker}: no price history.")
                continue

            # ---- calibration on live overlap ----
            ratios = []
            for _, lr in live[live["ticker"] == ticker].iterrows():
                pr = row_on(df, pd.Timestamp(lr["date"]))
                if pr is not None and np.isfinite(pr["sigma_proxy"]) and np.isfinite(lr.get("sigma_eff", np.nan)):
                    ratios.append(lr["sigma_eff"] / pr["sigma_proxy"])
            mult = float(np.median(ratios)) if ratios else DEFAULT_MULTIPLIER
            calib.append((ticker, mult, len(ratios)))

            beta = INDEX_BETAS.get(ticker)
            if beta is None:
                try:
                    beta = tk.info.get("beta", 1.2) or 1.2
                except Exception:
                    beta = 1.2

            n = 0
            daily_trading = [d for d in daily if d in df.index]   # skip market holidays
            for d in monthly + daily_trading:
                pr = row_on(df, d)
                if pr is None or pr["n_days"] < MIN_HISTORY_DAYS or not np.isfinite(pr["sigma_proxy"]):
                    continue
                spot = float(pr["Close"])
                r = rate_on(rates, d)
                q = float(pr["q"])
                sigma = mult * float(pr["sigma_proxy"])
                F = spot * math.exp((r - q) * T)
                K = spot / 2.0
                prob_Q = lognormal_prob_below(F, K, T, sigma)
                prob_P = lognormal_prob_below(F, K * math.exp(-beta * EQUITY_RISK_PREMIUM * T), T, sigma)
                rows.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "spot": round(spot, 2),
                    "target_strike": round(K, 2),
                    "prob_Q": round(prob_Q * 100, 2),
                    "prob_P": round(prob_P * 100, 2),
                    "sigma_eff": round(sigma, 4),
                    "method": "realized_vol_calibrated",
                })
                n += 1
            print(f"  {ticker}: multiplier {mult:.3f} from {len(ratios)} live days -> {n} backfill rows")
        except Exception as e:
            print(f"  Error backfilling {ticker}: {e}")

    out = pd.concat([pd.DataFrame(rows), live], ignore_index=True)
    out = out.drop_duplicates(subset=["date", "ticker"], keep="last")
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    out = out[["date", "ticker", "spot", "target_strike", "prob_Q", "prob_P", "sigma_eff", "method"]]
    out.to_csv(HISTORY_FILE, index=False)

    print("\nCalibration (sigma_eff / sigma_proxy):")
    for t, m, k in calib:
        print(f"  {t:5s} {m:6.3f}  ({k} live days{'; DEFAULT' if k == 0 else ''})")
    print(f"\nWrote {len(out)} rows to {HISTORY_FILE}.")


if __name__ == "__main__":
    run_backfill()
