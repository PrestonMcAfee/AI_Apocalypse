"""
Historical backfill for both metrics, calibrated to the live option-implied method.

  history.csv   : probability of finishing <= 50% of the current price
  history52.csv : probability of finishing <= 50% of the 52-week high

Yahoo does not store historical option chains, so history is estimated from
realized volatility, then scaled per ticker so that it matches the live
option-based probabilities where the two overlap.

  1. Two-year volatility proxy. Two-year implied volatility moves much less
     than short-window realized volatility, because volatility mean-reverts:
         sigma_proxy^2 = W_SHORT * RV_63d^2 + (1 - W_SHORT) * EWMA_var
     EWMA_var is an exponentially weighted average of squared daily returns
     with a ~6-month half-life. Unlike a fixed 2-year window, a shock (e.g.
     March 2020) fades gradually rather than staying at full weight for two
     years and then dropping out all at once.
  2. Level calibration. Each live row stores sigma_eff: the single lognormal
     volatility that reproduces the option-implied probability at K = S/2.
     multiplier = median(sigma_eff / sigma_proxy) over live days, per ticker.
  3. Skew calibration. Live runs also record sigma_eff at K/S = 0.3 ... 0.9
     (skew_curve.csv). rel(m) = median(sigma_eff(m) / sigma_eff(0.5)), so the
     52-week strike, which can sit anywhere relative to spot, gets the right
     effective volatility. rel(0.5) = 1 by construction.
  4. History. Q = N(-d2) with sigma = multiplier * sigma_proxy * rel(K/S).

Sampling: the 15th of each month through DAILY_FROM, then every trading day
up to the first live option-based row. Live rows (method == "options") are
never modified. Dates where a ticker has under MIN_HISTORY_DAYS of trading
history are skipped, so recent IPOs start with the live data.

Rerun after live data accumulates to refine the calibration.
"""
import os
import math
from datetime import datetime, date

import numpy as np
import pandas as pd
import yfinance as yf

from build_site import (
    TICKERS, HISTORY_FILE, HISTORY52_FILE, SKEW_FILE, SKEW_GRID, HORIZON_YEARS,
    DEFAULT_RISK_FREE_RATE, EQUITY_RISK_PREMIUM, INDEX_BETAS, lognormal_prob_below,
)

START_YEAR = 2021
DAILY_FROM = date(2026, 10, 1)      # daily sampling from here until live data begins
PRICE_HISTORY_START = "2018-06-01"  # warm-up for the EWMA and 52-week high
SHORT_WIN = 63
EWMA_HALFLIFE = 126                 # trading days (~6 months)
W_SHORT = 0.30
MIN_HISTORY_DAYS = 252
DEFAULT_MULTIPLIER = 1.12 * 1.18    # old fixed factors; used only if a ticker has no live rows
PRE_FUTURES_RATE = 0.0015           # 2Y yield before 2YY=F futures launched (Aug 2021)
FUTURES_START = pd.Timestamp("2021-08-16")
COLS = ["date", "ticker", "spot", "target_strike", "prob_Q", "prob_P", "sigma_eff", "method"]
COLS52 = ["date", "ticker", "spot", "high_52w", "target_strike", "prob_Q", "prob_P", "method"]


def sample_dates(first_live):
    monthly = [pd.Timestamp(date(y, m, 15))
               for y in range(START_YEAR, first_live.year + 1) for m in range(1, 13)
               if date(y, m, 15) < min(DAILY_FROM, first_live)]
    daily = list(pd.bdate_range(DAILY_FROM, first_live - pd.Timedelta(days=1)))
    return monthly, daily


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
    var_short = lr.rolling(SHORT_WIN).var() * 252
    var_long = (lr ** 2).ewm(halflife=EWMA_HALFLIFE, min_periods=SHORT_WIN).mean() * 252
    df["sigma_proxy"] = np.sqrt(W_SHORT * var_short + (1 - W_SHORT) * var_long)
    df["n_days"] = np.arange(1, len(df) + 1)
    df["high_52w"] = df["High"].rolling(252, min_periods=1).max()
    divs = df["Dividends"] if "Dividends" in df else pd.Series(0.0, index=df.index)
    df["q"] = (divs.rolling("365D").sum() / df["Close"]).fillna(0.0)
    return df


def row_on(df, d):
    sub = df[df.index <= d]
    return None if sub.empty else sub.iloc[-1]


def skew_function(skew_df, ticker):
    """rel(m) = sigma_eff(m) / sigma_eff(0.5), median across live days; flat outside the grid."""
    sub = skew_df[skew_df["ticker"] == ticker] if skew_df is not None else pd.DataFrame()
    if sub.empty:
        return (lambda m: 1.0), False
    piv = sub.pivot_table(index="date", columns="moneyness", values="sigma_eff")
    if 0.5 not in piv.columns:
        return (lambda m: 1.0), False
    rel = piv.div(piv[0.5], axis=0).median().dropna().sort_index()
    xs, ys = rel.index.values.astype(float), rel.values.astype(float)
    return (lambda m: float(np.interp(m, xs, ys))), True


def read_live(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[df.get("method", pd.Series(dtype=str)) == "options"].copy()


def run_backfill():
    live = read_live(HISTORY_FILE)
    live52 = read_live(HISTORY52_FILE)
    skew_df = pd.read_csv(SKEW_FILE) if os.path.exists(SKEW_FILE) else None

    if live.empty:
        first_live = pd.Timestamp(datetime.now().date())
        print("WARNING: no option-based live rows yet; using default multiplier for every ticker.")
    else:
        first_live = pd.Timestamp(live["date"].min())
    monthly, daily = sample_dates(first_live.date())
    print(f"First live date: {first_live.date()} | {len(monthly)} monthly + {len(daily)} daily backfill dates")

    rates = get_rates()
    T = HORIZON_YEARS
    rows, rows52, calib = [], [], []

    for ticker in TICKERS:
        try:
            tk = yf.Ticker(ticker)
            df = ticker_frame(tk)
            if df.empty:
                print(f"  {ticker}: no price history.")
                continue

            ratios = []
            for _, lr in live[live["ticker"] == ticker].iterrows():
                pr = row_on(df, pd.Timestamp(lr["date"]))
                if pr is not None and np.isfinite(pr["sigma_proxy"]) and np.isfinite(lr.get("sigma_eff", np.nan)):
                    ratios.append(lr["sigma_eff"] / pr["sigma_proxy"])
            mult = float(np.median(ratios)) if ratios else DEFAULT_MULTIPLIER
            rel, has_skew = skew_function(skew_df, ticker)
            calib.append((ticker, mult, len(ratios), has_skew, rel))

            beta = INDEX_BETAS.get(ticker)
            if beta is None:
                try:
                    beta = tk.info.get("beta", 1.2) or 1.2
                except Exception:
                    beta = 1.2
            phys_shift = math.exp(-beta * EQUITY_RISK_PREMIUM * T)

            n = 0
            for d in monthly + [d for d in daily if d in df.index]:
                pr = row_on(df, d)
                if pr is None or pr["n_days"] < MIN_HISTORY_DAYS or not np.isfinite(pr["sigma_proxy"]):
                    continue
                spot = float(pr["Close"])
                r = rate_on(rates, d)
                F = spot * math.exp((r - float(pr["q"])) * T)
                base = mult * float(pr["sigma_proxy"])
                ds = d.strftime("%Y-%m-%d")

                # 50% of current price (m = 0.5, rel = 1)
                K = spot / 2.0
                sig = base * rel(0.5)
                rows.append({
                    "date": ds, "ticker": ticker, "spot": round(spot, 2), "target_strike": round(K, 2),
                    "prob_Q": round(lognormal_prob_below(F, K, T, sig) * 100, 2),
                    "prob_P": round(lognormal_prob_below(F, K * phys_shift, T, sig) * 100, 2),
                    "sigma_eff": round(sig, 4), "method": "realized_vol_calibrated",
                })

                # 50% of 52-week high
                H = float(pr["high_52w"])
                K52 = H / 2.0
                sig52 = base * rel(K52 / spot)
                rows52.append({
                    "date": ds, "ticker": ticker, "spot": round(spot, 2), "high_52w": round(H, 2),
                    "target_strike": round(K52, 2),
                    "prob_Q": round(lognormal_prob_below(F, K52, T, sig52) * 100, 2),
                    "prob_P": round(lognormal_prob_below(F, K52 * phys_shift, T, sig52) * 100, 2),
                    "method": "realized_vol_calibrated",
                })
                n += 1
            print(f"  {ticker}: multiplier {mult:.3f} ({len(ratios)} live days), "
                  f"skew {'calibrated' if has_skew else 'NONE'} -> {n} backfill dates")
        except Exception as e:
            print(f"  Error backfilling {ticker}: {e}")

    def write(path, new_rows, live_rows, cols):
        out = pd.concat([pd.DataFrame(new_rows, columns=cols), live_rows], ignore_index=True)
        out = out.drop_duplicates(subset=["date", "ticker"], keep="last")
        out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
        out = out[[c for c in cols if c in out.columns]]
        out.to_csv(path, index=False)
        print(f"Wrote {len(out)} rows to {path}.")

    print()
    write(HISTORY_FILE, rows, live, COLS)
    write(HISTORY52_FILE, rows52, live52, COLS52)

    print("\nCalibration: level multiplier, and effective-vol ratio rel(K/S) vs K/S = 0.5")
    print("  ticker  mult   " + "  ".join(f"{m:>4}" for m in SKEW_GRID))
    for t, m, k, has_skew, rel in calib:
        rels = "  ".join(f"{rel(x):4.2f}" for x in SKEW_GRID) if has_skew else "(no skew data)"
        print(f"  {t:6s} {m:5.3f}  {rels}   [{k} live days]")


if __name__ == "__main__":
    run_backfill()
