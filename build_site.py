import os
import sys
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import yfinance as yf
import jinja2

# Configuration
TICKERS = [
    "NVDA", "GOOG", "MSFT", "AAPL", "AMZN", 
    "META", "TSLA", "SPCX", "ANT", "SPY", "QQQ", "DIA"
]
OUTPUT_FILE = "index.html"
HISTORY_FILE = "history.csv"
TEMPLATE_FILE = "template.html"
HORIZON_YEARS = 2.0  # Constant 2-year forward window
DEFAULT_RISK_FREE_RATE = 0.039  # Fallback ~2-year Treasury yield
DIVIDEND_YIELD = 0.005
VRP_MULTIPLIER = 1.12
SKEW_MULTIPLIER = 1.18


def get_market_today_str():
    """Returns current trading date string (YYYY-MM-DD) in US/Eastern market time."""
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def is_today_already_complete(today_str):
    """
    Checks whether history.csv already contains a complete, valid set of all 12 tickers
    for today's US/Eastern trading date. If so, subsequent hourly runs suspend until tomorrow.
    """
    if not os.path.exists(HISTORY_FILE) or os.path.getsize(HISTORY_FILE) == 0:
        return False
    try:
        df = pd.read_csv(HISTORY_FILE)
        today_rows = df[df["date"] == today_str].dropna(subset=["ticker", "spot", "prob_Q", "prob_P"])
        present_tickers = set(today_rows["ticker"].unique())
        if set(TICKERS).issubset(present_tickers):
            # Ensure none of the recorded probabilities are broken zero/NaN artifacts
            if (today_rows["prob_Q"] > 0.01).all() and (today_rows["spot"] > 0).all():
                return True
    except Exception:
        pass
    return False


def get_current_2y_treasury_rate():
    """Fetches the latest active 2-Year US Treasury yield (Yahoo ticker: 2YY=F). Returns None if unavailable."""
    try:
        tk = yf.Ticker("2YY=F")
        fast_info = getattr(tk, "fast_info", None)
        rate_val = getattr(fast_info, "last_price", None) if fast_info else None

        if rate_val is None:
            hist = tk.history(period="5d")
            if not hist.empty:
                rate_val = float(hist["Close"].iloc[-1])

        if rate_val is not None and 0.1 < rate_val < 20.0:
            return float(rate_val) / 100.0
    except Exception as e:
        print(f"Warning: Could not fetch dynamic 2Y rate ({e}).")
    return None


def bs_price(S, K, T, r, q, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def solve_iv_from_mid(mid_price, S, K, T, r, q, option_type="call"):
    """Computes Black-Scholes implied volatility directly from bid/ask mid-price via Brent's method."""
    disc_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    intrinsic = max(0.0, disc_S - disc_K) if option_type == "call" else max(0.0, disc_K - disc_S)
    if mid_price <= intrinsic + 1e-3:
        return None
    try:
        iv = brentq(
            lambda s: bs_price(S, K, T, r, q, s, option_type) - mid_price,
            0.08, 1.80, xtol=1e-5
        )
        return float(iv) if 0.10 <= iv <= 1.50 else None
    except Exception:
        return None


def get_realized_vol_proxy(tk):
    """Computes 63-day annualized realized volatility scaled by VRP for tickers without listed options."""
    try:
        hist = tk.history(period="6mo")
        if len(hist) >= 30:
            log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
            rv = float(log_ret.tail(63).std() * np.sqrt(252))
            if 0.08 <= rv <= 1.50:
                return rv * VRP_MULTIPLIER
    except Exception:
        pass
    return None


def get_valid_atm_volatility(tk, ticker_symbol, spot, r, q=DIVIDEND_YIELD):
    """
    Extracts liquid ATM implied volatility from live LEAPS chains using actual bid/ask mid-prices.
    Rejects stale after-hours quotes (where bid == 0 or Yahoo's IV solver resets to ~5%).
    Returns (atm_iv, source_exp, is_valid, is_rv_proxy).
    """
    try:
        exps = tk.options
        if not exps:
            # If the security genuinely has no listed options on Yahoo Finance, use realized vol proxy
            rv_iv = get_realized_vol_proxy(tk)
            if rv_iv is not None:
                return rv_iv, "RV-Proxy", True, True
            return None, None, False, False

        today_dt = datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)

        # Inspect furthest available expirations (LEAPS)
        for target_exp in reversed(exps[-2:]):
            exp_dt = datetime.strptime(target_exp, "%Y-%m-%d")
            T_exp = (exp_dt - today_dt).days / 365.25
            if T_exp < 0.25:
                continue

            chain = tk.option_chain(target_exp)
            vols = []

            for opt_type, df_chain in [("call", chain.calls), ("put", chain.puts)]:
                df_clean = df_chain.dropna(subset=["strike", "bid", "ask"]).copy()
                # Require active two-sided market quotes (bid > 0 and ask > bid)
                df_clean = df_clean[(df_clean["bid"] > 0.05) & (df_clean["ask"] > df_clean["bid"])]
                # Restrict to strikes within 15% of spot
                df_clean = df_clean[(df_clean["strike"] >= spot * 0.85) & (df_clean["strike"] <= spot * 1.15)]

                if df_clean.empty:
                    continue

                df_clean["mid"] = 0.5 * (df_clean["bid"] + df_clean["ask"])
                df_clean["rel_spread"] = (df_clean["ask"] - df_clean["bid"]) / df_clean["mid"]
                # Reject wide/stale after-hours spreads (> 35% of mid-price)
                df_clean = df_clean[df_clean["rel_spread"] <= 0.35]

                if df_clean.empty:
                    continue

                near_atm = df_clean.iloc[(df_clean["strike"] - spot).abs().argsort()[:2]]
                for _, row in near_atm.iterrows():
                    solved_iv = solve_iv_from_mid(
                        float(row["mid"]), spot, float(row["strike"]), T_exp, r, q, opt_type
                    )
                    if solved_iv is not None:
                        vols.append(solved_iv)

            if len(vols) >= 2:
                return float(np.median(vols)), target_exp, True, False

        return None, exps[-1], False, False
    except Exception as e:
        print(f"  [{ticker_symbol}] Options fetch/validation error: {e}")
        return None, None, False, False


def compute_half_drawdown_stats(ticker_symbol, r):
    tk = yf.Ticker(ticker_symbol)

    # 1. Spot Price (prefer live intraday/close price)
    spot = None
    try:
        fast_info = tk.fast_info
        spot = getattr(fast_info, "last_price", None)
    except Exception:
        pass

    if not spot or np.isnan(spot) or spot <= 0:
        hist = tk.history(period="5d")
        if hist.empty:
            return None
        spot = float(hist["Close"].iloc[-1])

    if not spot or np.isnan(spot) or spot <= 0:
        return None

    target_strike = round(spot / 2.0, 2)

    # 2. Rolling 2-Year Target Horizon
    T = HORIZON_YEARS
    target_dt = datetime.now(ZoneInfo("America/New_York")) + timedelta(days=int(T * 365.25))
    target_date_str = target_dt.strftime("%Y-%m-%d")

    # 3. Validated ATM Volatility & Downside Skew Modeling
    atm_sigma, src_exp, is_valid, is_rv_proxy = get_valid_atm_volatility(tk, ticker_symbol, spot, r)
    if not is_valid or atm_sigma is None:
        return None

    wing_sigma = atm_sigma * SKEW_MULTIPLIER

    # 4. Closed-form Risk-Neutral Probability Q(S_T <= K)
    q = DIVIDEND_YIELD
    d1 = (np.log(spot / target_strike) + (r - q + 0.5 * wing_sigma**2) * T) / (wing_sigma * np.sqrt(T))
    d2 = d1 - wing_sigma * np.sqrt(T)
    prob_Q = float(norm.cdf(-d2))

    # Sanity check: reject impossible numerical artifacts
    if np.isnan(prob_Q) or prob_Q <= 1e-4 or prob_Q >= 0.99:
        return None

    # 5. Analytical Physical Scenario: Lognormal CDF + CAPM Drift (mu = r + beta * ERP - q)
    if ticker_symbol == "SPY":
        beta = 1.0
    elif ticker_symbol == "QQQ":
        beta = 1.15
    elif ticker_symbol == "DIA":
        beta = 0.85
    else:
        try:
            beta = tk.info.get("beta", 1.2) or 1.2
        except Exception:
            beta = 1.2

    mu = r + (beta * 0.045) - q
    E_ST_P = spot * np.exp(mu * T)
    d2_P = (np.log(E_ST_P / target_strike) - 0.5 * (wing_sigma**2) * T) / (wing_sigma * np.sqrt(T))
    prob_P = float(norm.cdf(-d2_P))

    if np.isnan(prob_P) or prob_P <= 0 or prob_P >= 0.99:
        return None

    return {
        "ticker": ticker_symbol,
        "spot": float(spot),
        "target_strike": target_strike,
        "expiration": target_date_str,
        "source_exp": src_exp or "N/A",
        "atm_iv": float(atm_sigma),
        "wing_iv": float(wing_sigma),
        "prob_Q": prob_Q,
        "prob_P": prob_P,
        "is_fallback": is_rv_proxy
    }


def update_history_csv(today_str, results):
    rows = []
    for res in results:
        rows.append({
            "date": today_str,
            "ticker": res["ticker"],
            "spot": round(res["spot"], 2),
            "target_strike": round(res["target_strike"], 2),
            "prob_Q": round(res["prob_Q"] * 100, 2),
            "prob_P": round(res["prob_P"] * 100, 2)
        })

    new_df = pd.DataFrame(rows)
    if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
        existing_df = pd.read_csv(HISTORY_FILE)
        existing_df = existing_df.dropna(subset=["date", "ticker", "prob_Q"])
        existing_df = existing_df[existing_df["date"] != today_str]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df = combined_df.drop_duplicates(subset=["date", "ticker"], keep="last")
    combined_df = combined_df.sort_values(by=["date", "ticker"]).reset_index(drop=True)
    combined_df.to_csv(HISTORY_FILE, index=False)
    return combined_df


def format_chart_payload(history_df):
    valid_df = history_df.dropna(subset=["date", "ticker", "prob_Q"]).copy()
    unique_dates = sorted(valid_df["date"].unique().tolist())

    palette = {
        "NVDA": "#00FF00",  # Green
        "GOOG": "#FF00FF",  # Magenta
        "MSFT": "#000000",  # Black
        "AAPL": "#FF0000",  # Red
        "AMZN": "#FF8C00",  # Orange
        "META": "#00BFFF",  # Cyan
        "TSLA": "#8A00FF",  # Purple
        "SPCX": "#888888",  # Slate
        "ANT":  "#FF9AFF",  # Light Purple
        "SPY":  "#b45309",  # Deep Ochre Brown
        "QQQ":  "#00009A",  # Dark Blue
        "DIA":  "#047857"   # Dark Forest Green
    }

    fallback_colors = [
        "#00FF00",  "#FF00FF",  "#000000",  "#FF0000",  "#FF8C00", "#00BFFF",
        "#8A00FF",  "#888888", "#FF9AFF",  "#b45309",  "#00009A", "#047857"
    ]

    datasets = []
    for idx, ticker in enumerate(TICKERS):
        sub = valid_df[valid_df["ticker"] == ticker].set_index("date")
        series = [None if pd.isna(v) else float(v) for v in sub["prob_Q"].reindex(unique_dates)]
        color = palette.get(ticker, fallback_colors[idx % len(fallback_colors)])

        datasets.append({
            "label": f"{ticker} Q(S <= 50%)",
            "data": series,
            "borderColor": color,
            "backgroundColor": color,
            "tension": 0.25,
            "borderWidth": 2,
            "pointRadius": 2.5,
            "fill": False
        })

    return json.dumps({"labels": unique_dates, "datasets": datasets})


def render_site(table_rows, chart_payload_json):
    with open(TEMPLATE_FILE, "r") as f:
        template_str = f.read()

    template = jinja2.Template(template_str)
    rendered = template.render(
        rows=table_rows,
        chart_json=chart_payload_json,
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    )
    with open(OUTPUT_FILE, "w") as f:
        f.write(rendered)


if __name__ == "__main__":
    today_str = get_market_today_str()
    print(f"US/Eastern Market Date: {today_str}")

    # 1. Check if we already captured a complete valid snapshot for today
    if is_today_already_complete(today_str):
        print(f"All {len(TICKERS)} tickers already have verified valid data for {today_str}. Suspending updates until tomorrow.")
        sys.exit(0)

    # 2. Fetch live 2-Year Treasury yield
    current_r = get_current_2y_treasury_rate()
    if current_r is None:
        print("Treasury yield unavailable or stale. Skipping update until next hourly check.")
        sys.exit(0)

    print(f"Active 2-Year Risk-Free Rate: {current_r * 100:.2f}%\n")

    # 3. Collect and strictly validate all 12 tickers
    results = []
    failed_tickers = []
    for ticker in TICKERS:
        try:
            data = compute_half_drawdown_stats(ticker, r=current_r)
            if data:
                results.append(data)
                proxy_tag = " [RV-PROXY]" if data["is_fallback"] else ""
                print(
                    f"Valid {ticker:4s}: Spot=${data['spot']:7.2f}, Exp={data['source_exp']}, "
                    f"ATM_IV={data['atm_iv']*100:4.1f}%, Q={data['prob_Q']*100:5.2f}%, P={data['prob_P']*100:5.2f}%{proxy_tag}"
                )
            else:
                failed_tickers.append(ticker)
                print(f"Invalid/Stale quotes for {ticker:4s} — skipping snapshot update.")
        except Exception as e:
            failed_tickers.append(ticker)
            print(f"Error processing {ticker}: {e}")

    # 4. Atomic Commit Guard: Only write history.csv and index.html when ALL 12 tickers succeed
    if len(results) == len(TICKERS) and not failed_tickers:
        hist_df = update_history_csv(today_str, results)
        chart_json = format_chart_payload(hist_df)
        render_site(results, chart_json)
        print(f"\nSuccess! Verified all {len(TICKERS)} tickers and updated {OUTPUT_FILE} and {HISTORY_FILE}.")
    else:
        print(
            f"\nIncomplete or stale market data ({len(results)}/{len(TICKERS)} valid; missing/stale: {failed_tickers}). "
            "Leaving history.csv and index.html untouched; will retry next hour."
        )
