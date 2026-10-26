import os
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
import jinja2

# Configuration
TICKERS = [
    "NVDA", "GOOG", "MSFT", "AAPL", "AMZN", 
    "META", "TSLA", "SPCX", "SPY", "QQQ", "DIA"
]
OUTPUT_FILE = "index.html"
HISTORY_FILE = "history.csv"
TEMPLATE_FILE = "template.html"
HORIZON_YEARS = 2.0  # Constant 2-year forward window
DEFAULT_RISK_FREE_RATE = 0.039  # Fallback ~2-year Treasury yield

def get_current_2y_treasury_rate(default_rate=DEFAULT_RISK_FREE_RATE):
    """Fetches the latest active 2-Year US Treasury yield (Yahoo ticker: 2YY=F)."""
    try:
        tk = yf.Ticker("2YY=F")
        fast_info = getattr(tk, "fast_info", None)
        rate_val = getattr(fast_info, "last_price", None) if fast_info else None
        
        if rate_val is None:
            hist = tk.history(period="5d")
            if not hist.empty:
                rate_val = float(hist["Close"].iloc[-1])
                
        if rate_val is not None and rate_val > 0:
            return float(rate_val) / 100.0
    except Exception as e:
        print(f"Warning: Could not fetch dynamic 2Y rate ({e}). Using default {default_rate:.3f}.")
    return default_rate

def get_atm_volatility(tk, spot):
    """
    Extracts liquid ATM implied volatility from the furthest available reliable chain.
    Also returns the actual expiration date and day count to avoid unmeasured maturity distortion.
    """
    try:
        exps = tk.options
        if not exps:
            return 0.35, None, False

        target_exp = exps[-1]
        chain = tk.option_chain(target_exp)

        calls = chain.calls.dropna(subset=['strike', 'impliedVolatility'])
        puts = chain.puts.dropna(subset=['strike', 'impliedVolatility'])

        calls = calls[(calls['impliedVolatility'] > 0.05) & (calls['impliedVolatility'] < 1.20)]
        puts = puts[(puts['impliedVolatility'] > 0.05) & (puts['impliedVolatility'] < 1.20)]

        c_atm = calls.iloc[(calls['strike'] - spot).abs().argsort()[:2]] if not calls.empty else None
        p_atm = puts.iloc[(puts['strike'] - spot).abs().argsort()[:2]] if not puts.empty else None

        vols = []
        if c_atm is not None and not c_atm.empty:
            vols.extend(c_atm['impliedVolatility'].tolist())
        if p_atm is not None and not p_atm.empty:
            vols.extend(p_atm['impliedVolatility'].tolist())

        if vols:
            return float(np.median(vols)), target_exp, False
        return 0.35, target_exp, True
    except Exception:
        return 0.35, None, True

def compute_half_drawdown_stats(ticker_symbol, r=DEFAULT_RISK_FREE_RATE):
    tk = yf.Ticker(ticker_symbol)

    # 1. Spot Price
    fast_info = tk.fast_info
    spot = getattr(fast_info, 'last_price', None)
    if not spot:
        hist = tk.history(period="5d")
        if hist.empty:
            return None
        spot = float(hist["Close"].iloc[-1])

    target_strike = round(spot / 2.0, 2)

    # 2. Rolling 2-Year Target Horizon
    T = HORIZON_YEARS
    target_dt = datetime.now() + timedelta(days=int(T * 365.25))
    target_date_str = target_dt.strftime("%Y-%m-%d")

    # 3. ATM Volatility & Downside Skew Modeling
    atm_sigma, src_exp, is_fallback = get_atm_volatility(tk, spot)

    # Downside skew premium for 50% OTM strike over ~2yr horizon
    skew_multiplier = 1.18
    wing_sigma = atm_sigma * skew_multiplier

    # 4. Closed-form Risk-Neutral Probability Q(S_T <= K) (Unclipped, unscaled)
    q = 0.005  # Approximate dividend yield
    F = spot * np.exp((r - q) * T)

    d1 = (np.log(spot / target_strike) + (r - q + 0.5 * wing_sigma**2) * T) / (wing_sigma * np.sqrt(T))
    d2 = d1 - wing_sigma * np.sqrt(T)

    prob_Q = float(norm.cdf(-d2))

    # 5. Analytical Physical Scenario: Lognormal CDF + CAPM Drift (mu = r + beta * ERP - q)
    beta = 1.0
    if ticker_symbol == "SPY":
        beta = 1.0
    elif ticker_symbol == "QQQ":
        beta = 1.15
    elif ticker_symbol == "DIA":
        beta = 0.85
    else:
        beta = tk.info.get('beta', 1.2) or 1.2

    mu = r + (beta * 0.045) - q
    E_ST_P = spot * np.exp(mu * T)
    d2_P = (np.log(E_ST_P / target_strike) - 0.5 * (wing_sigma**2) * T) / (wing_sigma * np.sqrt(T))
    prob_P = float(norm.cdf(-d2_P))

    return {
        "ticker": ticker_symbol,
        "spot": spot,
        "target_strike": target_strike,
        "expiration": target_date_str,
        "source_exp": src_exp or "N/A",
        "atm_iv": atm_sigma,
        "wing_iv": wing_sigma,
        "prob_Q": prob_Q,
        "prob_P": prob_P,
        "is_fallback": is_fallback
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
        "AAPL": "#FF8C00",  # Dark Orange
        "AMZN": "#FF8D1A",  # Orange
        "META": "#00BFFF",  # Cyan
        "TSLA": "#8A2BE2",  # Purple
        "SPCX": "#475569",  # Slate
        "SPY":  "#b45309",  # Deep Ochre Brown
        "QQQ":  "#1d4ed8",  # Deep Cobalt Blue
        "DIA":  "#047857"   # Dark Forest Green
    }
    
    fallback_colors = [
        "#00FF00",  "#FF00FF",  "#000000",  "#FF8C00",  "#FF8D1A",
        "#00BFFF",  "#8A2BE2",  "#475569",  "#b45309",  "#1d4ed8", "#047857"
    ]
    
    datasets = []
    for idx, ticker in enumerate(TICKERS):
        sub = valid_df[valid_df["ticker"] == ticker].set_index("date")
        series = sub["prob_Q"].reindex(unique_dates).tolist()
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
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    current_r = get_current_2y_treasury_rate(default_rate=DEFAULT_RISK_FREE_RATE)
    print(f"Active 2-Year Risk-Free Rate: {current_r * 100:.2f}%\n")

    results = []
    for ticker in TICKERS:
        try:
            data = compute_half_drawdown_stats(ticker, r=current_r)
            if data:
                results.append(data)
                fb_note = " [FALLBACK]" if data["is_fallback"] else ""
                print(f"Processed {ticker:4s}: Spot=${data['spot']:7.2f}, Exp={data['source_exp']}, Q={data['prob_Q']*100:5.2f}%, P={data['prob_P']*100:5.2f}%{fb_note}")
        except Exception as e:
            print(f"Failed processing {ticker}: {e}")

    if results:
        hist_df = update_history_csv(today_str, results)
        chart_json = format_chart_payload(hist_df)
        render_site(results, chart_json)
        print(f"\nSuccessfully wrote {OUTPUT_FILE} and updated {HISTORY_FILE}.")
    else:
        print("\nNo results gathered. File not updated.")
