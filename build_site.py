from datetime import datetime, timedelta
import json
import os
import jinja2
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf

# Configuration
TICKERS = ["NVDA", "GOOG", "MSFT", "AAPL", "AMZN", "META", "TSLA", "SPCX"]
OUTPUT_FILE = "index.html"
HISTORY_FILE = "history.csv"
TEMPLATE_FILE = "template.html"
HORIZON_YEARS = 2.0  # Constant 2-year forward window
RISK_FREE_RATE = 0.039  # ~2-year Treasury yield


def get_atm_volatility(tk, spot):
  """Extracts liquid ATM implied volatility from the furthest available reliable chain."""
  try:
    exps = tk.options
    if not exps:
      return 0.35

    target_exp = exps[-1]
    chain = tk.option_chain(target_exp)

    calls = chain.calls.dropna(subset=["strike", "impliedVolatility"])
    puts = chain.puts.dropna(subset=["strike", "impliedVolatility"])

    calls = calls[
        (calls["impliedVolatility"] > 0.10)
        & (calls["impliedVolatility"] < 1.20)
    ]
    puts = puts[
        (puts["impliedVolatility"] > 0.10) & (puts["impliedVolatility"] < 1.20)
    ]

    c_atm = (
        calls.iloc[(calls["strike"] - spot).abs().argsort()[:2]]
        if not calls.empty
        else None
    )
    p_atm = (
        puts.iloc[(puts["strike"] - spot).abs().argsort()[:2]]
        if not puts.empty
        else None
    )

    vols = []
    if c_atm is not None and not c_atm.empty:
      vols.extend(c_atm["impliedVolatility"].tolist())
    if p_atm is not None and not p_atm.empty:
      vols.extend(p_atm["impliedVolatility"].tolist())

    return float(np.median(vols)) if vols else 0.35
  except Exception:
    return 0.35


def compute_half_drawdown_stats(ticker_symbol):
  tk = yf.Ticker(ticker_symbol)

  # 1. Spot Price
  fast_info = tk.fast_info
  spot = getattr(fast_info, "last_price", None)
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
  atm_sigma = get_atm_volatility(tk, spot)

  # Downside skew premium for 50% OTM strike over ~2yr horizon
  skew_multiplier = 1.18
  wing_sigma = atm_sigma * skew_multiplier

  # 4. Closed-form Risk-Neutral Probability Q(S_T <= K)
  r = RISK_FREE_RATE
  q = 0.005  # Approximate dividend yield
  F = spot * np.exp((r - q) * T)

  d1 = (np.log(spot / target_strike) + (r - q + 0.5 * wing_sigma**2) * T) / (
      wing_sigma * np.sqrt(T)
  )
  d2 = d1 - wing_sigma * np.sqrt(T)

  base_prob = norm.cdf(-d2)

  # Skew slope adjustment for probability mass concentration in the wing
  skew_slope_adjustment = 0.85
  prob_Q = float(np.clip(base_prob * skew_slope_adjustment, 0.01, 0.99))

  # 5. Physical Probability (CAPM Drift mu = r + beta * ERP)
  beta = tk.info.get("beta", 1.2) or 1.2
  mu = r + (beta * 0.045)
  E_ST_P = spot * np.exp(mu * T)
  d2_P = (np.log(E_ST_P / target_strike) - 0.5 * (wing_sigma**2) * T) / (
      wing_sigma * np.sqrt(T)
  )
  prob_P = float(norm.cdf(-d2_P))

  return {
      "ticker": ticker_symbol,
      "spot": spot,
      "target_strike": target_strike,
      "expiration": target_date_str,
      "atm_iv": atm_sigma,
      "wing_iv": wing_sigma,
      "prob_Q": prob_Q,
      "prob_P": prob_P,
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
        "prob_P": round(res["prob_P"] * 100, 2),
    })

  new_df = pd.DataFrame(rows)
  if os.path.exists(HISTORY_FILE):
    existing_df = pd.read_csv(HISTORY_FILE)
    existing_df = existing_df.dropna(subset=["date", "ticker", "prob_Q"])
    existing_df = existing_df[existing_df["date"] != today_str]
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
  else:
    combined_df = new_df

  combined_df.to_csv(HISTORY_FILE, index=False)
  return combined_df


def format_chart_payload(history_df):
  valid_df = history_df.dropna(subset=["date", "ticker", "prob_Q"]).copy()
  unique_dates = sorted(valid_df["date"].unique().tolist())

  palette = {
      "NVDA": "#10b981",  # Emerald Green
      "GOOG": "#3b82f6",  # Vibrant Blue
      "MSFT": "#8b5cf6",  # Purple
      "AAPL": "#f59e0b",  # Amber / Gold
      "AMZN": "#ef4444",  # Red
      "META": "#06b6d4",  # Cyan
      "TSLA": "#e11d48",  # Crimson / Rose
      "SPCX": "#475569",  # Slate Navy
  }

  fallback_colors = [
      "#10b981",
      "#3b82f6",
      "#8b5cf6",
      "#f59e0b",
      "#ef4444",
      "#06b6d4",
      "#e11d48",
      "#475569",
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
        "pointRadius": 3,
        "fill": False,
    })

  return json.dumps({"labels": unique_dates, "datasets": datasets})


def render_site(table_rows, chart_payload_json):
  with open(TEMPLATE_FILE, "r") as f:
    template_str = f.read()

  template = jinja2.Template(template_str)
  rendered = template.render(
      rows=table_rows,
      chart_json=chart_payload_json,
      timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
  )
  with open(OUTPUT_FILE, "w") as f:
    f.write(rendered)


if __name__ == "__main__":
  today_str = datetime.utcnow().strftime("%Y-%m-%d")
  results = []
  for ticker in TICKERS:
    try:
      data = compute_half_drawdown_stats(ticker)
      if data:
        results.append(data)
        print(
            f"Processed {ticker}: Spot=${data['spot']:.2f},"
            f" Target={data['expiration']}, Q={data['prob_Q']*100:.1f}%"
        )
    except Exception as e:
      print(f"Failed processing {ticker}: {e}")

  if results:
    hist_df = update_history_csv(today_str, results)
    chart_json = format_chart_payload(hist_df)
    render_site(results, chart_json)
    print(f"\nSuccessfully wrote {OUTPUT_FILE} and updated {HISTORY_FILE}.")
  else:
    print("\nNo results gathered. File not updated.")
