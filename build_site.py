from datetime import datetime
import json
import os
import jinja2
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import norm
import yfinance as yf

TICKERS = ["NVDA", "GOOG", "MSFT", "AAPL", "AMZN", "META"]
OUTPUT_FILE = "index.html"
HISTORY_FILE = "history.csv"
TARGET_EXP_YEAR = 2028
RISK_FREE_RATE = 0.039  # ~2-year Treasury yield


def get_target_expiration(ticker_obj, target_year=TARGET_EXP_YEAR):
  exps = ticker_obj.options
  late_target = [
      e
      for e in exps
      if e.startswith(str(target_year)) and int(e.split("-")[1]) >= 9
  ]
  if late_target:
    return late_target[-1]
  return exps[-1] if exps else None


def compute_half_drawdown_stats(ticker_symbol):
  tk = yf.Ticker(ticker_symbol)

  # Fetch Spot Price
  fast_info = tk.fast_info
  spot = getattr(fast_info, "last_price", None)
  if not spot:
    hist = tk.history(period="5d")
    if hist.empty:
      return None
    spot = float(hist["Close"].iloc[-1])

  target_strike = round(spot / 2.0, 2)
  exp_date_str = get_target_expiration(tk)
  if not exp_date_str:
    return None

  exp_dt = datetime.strptime(exp_date_str, "%Y-%m-%d")
  T = max((exp_dt - datetime.now()).days / 365.25, 0.05)

  # Pull Puts
  chain = tk.option_chain(exp_date_str)
  puts = chain.puts.dropna(subset=["strike", "impliedVolatility"])
  puts = puts[
      (puts["impliedVolatility"] > 0.05) & (puts["impliedVolatility"] < 2.5)
  ]
  if len(puts) < 6:
    return None

  r = RISK_FREE_RATE
  q = 0.005  # Average dividend yield
  F = spot * np.exp((r - q) * T)

  strikes = puts["strike"].values
  ivs = puts["impliedVolatility"].values
  sort_idx = np.argsort(strikes)
  strikes = strikes[sort_idx]
  ivs = ivs[sort_idx]

  log_m = np.log(strikes / F)
  total_var = (ivs**2) * T
  spline = CubicSpline(log_m, total_var, bc_type="natural")

  # Numerical differentiation for Breeden-Litzenberger
  dense_k = np.linspace(strikes.min(), strikes.max(), 1000)
  dense_log_m = np.log(dense_k / F)
  dense_w = np.maximum(spline(dense_log_m), 1e-4)
  dense_sigma = np.sqrt(dense_w / T)

  d1 = (np.log(spot / dense_k) + (r - q + 0.5 * dense_sigma**2) * T) / (
      dense_sigma * np.sqrt(T)
  )
  d2 = d1 - dense_sigma * np.sqrt(T)
  dense_puts = np.exp(-r * T) * (dense_k * norm.cdf(-d2) - F * norm.cdf(-d1))

  # Q(S_T <= K) = exp(rT) * dP/dK
  dP_dK = np.gradient(dense_puts, dense_k)
  Q_cdf = np.clip(np.exp(r * T) * dP_dK, 0.0, 1.0)

  prob_Q = float(np.interp(target_strike, dense_k, Q_cdf))
  atm_sigma = float(np.interp(0.0, dense_log_m, dense_sigma))
  wing_sigma = float(
      np.interp(np.log(target_strike / F), dense_log_m, dense_sigma)
  )

  # Physical CAPM heuristic
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
      "expiration": exp_date_str,
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
    existing_df = existing_df[existing_df["date"] != today_str]
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
  else:
    combined_df = new_df

  combined_df.to_csv(HISTORY_FILE, index=False)
  return combined_df


def format_chart_payload(history_df):
  unique_dates = sorted(history_df["date"].unique().tolist())
  colors = [
      "#10b981",
      "#3b82f6",
      "#8b5cf6",
      "#f59e0b",
      "#ef4444",
      "#06b6d4",
  ]

  datasets = []
  for idx, ticker in enumerate(TICKERS):
    sub = history_df[history_df["ticker"] == ticker].set_index("date")
    series = sub["prob_Q"].reindex(unique_dates).tolist()
    datasets.append({
        "label": f"{ticker} Q(S <= 50%)",
        "data": series,
        "borderColor": colors[idx % len(colors)],
        "backgroundColor": colors[idx % len(colors)],
        "tension": 0.25,
        "borderWidth": 2,
        "pointRadius": 3,
        "fill": False,
    })

  return json.dumps({"labels": unique_dates, "datasets": datasets})


def generate_html(table_rows, chart_payload_json):
  template_str = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Apocalypse: 50% Drawdown Tracker</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #ffffff; color: #111; margin: 40px auto; max-width: 960px; padding: 0 20px; line-height: 1.5; }
            h2 { margin-bottom: 4px; font-weight: 700; }
            p.sub { color: #555; margin-top: 0; margin-bottom: 24px; font-size: 0.95rem; }
            .chart-card { background: #fafafa; border: 1px solid #e5e5e5; border-radius: 8px; padding: 20px; margin-bottom: 35px; }
            table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.95rem; }
            th, td { text-align: right; padding: 10px 14px; border-bottom: 1px solid #e5e5e5; }
            th:first-child, td:first-child { text-align: left; }
            th { background: #fafafa; font-weight: 600; color: #333; }
            tr:hover { background: #fdfdfd; }
            .highlight { font-weight: 600; color: #b91c1c; }
            footer { margin-top: 35px; font-size: 0.8rem; color: #777; border-top: 1px solid #eee; padding-top: 12px; }
            a { color: #2563eb; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h2>AI Apocalypse: Probability of 50% Drawdown (Late 2028)</h2>
        <p class="sub">Market-implied risk-neutral distribution extracted from LEAPS chains via Breeden–Litzenberger (1978).</p>

        <div class="chart-card">
            <canvas id="historyChart" style="max-height: 380px;"></canvas>
        </div>

        <h3 style="font-weight: 600; margin-bottom: 8px;">Latest Snapshot</h3>
        <table>
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Spot Price</th>
                    <th>50% Strike (K)</th>
                    <th>Expiry</th>
                    <th>ATM IV</th>
                    <th>Wing IV</th>
                    <th>Risk-Neutral Q(S &le; K)</th>
                    <th>Physical P(S &le; K)</th>
                </tr>
            </thead>
            <tbody>
                {% for row in rows %}
                <tr>
                    <td><strong>{{ row.ticker }}</strong></td>
                    <td>${{ "%.2f"|format(row.spot) }}</td>
                    <td>${{ "%.2f"|format(row.target_strike) }}</td>
                    <td>{{ row.expiration }}</td>
                    <td>{{ "%.1f"|format(row.atm_iv * 100) }}%</td>
                    <td>{{ "%.1f"|format(row.wing_iv * 100) }}%</td>
                    <td class="highlight">{{ "%.1f"|format(row.prob_Q * 100) }}%</td>
                    <td>{{ "%.1f"|format(row.prob_P * 100) }}%</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <footer>
            Updated: {{ timestamp }} UTC | Repository: <a href="https://github.com/PrestonMcAfee/AI_Apocalypse">PrestonMcAfee/AI_Apocalypse</a>
        </footer>

        <script>
            const chartData = {{ chart_json | safe }};
            const ctx = document.getElementById('historyChart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: chartData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: { position: 'top', labels: { boxWidth: 12, font: { size: 12 } } },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%';
                                }
                            }
                        }
                    },
                    scales: {
                        x: { grid: { display: false } },
                        y: {
                            title: { display: true, text: 'Risk-Neutral Probability (%)' },
                            ticks: { callback: v => v + '%' },
                            suggestedMin: 0
                        }
                    }
                }
            });
        </script>
    </body>
    </html>
    """
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
    except Exception as e:
      print(f"Failed processing {ticker}: {e}")

  if results:
    hist_df = update_history_csv(today_str, results)
    chart_json = format_chart_payload(hist_df)
    generate_html(results, chart_json)
    print(f"Updated {HISTORY_FILE} and rendered {OUTPUT_FILE}.")
