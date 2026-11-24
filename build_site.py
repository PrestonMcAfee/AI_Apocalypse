"""
Daily build: market-implied probability that each ticker ends a rolling 2-year
window at or below 50% of today's price.

Method (risk-neutral, from option prices):
  1. For the listed expiries that bracket 2 years, take bid/ask mid prices of
     out-of-the-money puts (K < F) and calls (K > F).
  2. Back out the implied forward F from put-call parity near the money, so
     dividends and borrow costs come from the market rather than an assumption.
  3. Convert each mid to an implied volatility and fit a smooth smile
     sigma(k), k = ln(K/F), per expiry.
  4. Interpolate total implied variance in time to exactly T = 2 years.
  5. Breeden-Litzenberger: Q(S_T <= K) = e^{rT} * dPut/dK, evaluated at
     K = S/2 on the fitted smile. Unlike N(-d2) with a single volatility,
     this accounts for the slope of the smile (skew).
"""
import os
import sys
import json
import math
from datetime import datetime, date

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
HISTORY_FILE = "history.csv"          # 50% of current price
HISTORY52_FILE = "history52.csv"      # 50% of 52-week high
SKEW_FILE = "skew_curve.csv"
LAST_RUN_FILE = "last_run.json"       # quality of the best run for the latest market date
RUN_WINDOW_ET = ("10:00", "15:55")    # scheduled runs only while option quotes are live          # effective vol by moneyness (calibrates the backfill)
SKEW_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # K / spot
TEMPLATE_FILE = "template.html"
HORIZON_YEARS = 2.0
DEFAULT_RISK_FREE_RATE = 0.039   # Fallback ~2-year Treasury yield
EQUITY_RISK_PREMIUM = 0.045      # Used only for the "physical" CAPM scenario
INDEX_BETAS = {"SPY": 1.0, "QQQ": 1.15, "DIA": 0.85}

MIN_EXPIRY_YEARS = 0.25          # Ignore short-dated chains when longer ones exist
PUT_MONEYNESS_MIN = 0.20         # Use OTM puts with K/F in [0.20, 1.0]
CALL_MONEYNESS_MAX = 1.40        # Use OTM calls with K/F in [1.0, 1.40]
MIN_SMILE_POINTS = 5
IV_BOUNDS = (0.03, 3.0)


# ---------------------------------------------------------------------------
# Market inputs
# ---------------------------------------------------------------------------
def get_current_2y_treasury_rate(default_rate=DEFAULT_RISK_FREE_RATE):
    """Latest 2-Year US Treasury yield (Yahoo ticker 2YY=F, quoted in percent)."""
    try:
        tk = yf.Ticker("2YY=F")
        hist = tk.history(period="5d")
        if not hist.empty:
            rate_val = float(hist["Close"].iloc[-1])
            if rate_val > 0:
                return rate_val / 100.0
    except Exception as e:
        print(f"Warning: Could not fetch 2Y rate ({e}).")
    print(f"Using default rate {default_rate:.3%}.")
    return default_rate


def get_market_date():
    """The trading date the prices belong to (US/Eastern), not the UTC run date."""
    try:
        hist = yf.Ticker("SPY").history(period="5d")
        if not hist.empty:
            ts = hist.index[-1]
            if ts.tzinfo is not None:
                ts = ts.tz_convert("America/New_York")
            return ts.date()
    except Exception as e:
        print(f"Warning: Could not determine market date ({e}).")
    return pd.Timestamp.now(tz="America/New_York").date()


def get_52w_high(tk, spot):
    """Highest intraday high over the past year (split-adjusted, not dividend-adjusted)."""
    try:
        hist = tk.history(period="1y", auto_adjust=False)
        if not hist.empty:
            return max(float(hist["High"].max()), spot), len(hist)
    except Exception as e:
        print(f"    52-week high unavailable ({e})")
    return spot, 0


def get_spot(tk):
    spot = None
    try:
        spot = tk.fast_info.last_price
    except Exception:
        pass
    if not spot or not np.isfinite(spot):
        hist = tk.history(period="5d")
        if hist.empty:
            return None
        spot = float(hist["Close"].iloc[-1])
    return float(spot)


# ---------------------------------------------------------------------------
# Black-Scholes on the forward
# ---------------------------------------------------------------------------
def bs_forward_price(F, K, T, sigma, disc, is_put):
    """Option price written on the forward: disc * E[payoff]."""
    if sigma <= 0 or T <= 0:
        intrinsic = max(K - F, 0.0) if is_put else max(F - K, 0.0)
        return disc * intrinsic
    sd = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    if is_put:
        return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))


def implied_vol(price, F, K, T, disc, is_put):
    intrinsic = disc * (max(K - F, 0.0) if is_put else max(F - K, 0.0))
    if not np.isfinite(price) or price <= intrinsic + 1e-10:
        return np.nan
    f = lambda s: bs_forward_price(F, K, T, s, disc, is_put) - price
    lo, hi = IV_BOUNDS
    try:
        if f(lo) > 0 or f(hi) < 0:
            return np.nan
        return brentq(f, lo, hi, xtol=1e-7)
    except Exception:
        return np.nan


def lognormal_prob_below(F, K, T, sigma):
    """N(-d2): lognormal probability of S_T <= K with a single volatility."""
    sd = sigma * math.sqrt(T)
    d2 = (math.log(F / K) - 0.5 * sd * sd) / sd
    return float(norm.cdf(-d2))


def effective_sigma(F, K, T, prob):
    """Single lognormal volatility that reproduces `prob` (used to calibrate the backfill)."""
    if not (0 < prob < 1):
        return np.nan
    try:
        return brentq(lambda s: lognormal_prob_below(F, K, T, s) - prob, 1e-4, 5.0)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Smile construction per expiry
# ---------------------------------------------------------------------------
def _mid_prices(df, allow_last):
    df = df.copy()
    bid = df.get("bid", pd.Series(np.nan, index=df.index)).astype(float)
    ask = df.get("ask", pd.Series(np.nan, index=df.index)).astype(float)
    good = (bid > 0) & (ask > 0) & (ask >= bid)
    df["mid"] = np.where(good, 0.5 * (bid + ask), np.nan)
    df["used_last"] = False
    if allow_last and "lastPrice" in df:
        last = df["lastPrice"].astype(float)
        fill = df["mid"].isna() & (last > 0)
        df.loc[fill, "mid"] = last[fill]
        df.loc[fill, "used_last"] = True
    return df


def implied_forward(calls, puts, spot, T, disc, fallback_F):
    """Put-call parity: C - P = disc * (F - K)  =>  F = K + (C - P)/disc."""
    m = pd.merge(calls[["strike", "mid"]], puts[["strike", "mid"]],
                 on="strike", suffixes=("_c", "_p")).dropna()
    if m.empty:
        return fallback_F, False
    m = m.iloc[(m["strike"] - spot).abs().argsort()[:6]]
    ests = m["strike"] + (m["mid_c"] - m["mid_p"]) / disc
    F = float(np.median(ests))
    # Sanity: forward must be within a plausible band of the carry-based guess
    if not np.isfinite(F) or abs(F / fallback_F - 1) > 0.25:
        return fallback_F, False
    return F, True


class Smile:
    """sigma(k), k = ln(K/F): quadratic over the data range, linear beyond it."""

    def __init__(self, k, iv):
        self.kmin, self.kmax = float(np.min(k)), float(np.max(k))
        deg = 2 if len(k) >= 6 else 1
        coef = np.polyfit(k, iv, deg)
        # Robust refit: drop points far from the first fit
        resid = iv - np.polyval(coef, k)
        mad = np.median(np.abs(resid - np.median(resid))) + 1e-6
        keep = np.abs(resid) <= max(5 * mad, 0.02)
        if keep.sum() >= max(MIN_SMILE_POINTS, deg + 1):
            k, iv = k[keep], iv[keep]
            coef = np.polyfit(k, iv, deg)
        if deg == 2 and coef[0] < 0:     # concave smile is implausible: fall back to linear
            coef = np.concatenate([[0.0], np.polyfit(k, iv, 1)])
        self.coef = np.atleast_1d(coef)
        self.n_points = len(k)

    def _poly(self, k):
        return float(np.polyval(self.coef, k))

    def _slope(self, k):
        return float(np.polyval(np.polyder(self.coef), k)) if len(self.coef) > 1 else 0.0

    def __call__(self, k):
        if k < self.kmin:
            s = self._poly(self.kmin) + self._slope(self.kmin) * (k - self.kmin)
        elif k > self.kmax:
            s = self._poly(self.kmax) + self._slope(self.kmax) * (k - self.kmax)
        else:
            s = self._poly(k)
        return max(s, IV_BOUNDS[0])


def build_expiry_smile(tk, exp_str, spot, market_date, r, q_guess):
    """Returns dict with T, F, smile, and quality flags, or None if unusable."""
    T = (pd.Timestamp(exp_str).date() - market_date).days / 365.25
    if T <= 0:
        return None
    chain = tk.option_chain(exp_str)
    disc = math.exp(-r * T)
    carry_F = spot * math.exp((r - q_guess) * T)

    for allow_last in (False, True):
        calls = _mid_prices(chain.calls.dropna(subset=["strike"]), allow_last)
        puts = _mid_prices(chain.puts.dropna(subset=["strike"]), allow_last)
        F, parity_ok = implied_forward(calls, puts, spot, T, disc, carry_F)

        otm_p = puts[(puts["strike"] < F) & (puts["strike"] >= PUT_MONEYNESS_MIN * F)].dropna(subset=["mid"])
        otm_c = calls[(calls["strike"] >= F) & (calls["strike"] <= CALL_MONEYNESS_MAX * F)].dropna(subset=["mid"])

        ks, ivs, used_last = [], [], False
        for df, is_put in ((otm_p, True), (otm_c, False)):
            for _, row in df.iterrows():
                iv = implied_vol(float(row["mid"]), F, float(row["strike"]), T, disc, is_put)
                if np.isfinite(iv):
                    ks.append(math.log(row["strike"] / F))
                    ivs.append(iv)
                    used_last |= bool(row["used_last"])

        n_put_side = sum(1 for k in ks if k < 0)
        if len(ks) >= MIN_SMILE_POINTS and n_put_side >= 3:
            return {
                "exp": exp_str, "T": T, "F": F, "disc": disc,
                "smile": Smile(np.array(ks), np.array(ivs)),
                "parity_ok": parity_ok, "used_last": used_last,
            }
    return None


# ---------------------------------------------------------------------------
# Combine expiries into a 2-year smile and compute probabilities
# ---------------------------------------------------------------------------
def trailing_dividend_yield(tk, spot):
    try:
        divs = tk.dividends
        if divs is None or divs.empty:
            return 0.0
        idx = divs.index.tz_localize(None) if divs.index.tz is not None else divs.index
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=365)
        return float(divs[idx >= cutoff].sum()) / spot
    except Exception:
        return 0.0


def two_year_smile(tk, spot, market_date, r, q_guess):
    """
    Pick the expiries that bracket 2 years and interpolate total variance
    w(k, T) = sigma(k)^2 * T linearly in T at fixed k = ln(K/F_T).
    If no listed expiry reaches 2 years, extend the longest usable smile by
    holding the smile fixed in standardized moneyness k / sqrt(T).
    """
    exps = list(tk.options or [])
    if not exps:
        return None
    dated = sorted(((pd.Timestamp(e).date() - market_date).days / 365.25, e) for e in exps)
    dated = [(T, e) for T, e in dated if T > 0]
    long_enough = [(T, e) for T, e in dated if T >= MIN_EXPIRY_YEARS] or dated
    below = [x for x in long_enough if x[0] <= HORIZON_YEARS]
    above = [x for x in long_enough if x[0] > HORIZON_YEARS]

    def first_usable(candidates):
        for T, e in candidates:
            try:
                s = build_expiry_smile(tk, e, spot, market_date, r, q_guess)
            except Exception as ex:
                print(f"    {e}: chain error ({ex})")
                s = None
            if s is not None:
                return s
        return None

    lo = first_usable(list(reversed(below)))   # longest expiry <= 2y
    hi = first_usable(above)                   # shortest expiry > 2y
    Tt = HORIZON_YEARS

    def growth(s):  # annualized log forward drift implied by the market
        return math.log(s["F"] / spot) / s["T"]

    if lo and hi:
        wgt = (Tt - lo["T"]) / (hi["T"] - lo["T"])
        g = (1 - wgt) * growth(lo) + wgt * growth(hi)

        def sigma_at(k):
            w_lo = lo["smile"](k) ** 2 * lo["T"]
            w_hi = hi["smile"](k) ** 2 * hi["T"]
            w = max((1 - wgt) * w_lo + wgt * w_hi, 1e-8)
            return math.sqrt(w / Tt)
        used = [lo, hi]
        extrapolated = False
    elif lo or hi:
        s = lo or hi
        g = growth(s)

        def sigma_at(k):
            return s["smile"](k * math.sqrt(s["T"] / Tt))
        used = [s]
        extrapolated = True
    else:
        return None

    F = spot * math.exp(g * Tt)
    return {
        "F": F, "T": Tt, "sigma_at": sigma_at,
        "expiries": [u["exp"] for u in used],
        "extrapolated": extrapolated,
        "used_last": any(u["used_last"] for u in used),
        "parity_ok": all(u["parity_ok"] for u in used),
        "max_listed_T": dated[-1][0] if dated else 0.0,
    }


def bl_prob_below(F, K, T, r, sigma_at):
    """Breeden-Litzenberger: Q(S_T <= K) = e^{rT} dP/dK, by central difference on the smile."""
    disc = math.exp(-r * T)
    h = 0.005 * K

    def put(Kx):
        return bs_forward_price(F, Kx, T, sigma_at(math.log(Kx / F)), disc, True)
    q = (put(K + h) - put(K - h)) / (2 * h) / disc
    return float(min(max(q, 0.0), 1.0))


def get_beta(tk, ticker_symbol):
    if ticker_symbol in INDEX_BETAS:
        return INDEX_BETAS[ticker_symbol]
    try:
        return tk.info.get("beta", 1.2) or 1.2
    except Exception:
        return 1.2


def compute_half_drawdown_stats(ticker_symbol, market_date, r=DEFAULT_RISK_FREE_RATE):
    tk = yf.Ticker(ticker_symbol)
    spot = get_spot(tk)
    if not spot:
        return None
    K = spot / 2.0
    q_guess = trailing_dividend_yield(tk, spot)

    surf = two_year_smile(tk, spot, market_date, r, q_guess)
    if surf is None:
        n_exp = len(tk.options or [])
        print(f"  {ticker_symbol}: no usable option chain ({n_exp} expiries listed); no probability recorded.")
        return {"ticker": ticker_symbol, "spot": spot, "target_strike": round(K, 2),
                "no_options": True, "is_fallback": False, "flag_note": ""}

    F, T, sigma_at = surf["F"], surf["T"], surf["sigma_at"]
    beta = get_beta(tk, ticker_symbol)

    # Physical scenario: same market-implied distribution shape, shifted so the
    # expected return is CAPM (r + beta*ERP) instead of the risk-free carry.
    # P(S_T <= K) = Q(S_T <= K * exp(-beta*ERP*T)).
    phys_shift = math.exp(-beta * EQUITY_RISK_PREMIUM * T)

    prob_Q = bl_prob_below(F, K, T, r, sigma_at)
    prob_P = bl_prob_below(F, K * phys_shift, T, r, sigma_at)

    # Alternative metric: finish at or below half of the 52-week high
    high_52w, n_days_52 = get_52w_high(tk, spot)
    K52 = high_52w / 2.0
    prob_Q52 = bl_prob_below(F, K52, T, r, sigma_at)
    prob_P52 = bl_prob_below(F, K52 * phys_shift, T, r, sigma_at)

    # Effective lognormal vol at several strikes, so the backfill can reproduce
    # the skew when the 52-week strike is not at 50% of spot.
    skew = []
    for m in SKEW_GRID:
        Km = m * spot
        se = effective_sigma(F, Km, T, bl_prob_below(F, Km, T, r, sigma_at))
        if np.isfinite(se):
            skew.append((m, se))

    target_dt = market_date + pd.Timedelta(days=int(T * 365.25))
    return {
        "ticker": ticker_symbol,
        "spot": spot,
        "target_strike": round(K, 2),
        "expiration": target_dt.strftime("%Y-%m-%d"),
        "source_exp": " / ".join(surf["expiries"]),
        "atm_iv": sigma_at(0.0),
        "wing_iv": sigma_at(math.log(K / F)),
        "sigma_eff": effective_sigma(F, K, T, prob_Q),
        "prob_Q": prob_Q,
        "prob_P": prob_P,
        "high_52w": high_52w,
        "high_52w_partial": n_days_52 < 240,
        "target_strike52": round(K52, 2),
        "drawdown": spot / high_52w - 1.0,
        "prob_Q52": prob_Q52,
        "prob_P52": prob_P52,
        "skew": skew,
        "no_options": False,
        "used_last": surf["used_last"],
        "is_fallback": surf["extrapolated"] or surf["used_last"],
        "flag_note": "; ".join(filter(None, [
            f"longest listed expiry is {surf['max_listed_T']:.2f}y, extrapolated to 2y" if surf["extrapolated"] else "",
            "some prices are last trades, not bid/ask mids" if surf["used_last"] else "",
        ])),
    }


# ---------------------------------------------------------------------------
# History, chart, site
# ---------------------------------------------------------------------------
def upsert_csv(path, new_df, keys):
    """Replace rows matching (keys) in new_df, keep everything else, sort, save."""
    if new_df.empty:
        return pd.read_csv(path) if os.path.exists(path) else new_df
    if os.path.exists(path) and os.path.getsize(path) > 0:
        existing = pd.read_csv(path)
        merged = existing.merge(new_df[keys], on=keys, how="left", indicator=True)
        existing = existing[(merged["_merge"] == "left_only").values]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset=keys, keep="last")
    combined = combined.sort_values(by=keys).reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def _r(x, n):
    return round(float(x), n) if x is not None and np.isfinite(x) else np.nan


def update_history_files(date_str, results):
    live = [res for res in results if not res.get("no_options")]
    h = upsert_csv(HISTORY_FILE, pd.DataFrame([{
        "date": date_str, "ticker": res["ticker"],
        "spot": _r(res["spot"], 2), "target_strike": _r(res["target_strike"], 2),
        "prob_Q": _r(res["prob_Q"] * 100, 2), "prob_P": _r(res["prob_P"] * 100, 2),
        "sigma_eff": _r(res["sigma_eff"], 4), "method": "options",
    } for res in live]), ["date", "ticker"])
    h52 = upsert_csv(HISTORY52_FILE, pd.DataFrame([{
        "date": date_str, "ticker": res["ticker"],
        "spot": _r(res["spot"], 2), "high_52w": _r(res["high_52w"], 2),
        "target_strike": _r(res["target_strike52"], 2),
        "prob_Q": _r(res["prob_Q52"] * 100, 2), "prob_P": _r(res["prob_P52"] * 100, 2),
        "method": "options",
    } for res in live]), ["date", "ticker"])
    upsert_csv(SKEW_FILE, pd.DataFrame([{
        "date": date_str, "ticker": res["ticker"], "moneyness": m, "sigma_eff": _r(se, 4),
    } for res in live for m, se in res["skew"]]), ["date", "ticker", "moneyness"])
    return h, h52


PALETTE = {
    "NVDA": "#00FF00", "GOOG": "#FF00FF", "MSFT": "#000000", "AAPL": "#FF0000",
    "AMZN": "#FF8C00", "META": "#00BFFF", "TSLA": "#8A00FF", "SPCX": "#888888",
    "ANT": "#FF9AFF", "SPY": "#b45309", "QQQ": "#00009A", "DIA": "#047857",
}


def format_chart_payload(history_df, label_suffix="Q(S <= 50%)"):
    valid_df = history_df.dropna(subset=["date", "ticker", "prob_Q"]).copy()
    unique_dates = sorted(valid_df["date"].unique().tolist())

    live_start = None
    if "method" in valid_df.columns:
        live = valid_df[valid_df["method"] == "options"]
        if not live.empty:
            live_start = live["date"].min()

    datasets = []
    for ticker in TICKERS:
        sub = valid_df[valid_df["ticker"] == ticker].set_index("date")
        series = sub["prob_Q"].reindex(unique_dates)
        series = [None if pd.isna(v) else float(v) for v in series]
        color = PALETTE.get(ticker, "#666666")
        datasets.append({
            "label": f"{ticker} {label_suffix}",
            "data": series,
            "borderColor": color,
            "backgroundColor": color,
            "tension": 0.25,
            "borderWidth": 2,
            "pointRadius": 2.5,
            "fill": False,
            "spanGaps": True,
        })

    return json.dumps({"labels": unique_dates, "datasets": datasets, "liveStart": live_start})


def render_site(table_rows, chart_payload_json, chart52_payload_json, market_date):
    with open(TEMPLATE_FILE, "r") as f:
        template = jinja2.Template(f.read())
    rendered = template.render(
        rows=table_rows,
        chart_json=chart_payload_json,
        chart52_json=chart52_payload_json,
        market_date=market_date.strftime("%Y-%m-%d"),
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
    )
    with open(OUTPUT_FILE, "w") as f:
        f.write(rendered)


def set_output(name, value):
    """Tell the GitHub workflow whether later steps (commit, deploy) should run."""
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"{name}={value}\n")


def load_last_run():
    try:
        with open(LAST_RUN_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x))


def render_from_saved():
    """Rebuild index.html from the CSVs and the table rows saved by the best run."""
    last = load_last_run()
    rows = last.get("rows")
    if not rows:
        print("No saved run to render from.")
        return False
    h = pd.read_csv(HISTORY_FILE)
    h52 = pd.read_csv(HISTORY52_FILE) if os.path.exists(HISTORY52_FILE) else h.iloc[0:0]
    render_site(rows, format_chart_payload(h), format_chart_payload(h52, "Q(S <= 50% of high)"),
                pd.Timestamp(last["market_date"]).date())
    print(f"Rendered {OUTPUT_FILE} from saved run of {last['market_date']}.")
    return True


def main():
    """
    Run at most once successfully per trading day.
    - If a complete run (every ticker with listed options priced from live
      bid/ask quotes) already exists for this market date, exit without changes.
    - Otherwise compute, and write only if this run is at least as good as
      the best earlier run for the same date (so a late run with stale
      after-hours quotes cannot overwrite a good mid-session snapshot).
    Scheduled runs also exit outside RUN_WINDOW_ET (US/Eastern), which keeps
    them out of pre-market and after-hours regardless of daylight saving time
    or GitHub start delays.
    FORCE_RUN=1 (manual runs) skips the window and the already-complete check.
    OVERWRITE=1 additionally replaces today's data even if an earlier run was cleaner.
    """
    if "--render-only" in sys.argv:
        set_output("updated", "true" if render_from_saved() else "false")
        return
    truthy = ("1", "true", "yes")
    force = os.environ.get("FORCE_RUN", "").lower() in truthy
    overwrite = os.environ.get("OVERWRITE", "").lower() in truthy
    now_et = pd.Timestamp.now(tz="America/New_York")
    if not force and not (RUN_WINDOW_ET[0] <= now_et.strftime("%H:%M") <= RUN_WINDOW_ET[1]):
        print(f"{now_et:%H:%M} ET is outside the run window {RUN_WINDOW_ET}; nothing to do.")
        set_output("updated", "false")
        return
    market_date = get_market_date()
    date_str = market_date.strftime("%Y-%m-%d")
    last = load_last_run()
    if not force and last.get("market_date") == date_str and last.get("complete"):
        print(f"Already have a complete run for {date_str}; nothing to do.")
        set_output("updated", "false")
        return

    current_r = get_current_2y_treasury_rate()
    print(f"Market date: {date_str} | 2-Year rate: {current_r * 100:.2f}%\n")

    results, n_errors = [], 0
    for ticker in TICKERS:
        try:
            data = compute_half_drawdown_stats(ticker, market_date, r=current_r)
            if data and data.get("no_options"):
                results.append(data)
            elif data:
                results.append(data)
                note = f" [{data['flag_note']}]" if data["flag_note"] else ""
                print(f"Processed {ticker:4s}: Spot=${data['spot']:8.2f}, Exp={data['source_exp']}, "
                      f"ATM={data['atm_iv']:.1%}, IV@K={data['wing_iv']:.1%}, "
                      f"Q={data['prob_Q']*100:5.2f}%, P={data['prob_P']*100:5.2f}% | "
                      f"52wH=${data['high_52w']:.2f}, Q52={data['prob_Q52']*100:5.2f}%{note}")
            else:
                n_errors += 1
        except Exception as e:
            n_errors += 1
            print(f"Failed processing {ticker}: {e}")

    priced = [r for r in results if not r.get("no_options")]
    n_clean = sum(1 for r in priced if not r.get("used_last"))
    complete = n_errors == 0 and n_clean == len(priced) and len(priced) > 0
    print(f"\nQuality: {n_clean}/{len(priced)} tickers priced from live quotes, {n_errors} errors"
          f" -> {'complete' if complete else 'INCOMPLETE (a later scheduled run will retry)'}")

    if not priced:
        print("No results gathered. Files not updated.")
        set_output("updated", "false")
        return
    if not overwrite and last.get("market_date") == date_str and n_clean < last.get("n_clean", 0):
        print(f"An earlier run today had better data ({last['n_clean']} clean tickers); keeping it.")
        set_output("updated", "false")
        return

    hist_df, hist52_df = update_history_files(date_str, results)
    render_site(results, format_chart_payload(hist_df),
                format_chart_payload(hist52_df, "Q(S <= 50% of high)"), market_date)
    with open(LAST_RUN_FILE, "w") as f:
        json.dump({"market_date": date_str, "n_clean": n_clean, "n_priced": len(priced),
                   "n_errors": n_errors, "complete": complete,
                   "run_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
                   "rows": [{k: v for k, v in r.items() if k != "skew"} for r in results]},
                  f, indent=1, default=_jsonable)
    set_output("updated", "true")
    print(f"Wrote {OUTPUT_FILE}, {HISTORY_FILE}, {HISTORY52_FILE}, {SKEW_FILE}.")


if __name__ == "__main__":
    main()
