def make_svg(lines, height=130, width=820):
    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="{height}">',
        '  <style>',
        '    .label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 15px; font-weight: 700; fill: #0f172a; }',
        '    .math { font-family: "Cambria Math", "Times New Roman", Times, serif; font-size: 17px; fill: #1e293b; font-style: italic; }',
        '    .math-normal { font-family: "Cambria Math", "Times New Roman", Times, serif; font-size: 17px; fill: #1e293b; font-style: normal; }',
        '    .desc { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 14px; fill: #334155; }',
        '    .sub { font-size: 11px; }',
        '    .sup { font-size: 11px; }',
        '  </style>',
        f'  <rect width="{width}" height="{height}" fill="#ffffff" rx="6" />'
    ]
    for content, y_pos in lines:
        svg.append(f'  <text x="25" y="{y_pos}">{content}</text>')
    svg.append('</svg>')
    return '\n'.join(svg)

# Card 1: Target Date & Strike
with open("explain1.svg", "w", encoding="utf-8") as f:
    f.write(make_svg([
        ('<tspan class="label">Target Date: </tspan><tspan class="math">T</tspan><tspan class="sub">target</tspan><tspan class="math-normal"> = </tspan><tspan class="math">t</tspan><tspan class="sub">current</tspan><tspan class="math-normal"> + 2.0 years (730.5 days)</tspan>', 48),
        ('<tspan class="label">Downside 50% Strike: </tspan><tspan class="math">K</tspan><tspan class="math-normal"> = 0.50 × </tspan><tspan class="math">S</tspan><tspan class="sub">0</tspan>', 92)
    ], height=130))

# Small helpers for the cards below
def L(t): return f'<tspan class="label">{t}</tspan>'
def M(t): return f'<tspan class="math">{t}</tspan>'
def N(t): return f'<tspan class="math-normal">{t}</tspan>'
def SB(t): return f'<tspan class="sub">{t}</tspan>'
def SP(t): return f'<tspan class="sup">{t}</tspan>'
def D(t): return f'<tspan class="desc">{t}</tspan>'

# Card 2: Risk-Free Rate & Forward Price
with open("explain2.svg", "w", encoding="utf-8") as f:
    f.write(make_svg([
        (L("Dynamic Risk-Free Rate: ") + M("r") + N(" = Yield(2YY=F) / 100"), 42),
        (L("Live Forward (put-call parity): ") + M("F") + N(" = ") + M("K") + N(" + ") + M("e") + SP("rT") + N("(") + M("C") + N(" − ") + M("P") + N("), near-the-money strikes"), 84),
        (L("Historical Forward: ") + M("F") + N(" = ") + M("S") + SB("0") + M(" e") + SP("(r − q)T") + N(", ") + M("q") + N(" = trailing 12-month dividend yield"), 126),
    ], height=165))

# Card 3: Live smile construction
with open("explain3.svg", "w", encoding="utf-8") as f:
    f.write(make_svg([
        (L("Smile per expiry: ") + N("σ") + SB("i") + N("(") + M("k") + N(") fit to OTM put/call mids, ") + M("k") + N(" = ln(") + M("K") + N("/") + M("F") + N(")"), 42),
        (L("Interpolate to 2 years: ") + N("σ(") + M("k") + N(")²·2 = (1−") + M("w") + N(")·σ₁(") + M("k") + N(")²·") + M("T") + N("₁ + ") + M("w") + N("·σ₂(") + M("k") + N(")²·") + M("T") + N("₂,   ") + M("T") + N("₁ ≤ 2 &lt; ") + M("T") + N("₂"), 84),
        (D("If no expiry reaches 2 years: σ(k) at 2y = σ at the longest expiry, matched on k/√T (flagged *)."), 126),
    ], height=165))

# Card 4: Breeden-Litzenberger
with open("explain4.svg", "w", encoding="utf-8") as f:
    f.write(make_svg([
        (L("Risk-Neutral Tail Probability: ") + M("Q") + N("(") + M("S") + SB("T") + N(" ≤ ") + M("K") + N(") = ") + M("e") + SP("rT") + N(" ∂") + M("Put") + N("(") + M("K") + N(", ") + M("T") + N(")/∂") + M("K") + N(",  ") + M("K") + N(" = 0.5 ") + M("S") + SB("0"), 48),
        (N("= N(−") + M("d") + SB("2") + N(") + ") + M("e") + SP("rT") + N(" · Vega · ∂σ/∂") + M("K") + D("     (the second term is the skew correction)"), 92),
    ], height=130))

# Card 5: Historical calibration
with open("explain5.svg", "w", encoding="utf-8") as f:
    f.write(make_svg([
        (L("Two-year volatility proxy: ") + N("σ") + SB("proxy") + N("² = 0.3 · (RV") + SB("63d") + N(")² + 0.7 · (RV") + SB("504d") + N(")²"), 42),
        (L("Per-ticker multiplier: ") + M("m") + N(" = median over live days of σ") + SB("eff") + N(" / σ") + SB("proxy") + N(",  where N(−") + M("d") + SB("2") + N("(σ") + SB("eff") + N(")) = ") + M("Q") + SB("live"), 84),
        (L("Historical probability: ") + M("Q") + N(" = N(−") + M("d") + SB("2") + N(") with σ = ") + M("m") + N(" · σ") + SB("proxy"), 126),
    ], height=165))

# Card 6: Physical Tail Probability P
with open("explain6.svg", "w", encoding="utf-8") as f:
    f.write(make_svg([
        (L("Physical Equity Drift (CAPM): ") + N("μ = ") + M("r") + N(" + β × ERP − ") + M("q") + N(", where ERP = 4.5%"), 48),
        (L("Physical Tail Probability: ") + M("P") + N("(") + M("S") + SB("T") + N(" ≤ ") + M("K") + N(") = ") + M("Q") + N("(") + M("S") + SB("T") + N(" ≤ ") + M("K") + N("·") + M("e") + SP("−β·ERP·T") + N(")"), 92),
    ], height=130))

print("Created explain1.svg through explain6.svg.")
