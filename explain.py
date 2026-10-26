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

# Card 2: Risk-Free Rate & Forward Price
with open("explain2.svg", "w", encoding="utf-8") as f:
    f.write(make_svg([
        ('<tspan class="label">Dynamic Risk-Free Rate: </tspan><tspan class="math">r</tspan><tspan class="math-normal"> = Yield(2YY=F) / 100</tspan>', 48),
        ('<tspan class="label">Forward Equity Price: </tspan><tspan class="math">F</tspan><tspan class="math-normal"> = </tspan><tspan class="math">S</tspan><tspan class="sub">0</tspan><tspan class="math"> e</tspan><tspan class="sup">(r − q)T</tspan><tspan class="math-normal">, where </tspan><tspan class="math">q</tspan><tspan class="math-normal"> = 0.005, </tspan><tspan class="math">T</tspan><tspan class="math-normal"> = 2.0</tspan>', 92)
    ], height=130))

# Card 3: Volatility Estimation Across Regimes
with open("explain3.svg", "w", encoding="utf-8") as f:
    f.write(make_svg([
        ('<tspan class="label">Live Regime (Oct 2026+): </tspan><tspan class="math-normal">σ</tspan><tspan class="sub">ATM</tspan><tspan class="math-normal"> = median(σ</tspan><tspan class="sub">LEAPS calls</tspan><tspan class="math-normal">, σ</tspan><tspan class="sub">LEAPS puts</tspan><tspan class="math-normal">)</tspan>', 42),
        ('<tspan class="label">Historical Proxy (2021–2026): </tspan><tspan class="math-normal">σ</tspan><tspan class="sub">RV</tspan><tspan class="math-normal"> = std(ln(</tspan><tspan class="math">S</tspan><tspan class="sub">τ</tspan><tspan class="math-normal">/</tspan><tspan class="math">S</tspan><tspan class="sub">τ−1</tspan><tspan class="math-normal">))</tspan><tspan class="sub">63d</tspan><tspan class="math-normal"> × √252</tspan>', 84),
        ('<tspan class="label">Implied Volatility Scaling: </tspan><tspan class="math-normal">σ</tspan><tspan class="sub">ATM</tspan><tspan class="math-normal"> ≈ σ</tspan><tspan class="sub">RV</tspan><tspan class="math-normal"> × VRP (with VRP = 1.12)</tspan>', 126)
    ], height=165))

# Card 4: Downside Skew Modeling
with open("explain4.svg", "w", encoding="utf-8") as f:
    f.write(make_svg([
        ('<tspan class="label">Downside Wing Volatility: </tspan><tspan class="math-normal">σ</tspan><tspan class="sub">wing</tspan><tspan class="math-normal"> = 1.18 × σ</tspan><tspan class="sub">ATM</tspan>', 48),
        ('<tspan class="desc">Captures structural implied volatility smile for deep 50% out-of-the-money puts over a 2-year horizon.</tspan>', 90)
    ], height=130))

# Card 5: Risk-Neutral Tail Probability Q
with open("explain5.svg", "w", encoding="utf-8") as f:
    f.write(make_svg([
        ('<tspan class="label">Risk-Neutral Tail Probability: </tspan><tspan class="math">Q</tspan><tspan class="math-normal">(</tspan><tspan class="math">S</tspan><tspan class="sub">T</tspan><tspan class="math-normal"> ≤ </tspan><tspan class="math">K</tspan><tspan class="math-normal">) = N(−</tspan><tspan class="math">d</tspan><tspan class="sub">2</tspan><tspan class="math-normal">)</tspan>', 48),
        ('<tspan class="math-normal">where </tspan><tspan class="math">d</tspan><tspan class="sub">2</tspan><tspan class="math-normal"> = [ ln(</tspan><tspan class="math">S</tspan><tspan class="sub">0</tspan><tspan class="math-normal">/</tspan><tspan class="math">K</tspan><tspan class="math-normal">) + (</tspan><tspan class="math">r</tspan><tspan class="math-normal"> − </tspan><tspan class="math">q</tspan><tspan class="math-normal"> − ½σ</tspan><tspan class="sub">wing</tspan><tspan class="sup">2</tspan><tspan class="math-normal">)</tspan><tspan class="math">T</tspan><tspan class="math-normal"> ] / ( σ</tspan><tspan class="sub">wing</tspan><tspan class="math-normal">√</tspan><tspan class="math">T</tspan><tspan class="math-normal"> )</tspan>', 92)
    ], height=130))

# Card 6: Physical Tail Probability P
with open("explain6.svg", "w", encoding="utf-8") as f:
    f.write(make_svg([
        ('<tspan class="label">Physical Equity Drift (CAPM): </tspan><tspan class="math-normal">μ = </tspan><tspan class="math">r</tspan><tspan class="math-normal"> + β × ERP − q, where ERP = 4.5%</tspan>', 42),
        ('<tspan class="label">Expected Future Price: </tspan><tspan class="math">E</tspan><tspan class="sub">P</tspan><tspan class="math-normal">[</tspan><tspan class="math">S</tspan><tspan class="sub">T</tspan><tspan class="math-normal">] = </tspan><tspan class="math">S</tspan><tspan class="sub">0</tspan><tspan class="math"> e</tspan><tspan class="sup">μT</tspan>', 84),
        ('<tspan class="label">Physical Tail Probability: </tspan><tspan class="math">P</tspan><tspan class="math-normal">(</tspan><tspan class="math">S</tspan><tspan class="sub">T</tspan><tspan class="math-normal"> ≤ </tspan><tspan class="math">K</tspan><tspan class="math-normal">) = N(−</tspan><tspan class="math">d</tspan><tspan class="sub">2</tspan><tspan class="sup">P</tspan><tspan class="math-normal">), where </tspan><tspan class="math">d</tspan><tspan class="sub">2</tspan><tspan class="sup">P</tspan><tspan class="math-normal"> = [ ln(</tspan><tspan class="math">E</tspan><tspan class="sub">P</tspan><tspan class="math-normal">[</tspan><tspan class="math">S</tspan><tspan class="sub">T</tspan><tspan class="math-normal">]/</tspan><tspan class="math">K</tspan><tspan class="math-normal">) − ½σ</tspan><tspan class="sub">wing</tspan><tspan class="sup">2</tspan><tspan class="math">T</tspan><tspan class="math-normal"> ] / ( σ</tspan><tspan class="sub">wing</tspan><tspan class="math-normal">√</tspan><tspan class="math">T</tspan><tspan class="math-normal"> )</tspan>', 126)
    ], height=165))

print("Created explain1.svg through explain6.svg.")
