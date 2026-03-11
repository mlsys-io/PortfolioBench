#!/usr/bin/env python3
"""
generate_report.py — Convert benchmark JSON output into a beautiful HTML report.

Usage:
    python generate_report.py benchmark_results.json -o report.html
    python generate_report.py benchmark_results.json  # writes to benchmark_report.html
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional


def load_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _pct_class(val: Optional[float]) -> str:
    if val is None:
        return "neutral"
    return "positive" if val >= 0 else "negative"


def _fmt_pct(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    return f"{val:+.2f}%"


def _fmt_sharpe(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    return f"{val:.4f}"


def _sharpe_class(val: Optional[float]) -> str:
    if val is None:
        return "neutral"
    if val >= 1.0:
        return "positive"
    if val >= 0:
        return "neutral"
    return "negative"


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}m {secs:.1f}s"


def _status_badge(status: str) -> str:
    cls = {"pass": "badge-pass", "fail": "badge-fail", "skip": "badge-skip",
           "error": "badge-fail", "ok": "badge-pass"}.get(status, "badge-skip")
    label = status.upper()
    return f'<span class="badge {cls}">{label}</span>'


def _build_summary_card(data: Dict[str, Any]) -> str:
    """Build the top-level summary card."""
    summary = data.get("summary", {})
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    skipped = summary.get("skipped", 0)
    duration = summary.get("duration_s", 0)
    mode = data.get("mode", "full")
    timestamp = data.get("timestamp", "")

    pass_pct = (passed / max(total, 1)) * 100
    fail_pct = (failed / max(total, 1)) * 100
    skip_pct = 100 - pass_pct - fail_pct

    overall_status = "PASS" if failed == 0 else "FAIL"
    overall_class = "summary-pass" if failed == 0 else "summary-fail"

    return f"""
    <div class="summary-card {overall_class}">
      <div class="summary-header">
        <h2>Benchmark Summary</h2>
        <span class="badge {'badge-pass' if failed == 0 else 'badge-fail'} badge-lg">{overall_status}</span>
      </div>
      <div class="summary-meta">
        <span>Mode: <strong>{mode.title()}</strong></span>
        <span>Duration: <strong>{_fmt_duration(duration)}</strong></span>
        <span>Date: <strong>{timestamp[:19] if timestamp else 'N/A'}</strong></span>
      </div>
      <div class="progress-bar">
        <div class="progress-pass" style="width:{pass_pct:.1f}%"></div>
        <div class="progress-fail" style="width:{fail_pct:.1f}%"></div>
        <div class="progress-skip" style="width:{skip_pct:.1f}%"></div>
      </div>
      <div class="summary-counts">
        <div class="count-box count-pass">
          <div class="count-number">{passed}</div>
          <div class="count-label">Passed</div>
        </div>
        <div class="count-box count-fail">
          <div class="count-number">{failed}</div>
          <div class="count-label">Failed</div>
        </div>
        <div class="count-box count-skip">
          <div class="count-number">{skipped}</div>
          <div class="count-label">Skipped</div>
        </div>
        <div class="count-box count-total">
          <div class="count-number">{total}</div>
          <div class="count-label">Total</div>
        </div>
      </div>
    </div>
    """


def _build_data_integrity_section(data: Dict[str, Any]) -> str:
    """Build the data integrity section."""
    di = data.get("data_integrity", {})
    if not di:
        return ""

    status = di.get("status", "skip")
    details = di.get("details", {})
    duration = di.get("duration_s", 0)

    rows = ""
    for key in ["total_files", "valid_files", "invalid_files", "unique_assets"]:
        val = details.get(key, "N/A")
        label = key.replace("_", " ").title()
        rows += f"<tr><td>{label}</td><td>{val}</td></tr>\n"

    tfs = details.get("timeframes", [])
    if tfs:
        rows += f"<tr><td>Timeframes</td><td>{', '.join(tfs)}</td></tr>\n"

    invalid_list = details.get("invalid_list", [])
    invalid_html = ""
    if invalid_list:
        items = "".join(f"<li>{f}</li>" for f in invalid_list)
        invalid_html = f'<div class="alert alert-warn"><strong>Invalid files:</strong><ul>{items}</ul></div>'

    return f"""
    <div class="section">
      <div class="section-header">
        <h3>Data Integrity Check</h3>
        {_status_badge(status)}
        <span class="duration">{_fmt_duration(duration)}</span>
      </div>
      <table class="detail-table">
        <tbody>{rows}</tbody>
      </table>
      {invalid_html}
    </div>
    """


def _build_alpha_section(data: Dict[str, Any]) -> str:
    """Build the alpha factor smoke test section."""
    alpha = data.get("alpha_smoke_test", {})
    if not alpha:
        return ""

    status = alpha.get("status", "skip")
    duration = alpha.get("duration_s", 0)
    details = alpha.get("details", {})

    rows = ""
    for name, info in details.items():
        cols = ", ".join(info.get("columns_added", []))
        row_count = info.get("rows", "N/A")
        rows += f"""
        <tr>
          <td><strong>{name.upper()}</strong></td>
          <td>{row_count} rows</td>
          <td class="mono">{cols}</td>
        </tr>
        """

    return f"""
    <div class="section">
      <div class="section-header">
        <h3>Alpha Factor Smoke Test</h3>
        {_status_badge(status)}
        <span class="duration">{_fmt_duration(duration)}</span>
      </div>
      <table class="detail-table">
        <thead><tr><th>Factor</th><th>Data</th><th>Columns Added</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """


def _build_pipeline_section(data: Dict[str, Any]) -> str:
    """Build the standalone portfolio pipeline section."""
    pipe = data.get("portfolio_pipeline", {})
    if not pipe:
        return ""

    status = pipe.get("status", "skip")
    duration = pipe.get("duration_s", 0)
    metrics = pipe.get("metrics", {})

    if not metrics:
        return f"""
        <div class="section">
          <div class="section-header">
            <h3>Standalone Portfolio Pipeline</h3>
            {_status_badge(status)}
            <span class="duration">{_fmt_duration(duration)}</span>
          </div>
          <p class="muted">No metrics available.</p>
        </div>
        """

    rows = ""
    for key, label in [
        ("total_return_pct", "Total Return"),
        ("annualised_return_pct", "Annualised Return"),
        ("annualised_sharpe", "Sharpe Ratio"),
        ("max_drawdown_pct", "Max Drawdown"),
        ("n_bars", "Bars"),
    ]:
        val = metrics.get(key)
        if val is None:
            formatted = "N/A"
            cls = ""
        elif key.endswith("_pct"):
            formatted = _fmt_pct(val)
            cls = _pct_class(val) if "drawdown" not in key else ("negative" if val < -10 else "neutral")
        elif key == "annualised_sharpe":
            formatted = _fmt_sharpe(val)
            cls = _sharpe_class(val)
        else:
            formatted = str(val)
            cls = ""
        rows += f'<tr><td>{label}</td><td class="{cls}">{formatted}</td></tr>\n'

    return f"""
    <div class="section">
      <div class="section-header">
        <h3>Standalone Portfolio Pipeline</h3>
        {_status_badge(status)}
        <span class="duration">{_fmt_duration(duration)}</span>
      </div>
      <table class="detail-table">
        <tbody>{rows}</tbody>
      </table>
    </div>
    """


def _build_backtest_section(title: str, backtests: List[Dict]) -> str:
    """Build a backtest results section with leaderboard and detail grid."""
    if not backtests:
        return ""

    passed = sum(1 for r in backtests if r.get("status") == "pass")
    failed = sum(1 for r in backtests if r.get("status") == "fail")
    total = len(backtests)

    # Leaderboard: best return per strategy
    strat_best: Dict[str, Dict] = {}
    for r in backtests:
        name = r.get("strategy", "?")
        ret = r.get("metrics", {}).get("total_return_pct")
        if ret is None:
            continue
        if name not in strat_best or ret > strat_best[name]["return"]:
            strat_best[name] = {
                "return": ret,
                "sharpe": r.get("metrics", {}).get("sharpe"),
                "trades": r.get("metrics", {}).get("trades", 0),
                "asset_class": r.get("asset_class", "?"),
                "timeframe": r.get("timeframe", "?"),
                "drawdown": r.get("metrics", {}).get("max_drawdown_pct"),
                "win_rate": r.get("metrics", {}).get("win_rate_pct"),
                "profit_factor": r.get("metrics", {}).get("profit_factor"),
            }

    ranked = sorted(strat_best.items(), key=lambda x: x[1]["return"], reverse=True)

    leaderboard_rows = ""
    for i, (name, info) in enumerate(ranked, 1):
        medal = {1: "gold", 2: "silver", 3: "bronze"}.get(i, "")
        medal_icon = {1: "&#129351;", 2: "&#129352;", 3: "&#129353;"}.get(i, str(i))
        ret_cls = _pct_class(info["return"])
        sharpe_cls = _sharpe_class(info.get("sharpe"))
        dd_cls = "negative" if info.get("drawdown") is not None and info["drawdown"] < -10 else "neutral"

        leaderboard_rows += f"""
        <tr class="{medal}">
          <td class="rank">{medal_icon}</td>
          <td class="strat-name">{name}</td>
          <td class="{ret_cls}">{_fmt_pct(info['return'])}</td>
          <td class="{sharpe_cls}">{_fmt_sharpe(info.get('sharpe'))}</td>
          <td>{info.get('trades', 'N/A')}</td>
          <td class="{dd_cls}">{_fmt_pct(info.get('drawdown'))}</td>
          <td>{_fmt_pct(info.get('win_rate'))}</td>
          <td>{_fmt_sharpe(info.get('profit_factor'))}</td>
          <td class="mono">{info['asset_class']}/{info['timeframe']}</td>
        </tr>
        """

    # Detail grid: per strategy x asset x timeframe
    detail_rows = ""
    for r in backtests:
        status = r.get("status", "?")
        m = r.get("metrics", {})
        detail_rows += f"""
        <tr>
          <td>{r.get('strategy', '?')}</td>
          <td>{r.get('asset_class', '?')}</td>
          <td>{r.get('timeframe', '?')}</td>
          <td>{_status_badge(status)}</td>
          <td class="{_pct_class(m.get('total_return_pct'))}">{_fmt_pct(m.get('total_return_pct'))}</td>
          <td class="{_sharpe_class(m.get('sharpe'))}">{_fmt_sharpe(m.get('sharpe'))}</td>
          <td>{m.get('trades', 'N/A')}</td>
          <td>{_fmt_duration(r.get('duration_s', 0))}</td>
        </tr>
        """

    return f"""
    <div class="section">
      <div class="section-header">
        <h3>{title}</h3>
        <span class="badge badge-info">{passed}/{total} passed</span>
      </div>

      <h4 class="subsection-title">Leaderboard</h4>
      <div class="table-wrapper">
        <table class="leaderboard-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Strategy</th>
              <th>Best Return</th>
              <th>Sharpe</th>
              <th>Trades</th>
              <th>Max DD</th>
              <th>Win Rate</th>
              <th>Profit Factor</th>
              <th>Best On</th>
            </tr>
          </thead>
          <tbody>{leaderboard_rows}</tbody>
        </table>
      </div>

      <details class="detail-toggle">
        <summary>Show all {total} backtest results</summary>
        <div class="table-wrapper">
          <table class="detail-grid-table">
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Asset Class</th>
                <th>Timeframe</th>
                <th>Status</th>
                <th>Return</th>
                <th>Sharpe</th>
                <th>Trades</th>
                <th>Duration</th>
              </tr>
            </thead>
            <tbody>{detail_rows}</tbody>
          </table>
        </div>
      </details>
    </div>
    """


def generate_html(data: Dict[str, Any]) -> str:
    """Generate the complete HTML report."""
    timestamp = data.get("timestamp", datetime.now().isoformat())

    summary_html = _build_summary_card(data)
    data_integrity_html = _build_data_integrity_section(data)
    alpha_html = _build_alpha_section(data)
    pipeline_html = _build_pipeline_section(data)
    trading_html = _build_backtest_section(
        "Trading Strategy Backtests", data.get("trading_backtests", [])
    )
    portfolio_html = _build_backtest_section(
        "Portfolio Strategy Backtests", data.get("portfolio_backtests", [])
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PortfolioBench — Benchmark Report</title>
<style>
  :root {{
    --bg: #0d1117;
    --surface: #161b22;
    --surface-2: #1c2129;
    --border: #30363d;
    --text: #e6edf3;
    --text-dim: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --green-bg: rgba(63,185,80,0.12);
    --red: #f85149;
    --red-bg: rgba(248,81,73,0.12);
    --yellow: #d29922;
    --yellow-bg: rgba(210,153,34,0.12);
    --blue: #58a6ff;
    --blue-bg: rgba(88,166,255,0.12);
    --purple: #bc8cff;
    --gold: #f0c040;
    --silver: #c0c0c0;
    --bronze: #cd7f32;
    --radius: 8px;
    --shadow: 0 1px 3px rgba(0,0,0,0.3);
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 0;
  }}

  .container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 24px 20px;
  }}

  /* Header */
  .report-header {{
    text-align: center;
    padding: 40px 20px 32px;
    background: linear-gradient(135deg, #161b22 0%, #1a2332 50%, #161b22 100%);
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
  }}

  .report-header h1 {{
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
  }}

  .report-header .subtitle {{
    color: var(--text-dim);
    font-size: 0.95rem;
  }}

  /* Summary Card */
  .summary-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: var(--shadow);
  }}

  .summary-pass {{ border-left: 4px solid var(--green); }}
  .summary-fail {{ border-left: 4px solid var(--red); }}

  .summary-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
  }}

  .summary-header h2 {{ font-size: 1.3rem; font-weight: 600; }}

  .summary-meta {{
    display: flex;
    gap: 24px;
    color: var(--text-dim);
    font-size: 0.875rem;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }}

  .summary-meta strong {{ color: var(--text); }}

  .progress-bar {{
    display: flex;
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
    background: var(--surface-2);
    margin-bottom: 20px;
  }}

  .progress-pass {{ background: var(--green); }}
  .progress-fail {{ background: var(--red); }}
  .progress-skip {{ background: var(--yellow); }}

  .summary-counts {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
  }}

  .count-box {{
    text-align: center;
    padding: 16px 8px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
  }}

  .count-pass {{ background: var(--green-bg); }}
  .count-fail {{ background: var(--red-bg); }}
  .count-skip {{ background: var(--yellow-bg); }}
  .count-total {{ background: var(--blue-bg); }}

  .count-number {{
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1.2;
  }}

  .count-pass .count-number {{ color: var(--green); }}
  .count-fail .count-number {{ color: var(--red); }}
  .count-skip .count-number {{ color: var(--yellow); }}
  .count-total .count-number {{ color: var(--accent); }}

  .count-label {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-dim);
    margin-top: 4px;
  }}

  /* Badges */
  .badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}

  .badge-pass {{ background: var(--green-bg); color: var(--green); border: 1px solid rgba(63,185,80,0.3); }}
  .badge-fail {{ background: var(--red-bg); color: var(--red); border: 1px solid rgba(248,81,73,0.3); }}
  .badge-skip {{ background: var(--yellow-bg); color: var(--yellow); border: 1px solid rgba(210,153,34,0.3); }}
  .badge-info {{ background: var(--blue-bg); color: var(--accent); border: 1px solid rgba(88,166,255,0.3); }}
  .badge-lg {{ padding: 4px 16px; font-size: 0.85rem; }}

  /* Sections */
  .section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: var(--shadow);
  }}

  .section-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }}

  .section-header h3 {{
    font-size: 1.1rem;
    font-weight: 600;
    margin: 0;
  }}

  .duration {{
    color: var(--text-dim);
    font-size: 0.8rem;
    margin-left: auto;
  }}

  .subsection-title {{
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--accent);
    margin: 16px 0 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
  }}

  /* Tables */
  .table-wrapper {{
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
  }}

  th, td {{
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}

  th {{
    font-weight: 600;
    color: var(--text-dim);
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.04em;
    background: var(--surface-2);
    position: sticky;
    top: 0;
  }}

  tbody tr:hover {{ background: rgba(88,166,255,0.04); }}

  .detail-table {{ max-width: 500px; }}
  .detail-table td:first-child {{ color: var(--text-dim); width: 40%; }}

  /* Leaderboard */
  .leaderboard-table .rank {{ width: 40px; text-align: center; font-size: 1.1rem; }}
  .leaderboard-table .strat-name {{ font-weight: 600; }}

  tr.gold td {{ background: rgba(240,192,64,0.06); }}
  tr.silver td {{ background: rgba(192,192,192,0.04); }}
  tr.bronze td {{ background: rgba(205,127,50,0.04); }}

  /* Value coloring */
  .positive {{ color: var(--green); font-weight: 600; }}
  .negative {{ color: var(--red); font-weight: 600; }}
  .neutral {{ color: var(--text-dim); }}
  .mono {{ font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace; font-size: 0.8rem; }}

  /* Detail toggle */
  .detail-toggle {{
    margin-top: 16px;
  }}

  .detail-toggle summary {{
    cursor: pointer;
    color: var(--accent);
    font-size: 0.875rem;
    font-weight: 500;
    padding: 8px 0;
    user-select: none;
  }}

  .detail-toggle summary:hover {{ text-decoration: underline; }}

  .detail-toggle[open] summary {{ margin-bottom: 12px; }}

  /* Alerts */
  .alert {{
    padding: 12px 16px;
    border-radius: var(--radius);
    margin-top: 12px;
    font-size: 0.85rem;
  }}

  .alert-warn {{
    background: var(--yellow-bg);
    border: 1px solid rgba(210,153,34,0.3);
    color: var(--yellow);
  }}

  .alert ul {{ margin: 8px 0 0 20px; }}
  .alert li {{ margin-bottom: 2px; }}

  .muted {{ color: var(--text-dim); font-style: italic; }}

  /* Footer */
  .report-footer {{
    text-align: center;
    padding: 24px;
    color: var(--text-dim);
    font-size: 0.8rem;
    border-top: 1px solid var(--border);
    margin-top: 32px;
  }}

  .report-footer a {{
    color: var(--accent);
    text-decoration: none;
  }}

  .report-footer a:hover {{ text-decoration: underline; }}

  /* Responsive */
  @media (max-width: 640px) {{
    .summary-counts {{ grid-template-columns: repeat(2, 1fr); }}
    .summary-meta {{ flex-direction: column; gap: 4px; }}
    .report-header h1 {{ font-size: 1.5rem; }}
    .container {{ padding: 12px 10px; }}
  }}
</style>
</head>
<body>

<div class="report-header">
  <h1>PortfolioBench</h1>
  <div class="subtitle">Benchmark Report &mdash; {timestamp[:19] if timestamp else 'N/A'}</div>
</div>

<div class="container">
  {summary_html}
  {data_integrity_html}
  {alpha_html}
  {pipeline_html}
  {trading_html}
  {portfolio_html}
</div>

<div class="report-footer">
  Generated by <a href="https://github.com/mlsys-io/PortfolioBench">PortfolioBench</a>
  &middot; {timestamp[:10] if timestamp else ''}
</div>

</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate an HTML benchmark report from JSON results.",
    )
    parser.add_argument("input", help="Path to benchmark JSON file")
    parser.add_argument("-o", "--output", default="benchmark_report.html",
                        help="Output HTML file (default: benchmark_report.html)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    data = load_json(args.input)
    html = generate_html(data)

    with open(args.output, "w") as f:
        f.write(html)

    print(f"Report generated: {args.output}")


if __name__ == "__main__":
    main()
