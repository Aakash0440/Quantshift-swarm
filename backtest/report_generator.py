# backtest/report_generator.py
# Generates a PDF summary report after each backtest run.
# Uses reportlab — pure Python, no LaTeX required.

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart
from datetime import datetime
import os


def generate_report(
    backtest_result,
    output_path: str = "/tmp/quantshift_backtest_report.pdf",
    title: str = "QUANTSHIFT v1 — Backtest Report",
) -> str:
    """
    Generate a PDF report from a BacktestResult.
    Returns the path to the generated PDF.
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    elements = []

    # ── Header ────────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        fontSize=18, spaceAfter=6, textColor=colors.HexColor("#1a1a2e"),
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=10, textColor=colors.grey, spaceAfter=20,
    )

    elements.append(Paragraph(title, title_style))
    elements.append(Paragraph(
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        subtitle_style,
    ))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0")))
    elements.append(Spacer(1, 0.4*cm))

    # ── Performance Summary Table ─────────────────────────────────────────
    metrics = backtest_result.metrics or {}

    elements.append(Paragraph("Performance Summary", styles["Heading2"]))
    elements.append(Spacer(1, 0.2*cm))

    def pct(val):
        return f"{float(val):+.2f}%"
    def num(val, decimals=3):
        return f"{float(val):.{decimals}f}"
    def color_val(val, good_if_positive=True):
        val = float(val)
        if good_if_positive:
            return colors.HexColor("#27ae60") if val > 0 else colors.HexColor("#e74c3c")
        else:
            return colors.HexColor("#e74c3c") if val > 0 else colors.HexColor("#27ae60")

    summary_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Total Return",        pct(metrics.get("total_return_pct", 0)),
         "Win Rate",            f"{float(metrics.get('win_rate', 0)):.1%}"],
        ["Annualized Return",   pct(metrics.get("annualized_return_pct", 0)),
         "Profit Factor",       num(metrics.get("profit_factor", 0), 2)],
        ["Sharpe Ratio",        num(metrics.get("sharpe_ratio", 0)),
         "Avg Win",             pct(metrics.get("avg_win_pct", 0))],
        ["Sortino Ratio",       num(metrics.get("sortino_ratio", 0)),
         "Avg Loss",            pct(metrics.get("avg_loss_pct", 0))],
        ["Max Drawdown",        pct(metrics.get("max_drawdown_pct", 0)),
         "Total Trades",        str(int(metrics.get("n_trades", 0)))],
        ["Calmar Ratio",        num(metrics.get("calmar_ratio", 0)),
         "Avg Hold Time",       f"{float(metrics.get('avg_holding_hours', 0)):.1f}h"],
        ["Starting Capital",    f"${backtest_result.starting_capital:,.0f}",
         "Final Capital",       f"${backtest_result.final_capital:,.0f}"],
    ]

    table_style = TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  10),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 9),
        ("BACKGROUND",   (0, 1), (-1, -1), colors.HexColor("#f9f9f9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ALIGN",        (1, 1), (1, -1),  "RIGHT"),
        ("ALIGN",        (3, 1), (3, -1),  "RIGHT"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("FONTNAME",     (0, 1), (0, -1),  "Helvetica-Bold"),
        ("FONTNAME",     (2, 1), (2, -1),  "Helvetica-Bold"),
    ])

    t = Table(summary_data, colWidths=[4.5*cm, 3.5*cm, 4.5*cm, 3.5*cm])
    t.setStyle(table_style)
    elements.append(t)
    elements.append(Spacer(1, 0.6*cm))

    # ── Equity Curve (simple ASCII-style via reportlab LinePlot) ──────────
    if backtest_result.equity_curve and len(backtest_result.equity_curve) > 2:
        elements.append(Paragraph("Equity Curve", styles["Heading2"]))
        elements.append(Spacer(1, 0.2*cm))

        equity = backtest_result.equity_curve
        # Downsample to max 200 points for the chart
        step = max(1, len(equity) // 200)
        sampled = equity[::step]

        drawing = Drawing(400, 150)
        lp = LinePlot()
        lp.x = 30
        lp.y = 10
        lp.width = 360
        lp.height = 130

        data_points = [(i, v) for i, v in enumerate(sampled)]
        lp.data = [data_points]
        lp.lines[0].strokeColor = colors.HexColor("#27ae60")
        lp.lines[0].strokeWidth = 1.5
        lp.xValueAxis.valueMin = 0
        lp.xValueAxis.valueMax = len(sampled)
        lp.yValueAxis.valueMin = min(sampled) * 0.98
        lp.yValueAxis.valueMax = max(sampled) * 1.02

        drawing.add(lp)
        elements.append(drawing)
        elements.append(Spacer(1, 0.6*cm))

    # ── Trade List (last 20) ──────────────────────────────────────────────
    trades = backtest_result.trades
    if trades:
        elements.append(Paragraph(
            f"Recent Trades (showing last {min(20, len(trades))} of {len(trades)})",
            styles["Heading2"],
        ))
        elements.append(Spacer(1, 0.2*cm))

        trade_data = [["Ticker", "Direction", "Entry", "Exit", "P&L %", "Confidence", "Regime"]]
        for trade in trades[-20:]:
            pnl_color = colors.HexColor("#27ae60") if trade.pnl >= 0 else colors.HexColor("#e74c3c")
            trade_data.append([
                trade.ticker,
                trade.direction,
                f"${trade.entry_price:.2f}",
                f"${trade.exit_price:.2f}" if trade.exit_price else "Open",
                f"{trade.pnl_pct:+.2f}%",
                f"{trade.signal_confidence:.0%}",
                trade.regime,
            ])

        trade_table_style = TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#2c3e50")),
            ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 8),
            ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f8f8")]),
            ("ALIGN",        (2, 1), (4, -1),  "RIGHT"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING",   (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ])

        tt = Table(trade_data, colWidths=[2*cm, 2.2*cm, 2.5*cm, 2.5*cm, 2*cm, 2.5*cm, 2.3*cm])
        tt.setStyle(trade_table_style)
        elements.append(tt)
        elements.append(Spacer(1, 0.6*cm))

    # ── Footer ────────────────────────────────────────────────────────────
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    elements.append(Spacer(1, 0.2*cm))
    footer_style = ParagraphStyle("Footer", parent=styles["Normal"], fontSize=7, textColor=colors.grey)
    elements.append(Paragraph(
        "QUANTSHIFT v1 — Not financial advice. Backtest results do not guarantee future performance. "
        "Always validate on live paper trading before deploying real capital.",
        footer_style,
    ))

    doc.build(elements)
    print(f"[ReportGenerator] PDF saved to: {output_path}")
    return output_path
