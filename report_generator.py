from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import date
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── Palette ───────────────────────────────────────────────────────────────────
C_NAVY      = "#0F172A"
C_DARK      = "#1E293B"
C_MID       = "#334155"
C_MUTED     = "#64748B"
C_SUBTLE    = "#94A3B8"
C_BORDER    = "#E2E8F0"
C_BG_ALT    = "#F8FAFC"
C_BLUE      = "#2563EB"
C_BLUE_LIGHT= "#60A5FA"
C_GREEN     = "#10B981"
C_RED       = "#EF4444"
C_AMBER     = "#F59E0B"
C_ACCENT    = "#1E40AF"


def _hex(h):
    return colors.HexColor(h)


def _section_header(c, text, y, width):
    """Draw a section header with underline. Returns new y."""
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(_hex(C_NAVY))
    c.drawString(40, y, text)
    c.setStrokeColor(_hex(C_BORDER))
    c.setLineWidth(0.8)
    c.line(40, y - 6, width - 40, y - 6)
    return y - 24


def _page_footer(c, width, ticker):
    """Draw footer with disclaimer and branding."""
    c.setStrokeColor(_hex(C_BORDER))
    c.setLineWidth(0.5)
    c.line(40, 42, width - 40, 42)
    c.setFont("Helvetica-Oblique", 7.5)
    c.setFillColor(_hex(C_SUBTLE))
    c.drawString(40, 28,
        "Disclaimer: This report is generated algorithmically and does not constitute financial advice.")
    c.setFont("Helvetica", 7.5)
    c.drawRightString(width - 40, 28, f"MarketLab  |  {ticker}  |  {date.today().strftime('%B %d, %Y')}")


def save_charts(ticker, df):
    charts = []
    plt.style.use('seaborn-v0_8-darkgrid')

    # ── 1. Price & Moving Averages ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['Close'],  label='Close Price',          color='#111827', linewidth=2)
    ax.plot(df['MA50'],   label='MA50 (Medium Term)',   linestyle='--',  color='#3B82F6', alpha=0.85)
    ax.plot(df['MA200'],  label='MA200 (Long Term)',    linestyle='--',  color='#EF4444', alpha=0.85)
    ax.plot(df['EMA20'],  label='EMA20 (Short Term)',                    color='#10B981', alpha=0.85)
    ax.set_title(f'{ticker} — Price Action & Moving Averages',
                 fontsize=13, fontweight='bold', color='#1F2937', pad=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    path = f'{ticker}_ma.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    charts.append(path)

    # ── 2. Bollinger Bands ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['Close'],    color='#111827', label='Close Price',          linewidth=1.5)
    ax.plot(df['BB_upper'], color='#EF4444', alpha=0.7,  label='Upper Band (Resistance)')
    ax.plot(df['BB_lower'], color='#10B981', alpha=0.7,  label='Lower Band (Support)')
    ax.fill_between(df.index, df['BB_lower'], df['BB_upper'], alpha=0.08, color='#6B7280')
    ax.set_title(f'{ticker} — Volatility (Bollinger Bands)',
                 fontsize=13, fontweight='bold', color='#1F2937', pad=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    path = f'{ticker}_bb.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    charts.append(path)

    # ── 3. RSI ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['RSI'], color='#8B5CF6', label='RSI (14)', linewidth=2)
    ax.axhline(y=70, color='#EF4444', linestyle='--', alpha=0.8, label='Overbought (>70)')
    ax.axhline(y=30, color='#10B981', linestyle='--', alpha=0.8, label='Oversold (<30)')
    ax.fill_between(df.index, 70, df['RSI'], where=(df['RSI'] >= 70), color='#EF4444', alpha=0.15)
    ax.fill_between(df.index, 30, df['RSI'], where=(df['RSI'] <= 30), color='#10B981', alpha=0.15)
    ax.set_ylim(0, 100)
    ax.set_title(f'{ticker} — Relative Strength Index (Momentum)',
                 fontsize=13, fontweight='bold', color='#1F2937', pad=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    path = f'{ticker}_rsi.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    charts.append(path)

    # ── 4. MACD ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['MACD'],   label='MACD Line',   color='#2563EB', linewidth=1.5)
    ax.plot(df['Signal'], label='Signal Line', color='#F59E0B', linewidth=1.5)
    hist_colors = ['#10B981' if v >= 0 else '#EF4444' for v in df['Histogram']]
    ax.bar(df.index, df['Histogram'], color=hist_colors, alpha=0.5, label='Histogram')
    ax.axhline(y=0, color='black', linewidth=0.6, alpha=0.5)
    ax.set_title(f'{ticker} — MACD (Trend & Momentum)',
                 fontsize=13, fontweight='bold', color='#1F2937', pad=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    path = f'{ticker}_macd.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    charts.append(path)

    # ── 5. Stochastic ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['%K'], label='%K (Fast)', color='#3B82F6', linewidth=1.5)
    ax.plot(df['%D'], label='%D (Slow)', color='#F59E0B', linewidth=1.5)
    ax.axhline(y=80, color='#EF4444', linestyle='--', alpha=0.8, label='Overbought (>80)')
    ax.axhline(y=20, color='#10B981', linestyle='--', alpha=0.8, label='Oversold (<20)')
    ax.set_ylim(0, 100)
    ax.set_title(f'{ticker} — Stochastic Oscillator',
                 fontsize=13, fontweight='bold', color='#1F2937', pad=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    path = f'{ticker}_stoch.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    charts.append(path)

    return charts


def get_qualitative_insights(df, metrics):
    insights = []

    rsi = df['RSI'].iloc[-1]
    if rsi >= 70:
        insights.append(
            f"<b>RSI ({rsi:.1f}):</b> Overbought territory. The asset may be overvalued short-term, "
            "suggesting a potential correction or consolidation ahead.")
    elif rsi <= 30:
        insights.append(
            f"<b>RSI ({rsi:.1f}):</b> Oversold territory. Heavy selling pressure may have created "
            "an undervalued setup with rebound potential.")
    else:
        insights.append(
            f"<b>RSI ({rsi:.1f}):</b> Neutral momentum. Neither overbought nor oversold — "
            "balanced market participation with no extreme signal.")

    macd = df['MACD'].iloc[-1]
    hist = df['Histogram'].iloc[-1]
    if macd > 0 and hist > 0:
        insights.append(
            "<b>MACD:</b> Positive and expanding — buyers are in control with accelerating "
            "<b>bullish</b> momentum.")
    elif macd < 0 and hist < 0:
        insights.append(
            "<b>MACD:</b> Negative and falling — sellers dominating, reflecting strong "
            "<b>bearish</b> pressure.")
    elif hist > 0:
        insights.append(
            "<b>MACD Histogram:</b> Turning positive — early signs of a bullish crossover "
            "with momentum shifting toward buyers.")

    beta = metrics.get('Beta', 1.0)
    if beta > 1.2:
        insights.append(
            f"<b>Beta ({beta:.2f}):</b> High systematic risk. This stock amplifies market moves — "
            "expect larger price swings in both directions.")
    elif beta < 0.8:
        insights.append(
            f"<b>Beta ({beta:.2f}):</b> Defensive profile. Low sensitivity to broad market "
            "fluctuations — suitable as a portfolio stabiliser.")

    sharpe = metrics.get('Sharpe Annualized', 0)
    if sharpe > 1:
        insights.append(
            f"<b>Sharpe Ratio ({sharpe:.2f}):</b> Excellent risk-adjusted returns — "
            "the asset has historically delivered superior compensation per unit of risk.")
    elif sharpe < 0:
        insights.append(
            f"<b>Sharpe Ratio ({sharpe:.2f}):</b> Negative risk-adjusted returns — "
            "historically the risk taken was not adequately compensated vs. a risk-free asset.")

    close  = df['Close'].iloc[-1]
    ma200  = df['MA200'].iloc[-1]
    if close > ma200:
        insights.append(
            f"<b>Long-term Trend:</b> Price is <b>above the 200-day MA</b> (${ma200:.2f}), "
            "confirming a structural macro uptrend.")
    else:
        insights.append(
            f"<b>Long-term Trend:</b> Price is <b>below the 200-day MA</b> (${ma200:.2f}), "
            "indicating structural long-term weakness.")

    return insights


def _draw_sentiment_section(c, ticker, sentiment, y_position, width, height, styles):
    reason_style = ParagraphStyle(
        'Reason', parent=styles['Normal'],
        fontSize=9.5, leading=14,
        textColor=_hex(C_MID), leftIndent=16
    )

    label      = sentiment.get('label',          'NEUTRAL')
    compound   = sentiment.get('compound',        0.0)
    confidence = sentiment.get('confidence',      0.0)
    news_count = sentiment.get('news_count',      0)
    pos_ratio  = sentiment.get('positive_ratio',  0.0)
    tail_risk  = sentiment.get('tail_risk',       False)
    headlines  = sentiment.get('headlines',       [])

    label_colors_map = {
        'BULLISH':          C_GREEN,
        'SLIGHTLY BULLISH': '#84CC16',
        'NEUTRAL':          C_SUBTLE,
        'SLIGHTLY BEARISH': C_AMBER,
        'BEARISH':          C_RED,
    }
    badge_color = _hex(label_colors_map.get(label, C_SUBTLE))

    y_position = _section_header(c, "News Sentiment Analysis", y_position, width)

    # Badge + stats row
    badge_h  = 22
    badge_y  = y_position - badge_h
    c.setFillColor(badge_color)
    c.roundRect(40, badge_y, 120, badge_h, 4, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(100, badge_y + 7, label)

    c.setFillColor(_hex(C_MID))
    c.setFont("Helvetica", 9)
    c.drawString(172, badge_y + 7,
        f"Compound: {compound:.4f}   |   Confidence: {confidence:.1f}%   |   Articles: {news_count}")
    y_position -= (badge_h + 14)

    # Pos / Neg bar
    bar_w   = width - 80
    bar_h   = 8
    bar_y   = y_position - bar_h
    pos_w   = bar_w * pos_ratio
    c.setFillColor(_hex(C_GREEN))
    c.roundRect(40, bar_y, pos_w, bar_h, 3, fill=True, stroke=False)
    c.setFillColor(_hex(C_RED))
    c.roundRect(40 + pos_w, bar_y, bar_w - pos_w, bar_h, 3, fill=True, stroke=False)
    c.setFont("Helvetica", 8)
    c.setFillColor(_hex(C_MUTED))
    c.drawString(40,            bar_y - 12, f"Positive {pos_ratio*100:.0f}%")
    c.drawRightString(width-40, bar_y - 12, f"{(1-pos_ratio)*100:.0f}% Negative")
    y_position -= (bar_h + 28)

    # Tail risk warning
    if tail_risk:
        box_h = 22
        box_y = y_position - box_h
        c.setFillColor(_hex("#FEF3C7"))
        c.roundRect(38, box_y, width - 76, box_h, 3, fill=True, stroke=False)
        c.setFillColor(_hex("#92400E"))
        c.setFont("Helvetica-Bold", 9)
        c.drawString(48, box_y + 7,
            "TAIL RISK DETECTED: Strong negative news — compound score adjusted.")
        y_position -= (box_h + 14)

    # Headlines
    if headlines:
        c.setFont("Helvetica-Bold", 9.5)
        c.setFillColor(_hex(C_NAVY))
        c.drawString(40, y_position, "Top Headlines:")
        y_position -= 16
        icons = {'Positive': '[+]', 'Negative': '[-]', 'Neutral': '[~]'}
        for h in headlines[:5]:
            icon = icons.get(h['sentiment'], '[~]')
            text = f"{icon}  {h['title'][:88]}  ({h['compound']:.3f})"
            p = Paragraph(text, reason_style)
            _, ph = p.wrap(width - 80, height)
            p.drawOn(c, 40, y_position - ph)
            y_position -= (ph + 6)
            if y_position < 80:
                c.showPage()
                y_position = height - 60

    return y_position - 10


def _draw_header(c, ticker, final_signal, signal_color, score, adj_score,
                 sent_score, confidence, width, height):
    """Draw the branded page header. Must be called after signal_color is defined."""
    header_h = 100

    # Background
    c.setFillColor(_hex(C_NAVY))
    c.rect(0, height - header_h, width, header_h, fill=True, stroke=False)

    # ── Icon box ─────────────────────────────────────────────────────────────
    ix, iy, isz = 14, height - header_h + 18, 64
    c.setFillColor(_hex(C_BLUE))
    c.roundRect(ix, iy, isz, isz, 10, fill=True, stroke=False)

    # Ascending trend line
    c.setStrokeColor(colors.white)
    c.setLineWidth(2.5)
    pts = [
        (ix + 10, iy + 14),
        (ix + 22, iy + 26),
        (ix + 36, iy + 20),
        (ix + 50, iy + 48),
    ]
    for i in range(len(pts) - 1):
        c.line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
    c.setFillColor(colors.white)
    c.circle(pts[0][0],  pts[0][1],  3.5, fill=True, stroke=False)
    c.circle(pts[-1][0], pts[-1][1], 5.5, fill=True, stroke=False)
    c.setFillColor(_hex("#1D4ED8"))
    c.circle(pts[-1][0], pts[-1][1], 2.0, fill=True, stroke=False)

    # ── Wordmark ─────────────────────────────────────────────────────────────
    wx = ix + isz + 12
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(wx,      height - header_h + 66, "Market")
    c.setFillColor(_hex(C_BLUE_LIGHT))
    c.drawString(wx + 74, height - header_h + 66, "Lab")
    c.setFillColor(_hex("#475569"))
    c.setFont("Helvetica", 6.5)
    c.drawString(wx, height - header_h + 46, "QUANTITATIVE  RESEARCH  FRAMEWORK")

    # ── Vertical divider ─────────────────────────────────────────────────────
    dvx = 258
    c.setStrokeColor(_hex("#334155"))
    c.setLineWidth(1)
    c.line(dvx, height - header_h + 16, dvx, height - header_h + 84)

    # ── Signal badge — RIGHT side, vertically centred ────────────────────────
    bw = 112
    bh = 30
    bx = width - bw - 14
    by = height - header_h + (header_h - bh) / 2
    c.setFillColor(signal_color)
    c.roundRect(bx, by, bw, bh, 6, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(bx + bw / 2, by + bh / 2 - 4, final_signal)

    # ── Title block — LEFT of badge, hard stop at bx - 12 ───────────────────
    tx = dvx + 16
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(tx, height - header_h + 70, "EXECUTIVE SUMMARY REPORT")

    c.setFont("Helvetica", 9.5)
    c.setFillColor(_hex(C_SUBTLE))
    c.drawString(tx, height - header_h + 52,
                 f"{ticker}  |  {date.today().strftime('%B %d, %Y')}")

    score_str = f"Score: {score} -> {adj_score}" if sent_score != 0 else f"Score: {score}"
    c.setFont("Helvetica", 8)
    c.setFillColor(_hex(C_MUTED))
    c.drawString(tx, height - header_h + 34,
                 f"{score_str}  |  Confidence: {confidence}")

    # ── Accent stripe ─────────────────────────────────────────────────────────
    c.setFillColor(_hex(C_ACCENT))
    c.rect(0, height - header_h, width, 3, fill=True, stroke=False)


def generate_pdf_report(all_data, stock_info, all_metrics, tickers,
                        all_sentiment=None, seasonality_charts=None):
    from signals import generate_signal

    if all_sentiment     is None: all_sentiment     = {}
    if seasonality_charts is None: seasonality_charts = {}

    filename = f"Financial_Analysis_Report_{date.today()}.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    styles = getSampleStyleSheet()

    insight_style = ParagraphStyle(
        'Insight', parent=styles['Normal'],
        fontSize=10, leading=15,
        textColor=_hex(C_MID), spaceAfter=10
    )
    reason_style = ParagraphStyle(
        'Reason', parent=styles['Normal'],
        fontSize=10, leading=14,
        textColor=_hex(C_MID), leftIndent=18
    )

    for ticker in tickers:
        df = all_data.get(ticker)
        if df is None:
            continue

        info          = stock_info[ticker]
        metrics       = all_metrics[ticker]
        sentiment     = all_sentiment.get(ticker, {})
        signal_result = generate_signal(df, info, metrics)

        signal     = signal_result.get('signal',           'UNKNOWN')
        score      = signal_result.get('score',            'N/A')
        confidence = signal_result.get('confidence_level', 'N/A')
        reasons    = signal_result.get('reasons',          [])

        # ── Sentiment-adjusted score ──────────────────────────────────────
        sent_score   = sentiment.get('score', 0)
        adj_score    = (score + sent_score) if isinstance(score, int) else score
        final_signal = signal
        if isinstance(adj_score, int):
            from signals import BUY_THRESHOLD, SELL_THRESHOLD
            is_bullish = df['Close'].iloc[-1] > df['MA200'].iloc[-1]
            if adj_score >= BUY_THRESHOLD:
                final_signal = "STRONG BUY"  if (adj_score >= 10 and is_bullish) else "BUY"
            elif adj_score <= SELL_THRESHOLD:
                final_signal = "STRONG SELL" if (adj_score <= -10 and not is_bullish) else "SELL"
            else:
                final_signal = "HOLD"

        # ── Signal color (defined BEFORE header) ─────────────────────────
        if "BUY" in final_signal:
            signal_color = _hex(C_GREEN)
        elif "SELL" in final_signal:
            signal_color = _hex(C_RED)
        else:
            signal_color = _hex(C_AMBER)

        # ── Header ───────────────────────────────────────────────────────
        _draw_header(c, ticker, final_signal, signal_color,
                     score, adj_score, sent_score, confidence, width, height)

        # ── Info table ───────────────────────────────────────────────────
        info_data = [
            ['Market Data',       'Value',
             'Technical Indicators', 'Value'],
            ['Current Price',     f"${info.get('currentPrice', 'N/A')}",
             'RSI (14)',          f"{df['RSI'].iloc[-1]:.2f}"],
            ['52-Week High',      f"${info.get('fiftyTwoWeekHigh', 'N/A')}",
             'MACD',              f"{df['MACD'].iloc[-1]:.4f}"],
            ['52-Week Low',       f"${info.get('fiftyTwoWeekLow', 'N/A')}",
             'MA 50',             f"${df['MA50'].iloc[-1]:.2f}"],
            ['Beta',              f"{metrics.get('Beta', 'N/A')}",
             'MA 200',            f"${df['MA200'].iloc[-1]:.2f}"],
            ['Sharpe Ratio',      f"{metrics['Sharpe Annualized']:.4f}",
             'Stoch %K',          f"{df['%K'].iloc[-1]:.2f}"],
            ['Ann. Return',       f"{metrics['Annualized Return']*100:.2f}%",
             'Stoch %D',          f"{df['%D'].iloc[-1]:.2f}"],
        ]

        table = Table(info_data, colWidths=[130, 100, 150, 90])
        table.setStyle(TableStyle([
            ('BACKGROUND',   (0, 0), (-1,  0), _hex(C_DARK)),
            ('TEXTCOLOR',    (0, 0), (-1,  0), colors.white),
            ('FONTNAME',     (0, 0), (-1,  0), 'Helvetica-Bold'),
            ('FONTSIZE',     (0, 0), (-1, -1), 9.5),
            ('BACKGROUND',   (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS',(0,1), (-1, -1), [colors.white, _hex(C_BG_ALT)]),
            ('GRID',         (0, 0), (-1, -1), 0.4, _hex(C_BORDER)),
            ('PADDING',      (0, 0), (-1, -1), 7),
            ('FONTNAME',     (0, 1), (0,  -1), 'Helvetica-Bold'),
            ('FONTNAME',     (2, 1), (2,  -1), 'Helvetica-Bold'),
            ('TEXTCOLOR',    (0, 1), (-1, -1), _hex(C_MID)),
            ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        table.wrapOn(c, width, height)
        tbl_h      = table._height
        tbl_x      = (width - 470) / 2
        tbl_y_top  = height - 112
        table.drawOn(c, tbl_x, tbl_y_top - tbl_h)
        y_position = tbl_y_top - tbl_h - 28

        # ── Algorithm triggers ───────────────────────────────────────────
        y_position = _section_header(c, "Algorithm Triggers", y_position, width)

        if signal in ('ERROR', 'WAIT'):
            c.setFillColor(colors.red if signal == 'ERROR' else colors.orange)
            c.setFont("Helvetica-Bold", 10)
            c.drawString(40, y_position, f"Status: {signal}")
            y_position -= 20

        for reason in reasons:
            p = Paragraph(f"• {reason}", reason_style)
            _, rh = p.wrap(width - 80, height)
            if y_position - rh < 80:
                _page_footer(c, width, ticker)
                c.showPage()
                y_position = height - 50
            p.drawOn(c, 40, y_position - rh)
            y_position -= (rh + 7)

        y_position -= 16

        # ── Sentiment ────────────────────────────────────────────────────
        if sentiment:
            if y_position < 220:
                _page_footer(c, width, ticker)
                c.showPage()
                y_position = height - 50
            y_position = _draw_sentiment_section(
                c, ticker, sentiment, y_position, width, height, styles)
            y_position -= 16

        # ── Qualitative insights ──────────────────────────────────────────
        if y_position < 160:
            _page_footer(c, width, ticker)
            c.showPage()
            y_position = height - 50

        y_position = _section_header(
            c, "Financial & Technical Interpretation", y_position, width)

        for insight in get_qualitative_insights(df, metrics):
            p = Paragraph(insight, insight_style)
            _, ih = p.wrap(width - 80, height)
            if y_position - ih < 80:
                _page_footer(c, width, ticker)
                c.showPage()
                y_position = height - 50
            p.drawOn(c, 40, y_position - ih)
            y_position -= (ih + 4)

        _page_footer(c, width, ticker)

        # ── Technical charts ─────────────────────────────────────────────
        if signal not in ('ERROR', 'WAIT'):
            charts = save_charts(ticker, df)
            first  = True
            for chart_path in charts:
                chart_h = 215
                if y_position - chart_h < 70:
                    c.showPage()
                    y_position = height - 40

                if first:
                    first = False
                    y_position -= 18
                    y_position = _section_header(
                        c, "Technical Charts", y_position, width)
                    y_position += 6

                c.drawImage(chart_path, 20, y_position - chart_h,
                            width=555, height=chart_h, preserveAspectRatio=True)
                y_position -= (chart_h + 12)

            c.setFont("Helvetica-Oblique", 7.5)
            c.setFillColor(_hex(C_SUBTLE))
            c.drawString(40, max(y_position - 8, 20),
                         "Charts generated using matplotlib and standard technical analysis formulas.")

        c.showPage()

    c.save()
    print(f"\n[+] Report saved: {filename}")

    # Cleanup temp chart images
    for ticker in tickers:
        for ext in ['ma', 'bb', 'rsi', 'macd', 'stoch']:
            p = f'{ticker}_{ext}.png'
            if os.path.exists(p):
                os.remove(p)
