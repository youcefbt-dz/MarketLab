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


def save_charts(ticker, df):
    charts = []
    plt.style.use('seaborn-v0_8-darkgrid')

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['Close'], label='Close Price', color='#111827', linewidth=2)
    ax.plot(df['MA50'], label='MA50 (Medium Term)', linestyle='--', color='#3B82F6', alpha=0.8)
    ax.plot(df['MA200'], label='MA200 (Long Term)', linestyle='--', color='#EF4444', alpha=0.8)
    ax.plot(df['EMA20'], label='EMA20 (Short Term)', color='#10B981', alpha=0.8)
    ax.set_title(f'{ticker} - Price Action & Moving Averages', fontsize=14, fontweight='bold', color='#1F2937')
    ax.legend(loc='upper left')
    path = f'{ticker}_ma.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    charts.append(path)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['Close'], color='#111827', label='Close Price', linewidth=1.5)
    ax.plot(df['BB_upper'], color='#EF4444', alpha=0.6, label='Upper Band (Resistance)')
    ax.plot(df['BB_lower'], color='#10B981', alpha=0.6, label='Lower Band (Support)')
    ax.fill_between(df.index, df['BB_lower'], df['BB_upper'], alpha=0.1, color='#6B7280')
    ax.set_title(f'{ticker} - Volatility (Bollinger Bands)', fontsize=14, fontweight='bold', color='#1F2937')
    ax.legend(loc='upper left')
    path = f'{ticker}_bb.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    charts.append(path)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['RSI'], color='#8B5CF6', label='RSI (14)', linewidth=2)
    ax.axhline(y=70, color='#EF4444', linestyle='--', label='Overbought (>70)')
    ax.axhline(y=30, color='#10B981', linestyle='--', label='Oversold (<30)')
    ax.fill_between(df.index, 70, df['RSI'], where=(df['RSI'] >= 70), color='#EF4444', alpha=0.2)
    ax.fill_between(df.index, 30, df['RSI'], where=(df['RSI'] <= 30), color='#10B981', alpha=0.2)
    ax.set_title(f'{ticker} - Relative Strength Index (Momentum)', fontsize=14, fontweight='bold', color='#1F2937')
    ax.legend(loc='upper left')
    path = f'{ticker}_rsi.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    charts.append(path)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['MACD'], label='MACD Line', color='#2563EB', linewidth=1.5)
    ax.plot(df['Signal'], label='Signal Line', color='#F59E0B', linewidth=1.5)
    colors_hist = ['#10B981' if v >= 0 else '#EF4444' for v in df['Histogram']]
    ax.bar(df.index, df['Histogram'], color=colors_hist, alpha=0.6, label='MACD Histogram')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_title(f'{ticker} - MACD (Trend & Momentum)', fontsize=14, fontweight='bold', color='#1F2937')
    ax.legend(loc='upper left')
    path = f'{ticker}_macd.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    charts.append(path)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['%K'], label='%K (Fast)', color='#3B82F6', linewidth=1.5)
    ax.plot(df['%D'], label='%D (Slow)', color='#F59E0B', linewidth=1.5)
    ax.axhline(y=80, color='#EF4444', linestyle='--', label='Overbought (>80)')
    ax.axhline(y=20, color='#10B981', linestyle='--', label='Oversold (<20)')
    ax.set_title(f'{ticker} - Stochastic Oscillator', fontsize=14, fontweight='bold', color='#1F2937')
    ax.legend(loc='upper left')
    path = f'{ticker}_stoch.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    charts.append(path)

    return charts


def get_qualitative_insights(df, metrics):
    insights = []

    rsi = df['RSI'].iloc[-1]
    if rsi >= 70:
        insights.append(f"<b>RSI ({rsi:.2f}):</b> The Relative Strength Index indicates an <b>Overbought</b> condition. The asset may be slightly overvalued in the short term, suggesting a potential price correction or consolidation phase soon.")
    elif rsi <= 30:
        insights.append(f"<b>RSI ({rsi:.2f}):</b> The indicator flashes an <b>Oversold</b> signal. The stock has faced significant selling pressure and might be undervalued, potentially setting up a rebound opportunity.")
    else:
        insights.append(f"<b>RSI ({rsi:.2f}):</b> The momentum is currently <b>Neutral</b>. The stock is neither heavily overbought nor oversold, indicating balanced market participation.")

    macd = df['MACD'].iloc[-1]
    hist = df['Histogram'].iloc[-1]
    if macd > 0 and hist > 0:
        insights.append("<b>MACD:</b> Positive and expanding. Buyers are in control, showing strong and accelerating <b>Bullish</b> momentum.")
    elif macd < 0 and hist < 0:
        insights.append("<b>MACD:</b> Negative and falling. Sellers are dominating, reflecting strong <b>Bearish</b> pressure.")
    elif hist > 0:
        insights.append("<b>MACD Histogram:</b> Turning positive, indicating early signs of a bullish crossover and shifting momentum in favor of buyers.")

    beta = metrics.get('Beta', 1.0)
    if beta > 1.2:
        insights.append(f"<b>Beta ({beta:.2f}):</b> High systematic risk. This stock is historically <b>more volatile than the broader market</b>. Expect larger price swings (both up and down).")
    elif beta < 0.8:
        insights.append(f"<b>Beta ({beta:.2f}):</b> Low systematic risk. This stock acts as a <b>defensive asset</b>, being less volatile and less sensitive to broader market movements.")

    sharpe = metrics.get('Sharpe Annualized', 0)
    if sharpe > 1:
        insights.append(f"<b>Sharpe Ratio ({sharpe:.2f}):</b> Excellent historical performance relative to the risk taken. The asset provides <b>superior risk-adjusted returns</b>.")
    elif sharpe < 0:
        insights.append(f"<b>Sharpe Ratio ({sharpe:.2f}):</b> Negative risk-adjusted returns. Historically, the risk taken has not been adequately compensated compared to a risk-free asset.")

    close = df['Close'].iloc[-1]
    ma200 = df['MA200'].iloc[-1]
    if close > ma200:
        insights.append(f"<b>Long-term Trend:</b> The current price is <b>above the 200-day Moving Average</b> (${ma200:.2f}), confirming a solid long-term macro uptrend.")
    else:
        insights.append(f"<b>Long-term Trend:</b> The current price is <b>below the 200-day Moving Average</b> (${ma200:.2f}), indicating structural long-term weakness.")

    return insights


def _draw_sentiment_section(c, ticker, sentiment, y_position, width, height, styles):
    """Draw the News Sentiment section in the PDF."""

    reason_style = ParagraphStyle(
        'Reason',
        parent=styles['Normal'],
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1F2937"),
        leftIndent=20
    )

    label        = sentiment.get('label', 'NEUTRAL')
    compound     = sentiment.get('compound', 0.0)
    confidence   = sentiment.get('confidence', 0.0)
    news_count   = sentiment.get('news_count', 0)
    pos_ratio    = sentiment.get('positive_ratio', 0.0)
    tail_risk    = sentiment.get('tail_risk', False)
    headlines    = sentiment.get('headlines', [])

    label_colors_map = {
        'BULLISH'         : '#10B981',
        'SLIGHTLY BULLISH': '#84CC16',
        'NEUTRAL'         : '#94A3B8',
        'SLIGHTLY BEARISH': '#F59E0B',
        'BEARISH'         : '#EF4444',
    }
    badge_color = colors.HexColor(label_colors_map.get(label, '#94A3B8'))

    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.HexColor("#0F172A"))
    c.drawString(40, y_position, "News Sentiment Analysis:")
    c.setStrokeColor(colors.HexColor("#E2E8F0"))
    c.setLineWidth(1)
    c.line(40, y_position - 5, width - 40, y_position - 5)
    y_position -= 30

    badge_h     = 22
    badge_y     = y_position - badge_h + 6
    text_center = badge_y + badge_h / 2 - 4

    c.setFillColor(badge_color)
    c.roundRect(40, badge_y, 130, badge_h, 4, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(105, text_center, label)

    c.setFillColor(colors.HexColor("#374151"))
    c.setFont("Helvetica", 10)
    c.drawString(185, text_center,
                 f"Compound: {compound:.4f}   |   Confidence: {confidence*100:.1f}%   |   "
                 f"News Analyzed: {news_count}")
    y_position -= 30

    c.setFillColor(colors.HexColor("#374151"))
    c.setFont("Helvetica", 10)
    c.drawString(40, y_position,
                 f"Positive: {pos_ratio*100:.0f}%   |   "
                 f"Negative: {(1 - pos_ratio)*100:.0f}%")
    y_position -= 25

    if tail_risk:
        y_position -= 8
        box_h  = 24
        box_y  = y_position - box_h
        text_y = box_y + (box_h - 10) / 2
        c.setFillColor(colors.HexColor("#FEF3C7"))
        c.roundRect(38, box_y, width - 76, box_h, 3, fill=True, stroke=False)
        c.setFillColor(colors.HexColor("#92400E"))
        c.setFont("Helvetica-Bold", 10)
        c.drawString(45, text_y,
                     "TAIL RISK DETECTED: Strong negative news found — compound score adjusted.")
        y_position -= (box_h + 12)

    if headlines:
        y_position -= 5
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(colors.HexColor("#0F172A"))
        c.drawString(40, y_position, "Top Headlines:")
        y_position -= 16

        sentiment_icon = {'Positive': '[+]', 'Negative': '[-]', 'Neutral': '[~]'}
        for h in headlines[:5]:
            icon  = sentiment_icon.get(h['sentiment'], '[~]')
            text  = f"{icon} {h['title'][:85]}  (score: {h['compound']:.3f})"
            p = Paragraph(text, reason_style)
            w, ph = p.wrap(width - 80, height)
            p.drawOn(c, 40, y_position - ph)
            y_position -= (ph + 6)
            if y_position < 80:
                c.showPage()
                y_position = height - 60

    y_position -= 10
    return y_position


def generate_pdf_report(all_data, stock_info, all_metrics, tickers, all_sentiment=None):
    from signals import generate_signal

    if all_sentiment is None:
        all_sentiment = {}

    filename = f"Financial_Analysis_Report_{date.today()}.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    styles = getSampleStyleSheet()

    insight_style = ParagraphStyle(
        'Insight',
        parent=styles['Normal'],
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor("#374151"),
        spaceAfter=12
    )

    reason_style = ParagraphStyle(
        'Reason',
        parent=styles['Normal'],
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1F2937"),
        bulletIndent=10,
        leftIndent=20
    )

    for ticker in tickers:
        df = all_data.get(ticker)
        if df is None:
            continue

        info          = stock_info[ticker]
        metrics       = all_metrics[ticker]
        sentiment     = all_sentiment.get(ticker, {})
        signal_result = generate_signal(df, info, metrics)

        signal     = signal_result.get('signal', 'UNKNOWN')
        score      = signal_result.get('score', 'N/A')
        confidence = signal_result.get('confidence_level', 'N/A')
        reasons    = signal_result.get('reasons', [])

        # Adjust score with sentiment
        sent_score    = sentiment.get('score', 0)
        adj_score     = (score + sent_score) if isinstance(score, int) else score
        final_signal  = signal
        if isinstance(adj_score, int):
            from signals import BUY_THRESHOLD, SELL_THRESHOLD
            is_bullish = df['Close'].iloc[-1] > df['MA200'].iloc[-1]
            if adj_score >= BUY_THRESHOLD:
                final_signal = "STRONG BUY" if (adj_score >= 10 and is_bullish) else "BUY"
            elif adj_score <= SELL_THRESHOLD:
                final_signal = "STRONG SELL" if (adj_score <= -10 and not is_bullish) else "SELL"
            else:
                final_signal = "HOLD"

        # --- Header ---
        c.setFillColor(colors.HexColor("#0F172A"))
        c.rect(0, height - 90, width, 90, fill=True, stroke=False)

        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 24)
        c.drawString(40, height - 45, "EXECUTIVE SUMMARY REPORT")
        c.setFont("Helvetica", 12)
        c.setFillColor(colors.HexColor('#94A3B8'))
        c.drawString(40, height - 70, f"Alphabet Inc. {ticker}  |  Generated: {date.today().strftime('%B %d, %Y')}")

        signal_color = (colors.HexColor("#10B981") if 'BUY'  in final_signal else
                        colors.HexColor("#EF4444") if 'SELL' in final_signal else
                        colors.HexColor("#F59E0B"))

        c.setFillColor(signal_color)
        c.roundRect(width - 140, height - 55, 100, 30, 5, fill=True, stroke=False)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(width - 90, height - 45, final_signal)

        c.setFillColor(colors.white)
        c.setFont("Helvetica", 10)
        score_str = f"Score: {score} → {adj_score}" if sent_score != 0 else f"Score: {score}"
        c.drawRightString(width - 40, height - 75, f"{score_str}  |  Confidence: {confidence}")

        # --- Info Table ---
        c.setFillColor(colors.black)
        info_data = [
            ['Market Data', 'Value', 'Technical Indicators', 'Value'],
            ['Current Price',     f"${info.get('currentPrice', 'N/A')}",  'RSI (14)',  f"{df['RSI'].iloc[-1]:.2f}"],
            ['52-Week High',      f"${info.get('fiftyTwoWeekHigh', 'N/A')}", 'MACD',   f"{df['MACD'].iloc[-1]:.4f}"],
            ['52-Week Low',       f"${info.get('fiftyTwoWeekLow', 'N/A')}",  'MA 50',  f"${df['MA50'].iloc[-1]:.2f}"],
            ['Beta (Volatility)', f"{metrics.get('Beta', 'N/A')}",         'MA 200',   f"${df['MA200'].iloc[-1]:.2f}"],
            ['Sharpe Ratio',      f"{metrics['Sharpe Annualized']:.4f}",   'Stoch %K', f"{df['%K'].iloc[-1]:.2f}"],
            ['Ann. Return',       f"{metrics['Annualized Return']*100:.2f}%", 'Stoch %D', f"{df['%D'].iloc[-1]:.2f}"],
        ]

        table = Table(info_data, colWidths=[120, 100, 140, 100])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
            ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUND', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
            ('PADDING',    (0, 0), (-1, -1), 8),
            ('FONTNAME',   (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME',   (2, 1), (2, -1), 'Helvetica-Bold'),
            ('TEXTCOLOR',  (0, 1), (-1, -1), colors.HexColor("#334155")),
        ]))

        table.wrapOn(c, width, height)
        table_height = table._height
        y_position = height - 120 - table_height
        table.drawOn(c, (width - 460) / 2, y_position)

        y_position -= 30

        # --- Algorithm Signal Reasons ---
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(colors.HexColor("#0F172A"))
        c.drawString(40, y_position, "Algorithm Triggers:")
        c.setStrokeColor(colors.HexColor("#E2E8F0"))
        c.setLineWidth(1)
        c.line(40, y_position - 5, width - 40, y_position - 5)
        y_position -= 25

        if signal in ('ERROR', 'WAIT'):
            c.setFillColor(colors.red if signal == 'ERROR' else colors.orange)
            c.drawString(40, y_position, f"Status Notice: {signal}")
            y_position -= 20
            c.setFillColor(colors.black)

        for reason in reasons:
            p = Paragraph(f"• {reason}", reason_style)
            w, h = p.wrap(width - 80, height)
            p.drawOn(c, 40, y_position - h)
            y_position -= (h + 8)
            if y_position < 100:
                c.showPage()
                y_position = height - 50

        y_position -= 20

        # --- Sentiment Section ---
        if sentiment:
            if y_position < 200:
                c.showPage()
                y_position = height - 60
            y_position = _draw_sentiment_section(
                c, ticker, sentiment, y_position, width, height, styles
            )
            y_position -= 20

        # --- Qualitative Insights ---
        if y_position < 150:
            c.showPage()
            y_position = height - 60

        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(colors.HexColor("#0F172A"))
        c.drawString(40, y_position, "Financial & Technical Interpretation:")
        c.setStrokeColor(colors.HexColor("#E2E8F0"))
        c.line(40, y_position - 5, width - 40, y_position - 5)
        y_position -= 25

        insights = get_qualitative_insights(df, metrics)
        for insight in insights:
            p = Paragraph(insight, insight_style)
            w, h = p.wrap(width - 80, height)
            p.drawOn(c, 40, y_position - h)
            y_position -= (h + 5)
            if y_position < 60:
                c.showPage()
                y_position = height - 60

        c.setFont("Helvetica-Oblique", 8)
        c.setFillColor(colors.HexColor("#94A3B8"))
        c.drawString(40, 30, "Disclaimer: This report is generated algorithmically and does not constitute financial advice.")

        # --- Charts ---
        if signal not in ('ERROR', 'WAIT'):
            charts = save_charts(ticker, df)
            for i, chart_path in enumerate(charts):
                chart_h = 220
                if y_position - chart_h < 60:
                    c.showPage()
                    y_position = height - 40

                if i == 0:
                    y_position -= 20
                    c.setFillColor(colors.HexColor("#0F172A"))
                    c.setFont("Helvetica-Bold", 13)
                    c.drawString(40, y_position, "Technical Charts:")
                    c.setStrokeColor(colors.HexColor("#E2E8F0"))
                    c.setLineWidth(1)
                    c.line(40, y_position - 5, width - 40, y_position - 5)
                    y_position -= 15

                c.drawImage(chart_path, 20, y_position - chart_h,
                            width=555, height=chart_h, preserveAspectRatio=True)
                y_position -= (chart_h + 10)

            c.setFont("Helvetica-Oblique", 8)
            c.setFillColor(colors.HexColor("#94A3B8"))
            c.drawString(40, max(y_position - 10, 20),
                         "Charts generated utilizing matplotlib and standard technical formulas.")

        c.showPage()

    c.save()
    print(f"\n[+] Professional Report successfully saved as: {filename}")

    for ticker in tickers:
        for ext in ['ma', 'bb', 'rsi', 'macd', 'stoch']:
            path = f'{ticker}_{ext}.png'
            if os.path.exists(path):
                os.remove(path)
