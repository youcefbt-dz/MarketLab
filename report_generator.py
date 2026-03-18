from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from datetime import date
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os



def save_charts(ticker, df):
    charts = []

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['Close'], label='Close', color='black')
    ax.plot(df['MA50'], label='MA50', linestyle='--')
    ax.plot(df['MA200'], label='MA200', linestyle='--')
    ax.plot(df['EMA20'], label='EMA20')
    ax.set_title(f'{ticker} - Price & Moving Averages')
    ax.legend()
    path = f'{ticker}_ma.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    charts.append(path)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['Close'], color='black', label='Close')
    ax.plot(df['BB_upper'], color='red', alpha=0.5, label='Upper')
    ax.plot(df['BB_lower'], color='green', alpha=0.5, label='Lower')
    ax.fill_between(df.index, df['BB_lower'], df['BB_upper'], alpha=0.1, color='gray')
    ax.set_title(f'{ticker} - Bollinger Bands')
    ax.legend()
    path = f'{ticker}_bb.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    charts.append(path)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['RSI'], color='purple', label='RSI')
    ax.axhline(y=70, color='red', linestyle='--', label='Overbought')
    ax.axhline(y=30, color='green', linestyle='--', label='Oversold')
    ax.fill_between(df.index, 70, df['RSI'], where=(df['RSI'] >= 70), color='red', alpha=0.3)
    ax.fill_between(df.index, 30, df['RSI'], where=(df['RSI'] <= 30), color='green', alpha=0.3)
    ax.set_title(f'{ticker} - RSI')
    ax.legend()
    path = f'{ticker}_rsi.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    charts.append(path)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['MACD'], label='MACD', color='blue')
    ax.plot(df['Signal'], label='Signal', color='orange')
    colors_hist = ['green' if v >= 0 else 'red' for v in df['Histogram']]
    ax.bar(df.index, df['Histogram'], color=colors_hist, alpha=0.5, label='Histogram')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_title(f'{ticker} - MACD')
    ax.legend()
    path = f'{ticker}_macd.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    charts.append(path)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['%K'], label='%K', color='blue')
    ax.plot(df['%D'], label='%D', color='orange')
    ax.axhline(y=80, color='red', linestyle='--', label='Overbought')
    ax.axhline(y=20, color='green', linestyle='--', label='Oversold')
    ax.set_title(f'{ticker} - Stochastic Oscillator')
    ax.legend()
    path = f'{ticker}_stoch.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    charts.append(path)

    return charts
def generate_pdf_report(all_data, stock_info, all_metrics, tickers):
    from signals import generate_signal

    filename = f"Stock_Report_{date.today()}.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4  # 595 x 842

    for ticker in tickers:
        df = all_data.get(ticker)
        if df is None:
            continue

        info = stock_info[ticker]
        metrics = all_metrics[ticker]
        signal_result = generate_signal(df, info, metrics)


        c.setFillColor(colors.HexColor("#031234"))
        c.rect(0, height - 80, width, 80, fill=True, stroke=False)


        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 22)
        c.drawString(40, height - 45, "STOCK ANALYSIS REPORT")
        c.setFont("Helvetica", 12)
        c.setFillColor(colors.HexColor('#9CA3AF'))
        c.drawString(40, height - 65, f"{ticker}  |  {date.today().strftime('%B %d, %Y')}")

        signal = signal_result['signal']
        score = signal_result['score']
        signal_color = colors.green if 'BUY' in signal else colors.red if 'SELL' in signal else colors.orange
        c.setFillColor(signal_color)
        c.setFont("Helvetica-Bold", 16)
        c.drawRightString(width - 40, height - 50, signal)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 11)
        c.drawRightString(width - 40, height - 68, f"Score: {score}/{signal_result['max_score']}")

        c.setFillColor(colors.black)
        info_data = [
            ['Metric', 'Value', 'Indicator', 'Value'],
            ['Price', f"${info.get('currentPrice', 'N/A')}",
             'RSI', f"{df['RSI'].iloc[-1]:.2f}"],
            ['52W High', f"${info.get('fiftyTwoWeekHigh', 'N/A')}",
             'MACD', f"{df['MACD'].iloc[-1]:.4f}"],
            ['52W Low', f"${info.get('fiftyTwoWeekLow', 'N/A')}",
             'MA50', f"{df['MA50'].iloc[-1]:.2f}"],
            ['Beta', f"{metrics.get('Beta', 'N/A')}",
             'MA200', f"{df['MA200'].iloc[-1]:.2f}"],
            ['Sharpe', f"{metrics['Sharpe Annualized']:.4f}",
             '%K', f"{df['%K'].iloc[-1]:.2f}"],
            ['Ann. Return', f"{metrics['Annualized Return']*100:.2f}%",
             '%D', f"{df['%D'].iloc[-1]:.2f}"],
        ]

        table = Table(info_data, colWidths=[100, 100, 100, 100])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#B5CAEC")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ('ROWBACKGROUND', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))

        table.wrapOn(c, width, height)
        table.drawOn(c, 40, height - 260)

        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(colors.black)
        c.drawString(40, height - 285, "Signal Analysis:")
        c.setFont("Helvetica", 10)
        y = height - 305
        for reason in signal_result.get('reasons', []):
            c.drawString(50, y, f"• {reason}")
            y -= 18
            if y < 50:
                c.showPage()
                y = height - 50
                
        charts = save_charts(ticker, df)
        for i in range(0, len(charts), 2):
            c.showPage()
            c.setFillColor(colors.HexColor('#111827'))
            c.rect(0, height - 40, width, 40, fill=True, stroke=False)
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, height - 25, f"Technical Charts - {ticker}")

            c.drawImage(charts[i], 20, height/2 + 30, width=555, height=320)
            if i + 1 < len(charts):
                c.drawImage(charts[i+1], 20, 50, width=555, height=320)

        c.showPage()

    c.save()
    print(f"✅ Report saved: {filename}")
    for ticker in tickers:
        for ext in ['ma', 'bb', 'rsi', 'macd', 'stoch']:
            path = f'{ticker}_{ext}.png'
            if os.path.exists(path):
                os.remove(path)
