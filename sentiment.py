import numpy as np
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timezone

analyzer = SentimentIntensityAnalyzer()

# ── Financial keyword boos
BULLISH_KEYWORDS = [
    'beats', 'beat', 'record', 'surges', 'jumps', 'raises guidance',
    'upgrade', 'outperform', 'buy rating', 'strong earnings', 'profit up',
    'revenue growth', 'exceeds', 'topped estimates'
]
BEARISH_KEYWORDS = [
    'misses', 'missed', 'cuts guidance', 'downgrade', 'underperform',
    'sell rating', 'loss', 'layoffs', 'recall', 'investigation',
    'below estimates', 'revenue decline', 'warning', 'lawsuit'
]

def _financial_boost(title: str) -> float:
    title_lower = title.lower()
    boost = 0.0
    for kw in BULLISH_KEYWORDS:
        if kw in title_lower:
            boost += 0.15
    for kw in BEARISH_KEYWORDS:
        if kw in title_lower:
            boost -= 0.15
    return max(-0.3, min(0.3, boost))


def _news_age_weight(publish_time) -> float:
    if not publish_time:
        return 0.75
    try:
        now = datetime.now(timezone.utc).timestamp()
        age_hours = (now - publish_time) / 3600
        if age_hours < 6:
            return 1.0
        elif age_hours < 24:
            return 0.85
        elif age_hours < 72:
            return 0.70
        else:
            return 0.55
    except Exception:
        return 0.75


def analyze_sentiment(ticker: str) -> dict:
    Returns:
        dict: label, score, compound, confidence, news_count,
              positive_ratio, tail_risk, headlines, error(optional)
    """
    try:
        stock = yf.Ticker(ticker)

        # ── Retry System ──────────────────────────────────────────────
        news = []
        for attempt in range(3):
            news = stock.news
            if news:
                break

        if not news:
            return _empty_result()

        headlines      = []
        weighted_comps = []
        raw_comps      = []

        for item in news[:10]:
            try:
                content  = item.get('content', {})
                title    = content.get('title', '') or item.get('title', '')
                pub_time = content.get('pubDate', None) or item.get('providerPublishTime', None)
                if isinstance(pub_time, str):
                    pub_time = None
            except Exception:
                title, pub_time = '', None

            if not title:
                continue

            # ── VADER + Financial Boost ───────────────────────────────
            vs       = analyzer.polarity_scores(title)
            compound = vs['compound'] + _financial_boost(title)
            compound = max(-1.0, min(1.0, compound))
            weight = _news_age_weight(pub_time)
            weighted_comps.append(compound * weight)
            raw_comps.append(compound)

            if compound >= 0.05:
                sentiment_label = 'Positive'
            elif compound <= -0.05:
                sentiment_label = 'Negative'
            else:
                sentiment_label = 'Neutral'

            headlines.append({
                'title'    : title,
                'compound' : round(compound, 4),
                'sentiment': sentiment_label,
                'weight'   : round(weight, 2)
            })
        if not weighted_comps and news:
            for item in news[:10]:
                try:
                    content = item.get('content', {})
                    title   = content.get('title', '') or item.get('title', '')
                    if not title:
                        continue
                    vs       = analyzer.polarity_scores(title)
                    compound = vs['compound'] + _financial_boost(title)
                    compound = max(-1.0, min(1.0, compound))
                    weighted_comps.append(compound)
                    raw_comps.append(compound)
                    if compound >= 0.05:
                        slabel = 'Positive'
                    elif compound <= -0.05:
                        slabel = 'Negative'
                    else:
                        slabel = 'Neutral'
                    headlines.append({
                        'title'    : title,
                        'compound' : round(compound, 4),
                        'sentiment': slabel,
                        'weight'   : 0.75 
                    })
                except Exception:
                    continue

        if not weighted_comps:
            return _empty_result()


        avg_compound = float(np.mean(weighted_comps))

        total          = len(raw_comps)
        positive_count = sum(1 for c in raw_comps if c >= 0.05)
        negative_count = sum(1 for c in raw_comps if c <= -0.05)
        positive_ratio = positive_count / total
        negative_ratio = negative_count / total

        # ── 3. Tail Risk ──────────────────────────────────────────────
        strong_negatives = [c for c in raw_comps if c <= -0.5]
        tail_risk        = len(strong_negatives) > 0

        if tail_risk:
            worst        = min(strong_negatives)
            avg_compound = (avg_compound * 0.6) + (worst * 0.4)

        if negative_ratio >= 0.7:
            avg_compound -= 0.1
        elif positive_ratio >= 0.7:
            avg_compound += 0.05

        avg_compound = max(-1.0, min(1.0, avg_compound))

        if avg_compound >= 0.15:
            label, score = 'BULLISH', 3
        elif avg_compound >= 0.05:
            label, score = 'SLIGHTLY BULLISH', 1
        elif avg_compound <= -0.15:
            label, score = 'BEARISH', -3
        elif avg_compound <= -0.05:
            label, score = 'SLIGHTLY BEARISH', -1
        else:
            label, score = 'NEUTRAL', 0

        # ── Confidence ────────────────────────────────────────────────
        confidence = round(min(abs(avg_compound) * 100, 100), 1)

        return {
            'label'         : label,
            'score'         : score,
            'compound'      : round(avg_compound, 4),
            'confidence'    : confidence,
            'news_count'    : len(headlines),
            'positive_ratio': round(positive_ratio, 2),
            'tail_risk'     : tail_risk,
            'headlines'     : headlines
        }

    except Exception as e:
        return {**_empty_result(), 'error': str(e)}


def _empty_result() -> dict:
    return {
        'label'         : 'NEUTRAL',
        'score'         : 0,
        'compound'      : 0.0,
        'confidence'    : 0.0,
        'news_count'    : 0,
        'positive_ratio': 0.0,
        'tail_risk'     : False,
        'headlines'     : []
    }


def print_sentiment(ticker: str, sentiment: dict):
    label_colors = {
        'BULLISH'         : '🟢',
        'SLIGHTLY BULLISH': '🟡',
        'NEUTRAL'         : '⚪',
        'SLIGHTLY BEARISH': '🟠',
        'BEARISH'         : '🔴'
    }

    if 'error' in sentiment:
        print(f"\n--- News Sentiment for {ticker} ---")
        print(f"⚠️  Error: {sentiment['error']}")
        return

    icon = label_colors.get(sentiment['label'], '⚪')

    print(f"\n--- News Sentiment for {ticker} ---")
    print(f"Overall   : {icon} {sentiment['label']}")
    print(f"Compound  : {sentiment['compound']:.4f}  |  "
          f"Confidence: {sentiment['confidence']}%  |  "
          f"News: {sentiment['news_count']}")
    print(f"Positive  : {sentiment['positive_ratio']*100:.0f}%  |  "
          f"Negative: {(1 - sentiment['positive_ratio'])*100:.0f}%")

    if sentiment.get('tail_risk'):
        print("  TAIL RISK: Strong negative news detected — result adjusted.")

    if sentiment['headlines']:
        print("\nTop Headlines:")
        for i, h in enumerate(sentiment['headlines'][:5], 1):
            icon_h = ('' if h['sentiment'] == 'Positive'
                      else '' if h['sentiment'] == 'Negative'
                      else '')
            print(f"  {i}. {icon_h} [{h['sentiment']:8s}] "
                  f"(w={h['weight']}) {h['title'][:75]}")
