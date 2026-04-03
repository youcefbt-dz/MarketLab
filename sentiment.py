import numpy as np
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timezone

analyzer = SentimentIntensityAnalyzer()

# ── Add custom words to VADER lexicon ────────────────────────────────────────
CUSTOM_VADER_WORDS = {
    'too soon'        : -0.5,
    'regret'          : -0.6,
    'mistake'         : -0.5,
    'should have'     : -0.4,
    'wish'            : -0.3,
    'opportunity'     : 0.3,
    'strong conviction': 0.5,
    'accumulating'    : 0.4,
}
for word, score in CUSTOM_VADER_WORDS.items():
    analyzer.lexicon[word] = score

# ── Influencers ───────────────────────────────────────────────────────────────
# FIX 3: weight and bias are now actually used in _influencer_boost
INFLUENCERS = {
    'warren buffett': {'weight': 1.5, 'bias': 'neutral'},
    'elon musk'     : {'weight': 1.3, 'bias': 'volatile'},
    'cathie wood'   : {'weight': 1.2, 'bias': 'bullish'},
    'bill ackman'   : {'weight': 1.2, 'bias': 'neutral'},
    'michael burry' : {'weight': 1.2, 'bias': 'bearish'},
    'ray dalio'     : {'weight': 1.2, 'bias': 'neutral'},
    'peter lynch'   : {'weight': 1.1, 'bias': 'bullish'},
    'charlie munger': {'weight': 1.3, 'bias': 'neutral'},
}

# ── Bias baseline adjustments ─────────────────────────────────────────────────
BIAS_BASELINE = {
    'bullish' :  0.05,
    'bearish' : -0.05,
    'volatile':  0.0,
    'neutral' :  0.0,
}

# ── Negative / Positive patterns ─────────────────────────────────────────────
NEGATIVE_PATTERNS = [
    ('sold',       'too soon'),
    ('should have','bought'),
    ('regret',     'selling'),
    ('mistake',    'sell'),
    ('wish i',     'held'),
    ('too early',  'exit'),
    ('left',       'money on table'),
    ('missed',     'opportunity'),
]
POSITIVE_PATTERNS = [
    ('bought',        'dip'),
    ('accumulating',  'shares'),
    ('adding',        'position'),
    ('conviction',    'buy'),
    ('loading up',    'on'),
    ('strong',        'buy'),
    ('undervalued',   'opportunity'),
]

# ── Source credibility weights ────────────────────────────────────────────────
SOURCE_WEIGHTS = {
    'reuters.com'      : 1.0,
    'bloomberg.com'    : 1.0,
    'ft.com'           : 1.0,
    'wsj.com'          : 1.0,
    'apnews.com'       : 1.0,
    'marketwatch.com'  : 0.85,
    'cnbc.com'         : 0.85,
    'finance.yahoo.com': 0.80,
    'barrons.com'      : 0.85,
    'morningstar.com'  : 0.85,
    'investopedia.com' : 0.75,
    'fool.com'         : 0.70,
    'seekingalpha.com' : 0.55,
    'benzinga.com'     : 0.60,
    'zacks.com'        : 0.65,
    '__default__'      : 0.50,
}

# ── Generic phrases → market-wide articles, always exclude ───────────────────
GENERIC_PHRASES = [
    'market update', 'stock picks', 'rule breaker', 'etf', 'index fund',
    's&p 500', 'portfolio', 'market madness', 'slam dunk', 'market cap game',
    'top stocks', 'best stocks', 'stocks to buy', 'stocks to watch',
    'wall street', 'dow jones', 'nasdaq composite', 'russell 2000',
]

# ── Financial keywords ────────────────────────────────────────────────────────
BULLISH_KEYWORDS = [
    'beats', 'beat', 'record', 'surges', 'jumps', 'raises guidance',
    'upgrade', 'outperform', 'buy rating', 'strong earnings', 'profit up',
    'revenue growth', 'exceeds', 'topped estimates', 'guidance raise',
    'record high', 'record revenue', 'record profit',
    'dividend increase', 'stock buyback', 'share buyback', 'repurchase',
    'partnership', 'acquisition', 'merger',
    'breakthrough', 'patent', 'regulatory approval', 'fda approval',
    'launched', 'new product', 'expansion',
    'rally', 'soars', 'climbs', 'strong demand', 'market share gain',
]
BEARISH_KEYWORDS = [
    'misses', 'missed', 'cuts guidance', 'downgrade', 'underperform',
    'sell rating', 'loss', 'below estimates', 'revenue decline', 'warning',
    'layoffs', 'recall', 'investigation', 'lawsuit', 'bankruptcy',
    'fraud', 'scandal', 'fine', 'penalty', 'ceo resigns', 'ceo fired',
    'cfo resigns', 'executive departure',
    'supply chain disruption', 'shortage', 'production halt',
    'plant closure', 'restructuring',
    'plunges', 'tumbles', 'crashes', 'selloff', 'slumps',
]


# ── Helpers ───────────────────────────────────────────────────────────────────

# FIX 1 & 2: Unified pub_time parser — handles int/float timestamps and
# ISO strings instead of discarding them.
def _parse_pub_time(pub_time) -> float | None:
    """Return a UTC timestamp float, or None if unparseable."""
    if pub_time is None:
        return None
    if isinstance(pub_time, (int, float)):
        return float(pub_time)
    if isinstance(pub_time, str):
        for fmt in ('%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%d %H:%M:%S'):
            try:
                dt = datetime.strptime(pub_time, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except ValueError:
                continue
    return None


def _news_age_weight(publish_time) -> float:
    parsed = _parse_pub_time(publish_time)
    if parsed is None:
        return 0.75
    try:
        age_h = (datetime.now(timezone.utc).timestamp() - parsed) / 3600
        if age_h < 6:  return 1.0
        if age_h < 24: return 0.85
        if age_h < 72: return 0.70
        return 0.55
    except Exception:
        return 0.75


def _match_patterns(title_lower: str, patterns: list) -> bool:
    """Return True if any (kw_a, kw_b) pair both appear in title."""
    return any(a in title_lower and b in title_lower for a, b in patterns)


# FIX 3: _influencer_boost now uses each influencer's weight and bias
def _influencer_boost(title: str) -> float:
    title_lower = title.lower()
    for name, info in INFLUENCERS.items():
        if name not in title_lower:
            continue

        iw   = info['weight']           # e.g. 1.5 for Buffett
        bias = BIAS_BASELINE[info['bias']]

        # FIX 5 (dedup): pattern check consolidated here, not repeated in _pattern_boost
        if _match_patterns(title_lower, NEGATIVE_PATTERNS):
            return round(-0.35 * iw + bias, 4)
        if _match_patterns(title_lower, POSITIVE_PATTERNS):
            return round( 0.35 * iw + bias, 4)

        if any(w in title_lower for w in ('sell', 'sold', 'exit')):
            return round(-0.25 * iw + bias, 4)
        if any(w in title_lower for w in ('buy', 'accumulate', 'add')):
            return round( 0.25 * iw + bias, 4)

        return round(0.1 * iw + bias, 4)   # neutral mention

    return 0.0


def _pattern_boost(title: str) -> float:
    """Pattern boost for headlines WITHOUT an influencer mention."""
    title_lower = title.lower()
    # Skip if an influencer is present — already handled by _influencer_boost
    if any(name in title_lower for name in INFLUENCERS):
        return 0.0
    if _match_patterns(title_lower, NEGATIVE_PATTERNS):
        return -0.3
    if _match_patterns(title_lower, POSITIVE_PATTERNS):
        return 0.3
    return 0.0


# FIX 4: Cap financial boost at ±0.30 (one strong keyword) before clamp
def _financial_boost(title: str) -> float:
    t = title.lower()
    bull = sum(0.15 for kw in BULLISH_KEYWORDS if kw in t)
    bear = sum(0.15 for kw in BEARISH_KEYWORDS if kw in t)
    raw  = min(bull, 0.30) - min(bear, 0.30)   # cap each side independently
    raw += _influencer_boost(title)
    raw += _pattern_boost(title)
    return max(-0.6, min(0.6, raw))


def _categorize_news_type(title: str) -> str:
    title_lower = title.lower()
    if any(w in title_lower for w in ['earnings', 'revenue', 'profit', 'loss', 'eps']):
        return 'earnings'
    if any(w in title_lower for w in ['analyst', 'upgrade', 'downgrade', 'price target', 'rating']):
        return 'analyst'
    if any(name in title_lower for name in INFLUENCERS):
        return 'influencer'
    if any(w in title_lower for w in ['ceo', 'cfo', 'executive', 'insider', 'director']):
        return 'insider'
    if any(w in title_lower for w in ['fda', 'approval', 'patent', 'regulatory']):
        return 'regulatory'
    return 'general'


NEWS_TYPE_WEIGHTS = {
    'earnings'  : 1.2,
    'analyst'   : 1.1,
    'influencer': 1.3,
    'insider'   : 1.2,
    'regulatory': 1.15,
    'general'   : 0.9,
}


def _get_company_keywords(ticker: str, info: dict) -> list:
    candidates = [ticker.lower()]
    for field in ('longName', 'shortName'):
        name = (info.get(field) or '').lower().strip()
        if name:
            candidates.append(name)
            first = name.split()[0]
            if len(first) > 3:
                candidates.append(first)
    seen, unique = set(), []
    for k in candidates:
        if k and k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


def _is_relevant(title: str, company_keywords: list) -> bool:
    text = title.lower()
    for kw in company_keywords:
        if kw in text:
            return True
    for phrase in GENERIC_PHRASES:
        if phrase in text:
            return False
    return False


def _source_credibility(url: str) -> float:
    if not url:
        return SOURCE_WEIGHTS['__default__']
    url_lower = url.lower()
    for domain, w in SOURCE_WEIGHTS.items():
        if domain != '__default__' and domain in url_lower:
            return w
    return SOURCE_WEIGHTS['__default__']


def _title_similarity(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def _deduplicate(items: list, threshold: float = 0.75) -> list:
    unique = []
    for c in items:
        if not any(_title_similarity(c['title'], e['title']) >= threshold for e in unique):
            unique.append(c)
    return unique


def _sentiment_label(compound: float) -> str:
    if compound >= 0.05:  return 'Positive'
    if compound <= -0.05: return 'Negative'
    return 'Neutral'


def _build_headline(title: str, compound: float, weight: float,
                    url: str, news_type: str) -> dict:
    return {
        'title'    : title,
        'compound' : round(compound, 4),
        'sentiment': _sentiment_label(compound),
        'weight'   : round(weight, 3),
        'url'      : url,
        'news_type': news_type,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def analyze_sentiment(ticker: str) -> dict:
    """
    Returns:
        dict: label, score, compound, confidence, news_count,
              positive_ratio, negative_ratio, tail_risk, headlines,
              excluded_count, excluded_headlines, error(optional)
    """
    try:
        stock = yf.Ticker(ticker)

        try:
            info = stock.info or {}
        except Exception:
            info = {}

        company_keywords = _get_company_keywords(ticker, info)

        # FIX 1: stock.news is a property — no retry loop needed
        news = stock.news or []
        if not news:
            return _empty_result()

        raw_headlines      = []
        excluded_headlines = []

        for item in news[:20]:
            try:
                content  = item.get('content', {})
                title    = content.get('title', '') or item.get('title', '')
                pub_time = content.get('pubDate') or item.get('providerPublishTime')
                url      = (content.get('canonicalUrl') or {}).get('url', '') \
                           or item.get('link', '')
            except Exception:
                title, pub_time, url = '', None, ''

            if not title:
                continue

            if not _is_relevant(title, company_keywords):
                excluded_headlines.append({'title': title, 'reason': 'not_relevant'})
                continue

            compound  = max(-1.0, min(1.0,
                            analyzer.polarity_scores(title)['compound'] + _financial_boost(title)))
            news_type = _categorize_news_type(title)
            # FIX 6 (entity boost): applied to weight, not compound
            has_influencer = any(name in title.lower() for name in INFLUENCERS)
            entity_bonus   = 0.1 if has_influencer and any(kw in title.lower() for kw in company_keywords[:3]) else 0.0
            weight = round(
                _news_age_weight(pub_time) *
                _source_credibility(url) *
                NEWS_TYPE_WEIGHTS.get(news_type, 1.0) +
                entity_bonus,
                3
            )

            raw_headlines.append(_build_headline(title, compound, weight, url, news_type))

        headlines = _deduplicate(raw_headlines)[:10]

        # FIX 5: Unified fallback — no duplicated loop; uses same _build_headline helper
        if not headlines and news:
            for item in news[:10]:
                try:
                    content = item.get('content', {})
                    title   = content.get('title', '') or item.get('title', '')
                    pub_time = content.get('pubDate') or item.get('providerPublishTime')
                    url      = (content.get('canonicalUrl') or {}).get('url', '') \
                               or item.get('link', '')
                    if not title:
                        continue
                    compound  = max(-1.0, min(1.0,
                                    analyzer.polarity_scores(title)['compound'] + _financial_boost(title)))
                    news_type = _categorize_news_type(title)
                    weight    = round(_news_age_weight(pub_time) * _source_credibility(url) * 0.75, 3)
                    headlines.append(_build_headline(title, compound, weight, url, news_type))
                except Exception:
                    continue

        if not headlines:
            r = _empty_result()
            r['excluded_count']     = len(excluded_headlines)
            r['excluded_headlines'] = excluded_headlines[:3]
            return r

        # ── Aggregation ───────────────────────────────────────────────
        weighted_comps = [h['compound'] * h['weight'] for h in headlines]
        raw_comps      = [h['compound'] for h in headlines]
        avg_compound   = float(np.mean(weighted_comps))

        total          = len(raw_comps)
        positive_ratio = sum(1 for c in raw_comps if c >= 0.05)  / total
        negative_ratio = sum(1 for c in raw_comps if c <= -0.05) / total   # FIX 7a: stored explicitly

        strong_neg = [h for h in headlines if h['compound'] <= -0.5]
        tail_risk  = len(strong_neg) > 0

        if tail_risk:
            worst        = min(h['compound'] for h in strong_neg)
            avg_compound = avg_compound * 0.6 + worst * 0.4

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

        # FIX 7b: confidence scales with news_count (reaches full weight at 5+ articles)
        news_factor = min(total / 5.0, 1.0)
        confidence  = round(min(abs(avg_compound) * 100, 100) * news_factor, 1)

        return {
            'label'             : label,
            'score'             : score,
            'compound'          : round(avg_compound, 4),
            'confidence'        : confidence,
            'news_count'        : len(headlines),
            'positive_ratio'    : round(positive_ratio, 2),
            'negative_ratio'    : round(negative_ratio, 2),
            'tail_risk'         : tail_risk,
            'headlines'         : headlines,
            'excluded_count'    : len(excluded_headlines),
            'excluded_headlines': excluded_headlines[:3],
        }

    except Exception as e:
        return {**_empty_result(), 'error': str(e)}


def _empty_result() -> dict:
    return {
        'label'             : 'NEUTRAL',
        'score'             : 0,
        'compound'          : 0.0,
        'confidence'        : 0.0,
        'news_count'        : 0,
        'positive_ratio'    : 0.0,
        'negative_ratio'    : 0.0,
        'tail_risk'         : False,
        'headlines'         : [],
        'excluded_count'    : 0,
        'excluded_headlines': [],
    }


def print_sentiment(ticker: str, sentiment: dict):
    label_colors = {
        'BULLISH'         : '🟢',
        'SLIGHTLY BULLISH': '🟡',
        'NEUTRAL'         : '⚪',
        'SLIGHTLY BEARISH': '🟠',
        'BEARISH'         : '🔴',
    }
    NEWS_TYPE_ICONS = {
        'earnings'  : '💰',
        'analyst'   : '📊',
        'influencer': '👤',
        'insider'   : '🏢',
        'regulatory': '⚖️',
        'general'   : '📰',
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

    # FIX 7a: Use explicit negative_ratio; show Neutral separately
    neutral_ratio = 1.0 - sentiment['positive_ratio'] - sentiment['negative_ratio']
    print(f"Positive  : {sentiment['positive_ratio'] * 100:.0f}%  |  "
          f"Negative  : {sentiment['negative_ratio'] * 100:.0f}%  |  "
          f"Neutral   : {neutral_ratio * 100:.0f}%")

    if sentiment.get('tail_risk'):
        print("  ⚠️  TAIL RISK: Strong negative news detected — result adjusted.")

    if sentiment['headlines']:
        print("\nTop Headlines:")
        for i, h in enumerate(sentiment['headlines'][:5], 1):
            icon_h     = '📈' if h['sentiment'] == 'Positive' else '📉' if h['sentiment'] == 'Negative' else '➖'
            type_icon  = NEWS_TYPE_ICONS.get(h.get('news_type', 'general'), '📰')
            print(f"  {i}. {icon_h} {type_icon} [{h['sentiment']:8s}] (w={h['weight']}) {h['title'][:75]}")

    excluded = sentiment.get('excluded_count', 0)
    if excluded:
        print(f"\n  🚫 Excluded {excluded} irrelevant headline(s):")
        for ex in sentiment.get('excluded_headlines', []):
            print(f"    - {ex['title'][:75]}")
