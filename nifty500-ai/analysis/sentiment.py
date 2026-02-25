"""
Nifty 500 AI — News Sentiment Analyzer (FinBERT)

Uses ProsusAI/finbert model to score financial news headlines
as positive, negative, or neutral with confidence scores.
Aggregates individual scores into a Fear & Greed index (0-100).

First-time usage downloads the FinBERT model (~400MB).

Usage:
    from analysis.sentiment import score_sentiment, aggregate_sentiment

    # Score a single headline
    result = score_sentiment("Nifty 50 hits all-time high on strong FII buying")
    print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2f})")

    # Aggregate multiple headlines into Fear & Greed score
    score = aggregate_sentiment(news_list)
    print(f"Market Sentiment: {score['score']}/100 — {score['label']}")
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy-load the model (only when first needed)
_model = None
_tokenizer = None


def _load_model():
    """
    Load the FinBERT model and tokenizer.

    Downloads the model on first use (~400MB).
    Model: ProsusAI/finbert — trained on financial text.
    """
    global _model, _tokenizer

    if _model is not None:
        return

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        logger.info("Loading FinBERT model (first time may take a few minutes)...")
        print("🤖 Loading FinBERT sentiment model...")
        print("   (First time downloads ~400MB — please wait)")

        model_name = "ProsusAI/finbert"
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _model.eval()  # Set to evaluation mode (no training)

        logger.info("FinBERT model loaded successfully")
        print("✅ FinBERT model loaded!")

    except ImportError:
        logger.error(
            "transformers or torch not installed. "
            "Run: pip install transformers torch"
        )
        raise
    except Exception as e:
        logger.error(f"Failed to load FinBERT model: {e}")
        raise


def score_sentiment(text: str) -> Dict:
    """
    Score the sentiment of a financial text using FinBERT.

    Args:
        text: Financial news headline or text

    Returns:
        Dict with keys:
            - sentiment: "positive", "negative", or "neutral"
            - confidence: float between 0.0 and 1.0
            - scores: dict with all three class probabilities

    Example:
        result = score_sentiment("RBI cuts repo rate by 25bps, market rallies")
        # {'sentiment': 'positive', 'confidence': 0.92, 'scores': {...}}
    """
    try:
        import torch

        _load_model()

        # Tokenize the text
        inputs = _tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Run through model
        with torch.no_grad():
            outputs = _model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # FinBERT labels: positive=0, negative=1, neutral=2
        scores = probabilities[0].tolist()
        labels = ["positive", "negative", "neutral"]

        # Find the highest scoring label
        max_idx = scores.index(max(scores))
        sentiment = labels[max_idx]
        confidence = scores[max_idx]

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "scores": {
                "positive": round(scores[0], 4),
                "negative": round(scores[1], 4),
                "neutral": round(scores[2], 4),
            },
        }

    except Exception as e:
        logger.error(f"Sentiment scoring failed: {e}")
        # Raise the exception so the caller can fall back to the simple scorer
        raise e


def score_sentiment_simple(text: str) -> Dict:
    """
    Simple keyword-based sentiment scoring (no ML model needed).

    Use this as a fallback when FinBERT is not installed or too slow.

    Args:
        text: News headline

    Returns:
        Dict with sentiment and confidence.
    """
    text_lower = text.lower()

    # Positive keywords
    positive_words = [
        "rally", "surge", "gain", "bull", "high", "record", "growth",
        "profit", "buy", "upgrade", "strong", "positive", "recovery",
        "breakout", "upside", "boom", "optimistic", "outperform",
    ]

    # Negative keywords
    negative_words = [
        "fall", "crash", "bear", "low", "loss", "decline", "drop",
        "sell", "downgrade", "weak", "negative", "recession", "crisis",
        "correction", "downside", "fear", "panic", "underperform",
    ]

    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    total = pos_count + neg_count

    if total == 0:
        return {"sentiment": "neutral", "confidence": 0.5}
    elif pos_count > neg_count:
        return {"sentiment": "positive", "confidence": min(0.9, pos_count / total)}
    elif neg_count > pos_count:
        return {"sentiment": "negative", "confidence": min(0.9, neg_count / total)}
    else:
        return {"sentiment": "neutral", "confidence": 0.5}


def aggregate_sentiment(news_list: List[Dict]) -> Dict:
    """
    Aggregate sentiment from multiple news articles into a single
    Fear & Greed score (0-100).

    Scoring:
        - Each article contributes: positive = +1, negative = -1, neutral = 0
        - Weighted by confidence level
        - Raw score normalized to 0-100 scale

    Fear & Greed Labels:
        0-20   = "Extreme Fear"
        21-40  = "Fear"
        41-60  = "Neutral"
        61-80  = "Greed"
        81-100 = "Extreme Greed"

    Args:
        news_list: List of dicts, each with at least 'headline' key.
            If 'sentiment' and 'confidence' are present, uses them.
            Otherwise scores each headline using FinBERT.

    Returns:
        Dict with score (0-100), label, article_count, breakdown.

    Example:
        news = [
            {"headline": "Market hits new high"},
            {"headline": "FII selling intensifies"},
        ]
        result = aggregate_sentiment(news)
        print(f"Score: {result['score']}/100 — {result['label']}")
    """
    if not news_list:
        return {
            "score": 50.0,
            "label": "Neutral",
            "article_count": 0,
            "breakdown": {"positive": 0, "negative": 0, "neutral": 0},
        }

    weighted_sum = 0.0
    total_weight = 0.0
    breakdown = {"positive": 0, "negative": 0, "neutral": 0}

    for article in news_list:
        # Get or calculate sentiment
        if article.get("sentiment") is not None and article.get("confidence") is not None:
            sentiment = article["sentiment"]
            confidence = article["confidence"]
        elif "headline" in article:
            try:
                result = score_sentiment(article["headline"])
                sentiment = result["sentiment"]
                confidence = result["confidence"]
            except Exception:
                result = score_sentiment_simple(article.get("headline", ""))
                sentiment = result["sentiment"]
                confidence = result["confidence"]
        else:
            continue

        # Convert sentiment to numeric score
        if sentiment == "positive":
            score_value = 1.0
            breakdown["positive"] += 1
        elif sentiment == "negative":
            score_value = -1.0
            breakdown["negative"] += 1
        else:
            score_value = 0.0
            breakdown["neutral"] += 1

        # Weighted contribution
        weighted_sum += score_value * confidence
        total_weight += confidence

    # Calculate normalized score (0-100)
    if total_weight > 0:
        raw_score = weighted_sum / total_weight  # -1 to +1
        normalized_score = (raw_score + 1) / 2 * 100  # 0 to 100
    else:
        normalized_score = 50.0

    normalized_score = round(max(0, min(100, normalized_score)), 1)

    # Determine label
    if normalized_score <= 20:
        label = "Extreme Fear"
    elif normalized_score <= 40:
        label = "Fear"
    elif normalized_score <= 60:
        label = "Neutral"
    elif normalized_score <= 80:
        label = "Greed"
    else:
        label = "Extreme Greed"

    return {
        "score": normalized_score,
        "label": label,
        "article_count": len(news_list),
        "breakdown": breakdown,
    }


def score_and_update_news(news_list: List[Dict]) -> List[Dict]:
    """
    Score sentiment for a list of news articles and update the database.

    Args:
        news_list: List of dicts with 'headline' key

    Returns:
        Same list with 'sentiment' and 'confidence' added to each article.
    """
    from database.db import get_connection

    scored_news = []
    use_finbert = True

    for article in news_list:
        headline = article.get("headline", "")
        if not headline:
            continue

        # Try FinBERT first, fall back to simple scoring
        try:
            if use_finbert:
                result = score_sentiment(headline)
            else:
                result = score_sentiment_simple(headline)
        except Exception:
            use_finbert = False
            result = score_sentiment_simple(headline)

        article["sentiment"] = result["sentiment"]
        article["confidence"] = result["confidence"]
        scored_news.append(article)

        # Update database if the article has an ID
        if article.get("id"):
            conn = get_connection()
            try:
                conn.execute(
                    """UPDATE news_sentiment
                    SET sentiment = ?, confidence = ?
                    WHERE id = ?""",
                    (result["sentiment"], result["confidence"], article["id"]),
                )
                conn.commit()
            except Exception as e:
                logger.error(f"Error updating news sentiment: {e}")
            finally:
                conn.close()

    return scored_news


# ==========================================
# Quick test
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test headlines (no ML model needed for simple scoring)
    test_headlines = [
        "Nifty 50 hits record high on strong FII buying",
        "Market crashes 500 points amid global sell-off",
        "RBI keeps repo rate unchanged, markets steady",
        "IT stocks rally as TCS posts strong quarterly results",
        "Banking stocks decline on NPA concerns",
    ]

    print("\n📊 Sentiment Analysis Test (Simple Scorer)\n")
    news_list = [{"headline": h} for h in test_headlines]

    for h in test_headlines:
        result = score_sentiment_simple(h)
        emoji = "🟢" if result["sentiment"] == "positive" else "🔴" if result["sentiment"] == "negative" else "⚪"
        print(f"  {emoji} [{result['sentiment']:8s}] ({result['confidence']:.2f}) — {h}")

    print("\n📈 Aggregate Sentiment:")
    agg = aggregate_sentiment(news_list)
    print(f"  Score: {agg['score']}/100 — {agg['label']}")
    print(f"  Breakdown: {agg['breakdown']}")

    # Uncomment to test with FinBERT (requires transformers + torch):
    # print("\n🤖 FinBERT Test:")
    # for h in test_headlines:
    #     result = score_sentiment(h)
    #     print(f"  [{result['sentiment']}] ({result['confidence']:.2f}) — {h}")
