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

        # Read label order from model config to avoid hardcoding
        scores = probabilities[0].tolist()
        id2label = _model.config.id2label
        labels = [id2label[i].lower() for i in range(len(id2label))]

        # Find the highest scoring label
        max_idx = scores.index(max(scores))
        sentiment = labels[max_idx]
        confidence = scores[max_idx]

        score_dict = {labels[i]: round(scores[i], 4) for i in range(len(labels))}
        # Ensure standard keys exist for downstream consumers
        result_scores = {
            "positive": score_dict.get("positive", 0.0),
            "negative": score_dict.get("negative", 0.0),
            "neutral":  score_dict.get("neutral", 0.0),
        }

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "scores": result_scores,
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


def analyze_sentiment(text: str) -> tuple:
    """
    Score a headline and return (sentiment_score_str, confidence_float).

    sentiment_score_str is a signed numeric string in [-1.0, +1.0] matching
    the format already stored in news_sentiment.sentiment (the API reads it
    back with float()). Positive = bullish, negative = bearish, ~0 = neutral.

    Falls back to keyword scoring when FinBERT (transformers/torch) is not
    installed, so this function never raises.
    """
    try:
        result = score_sentiment(text)
    except Exception:
        result = score_sentiment_simple(text)

    label      = result["sentiment"]       # "positive" | "negative" | "neutral"
    confidence = float(result["confidence"])

    if label == "positive":
        score = confidence
    elif label == "negative":
        score = -confidence
    else:
        score = 0.0

    return str(round(score, 4)), confidence


# Singleton pipeline for batch inference (shared across all callers)
_batch_pipeline = None
_BATCH_SIZE = 32


def _get_batch_pipeline():
    global _batch_pipeline
    if _batch_pipeline is None:
        from transformers import pipeline
        _batch_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=1,
        )
        logger.info("FinBERT batch pipeline loaded")
    return _batch_pipeline


def analyze_sentiment_batch(texts: List[str]) -> List[tuple]:
    """
    Score a list of headlines in batched forward passes (~20× faster than one-by-one).

    Processes _BATCH_SIZE=32 articles per forward pass. Returns one
    (sentiment_score_str, confidence_float) tuple per input text, in the same
    order as `texts`. Falls back to keyword scoring per-item on any error.

    Args:
        texts: List of headline strings (any length).

    Returns:
        List of (score_str, confidence) tuples matching input order.
    """
    if not texts:
        return []

    results: List[tuple] = []
    try:
        pipe = _get_batch_pipeline()
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = [t[:512] for t in texts[i:i + _BATCH_SIZE]]
            preds = pipe(batch)
            for pred in preds:
                top = pred[0] if isinstance(pred, list) else pred
                label = top["label"].lower()
                conf  = float(top["score"])
                if label == "positive":
                    score = conf
                elif label == "negative":
                    score = -conf
                else:
                    score = 0.0
                results.append((str(round(score, 4)), conf))
    except Exception as exc:
        logger.warning(f"Batch FinBERT failed ({exc}), falling back to per-item keyword scoring")
        # Keyword fallback for any items not yet processed
        already = len(results)
        for text in texts[already:]:
            r = score_sentiment_simple(text)
            conf = float(r["confidence"])
            label = r["sentiment"]
            score = conf if label == "positive" else (-conf if label == "negative" else 0.0)
            results.append((str(round(score, 4)), conf))

    return results


def score_and_update_news(news_list: List[Dict]) -> List[Dict]:
    """
    Score sentiment for a list of news articles and update the database.

    Uses batch FinBERT inference for efficiency — one forward pass per 32
    articles instead of one per article.

    Args:
        news_list: List of dicts with 'headline' key

    Returns:
        Same list with 'sentiment' and 'confidence' added to each article.
    """
    from database.db import get_connection, _execute, release_connection, _executemany

    valid = [a for a in news_list if a.get("headline")]
    if not valid:
        return []

    headlines = [a["headline"] for a in valid]
    scored_pairs = analyze_sentiment_batch(headlines)

    db_updates = []
    for article, (score_str, conf) in zip(valid, scored_pairs):
        score_f = float(score_str)
        label = "positive" if score_f > 0 else ("negative" if score_f < 0 else "neutral")
        article["sentiment"]  = label
        article["confidence"] = conf
        if article.get("id"):
            db_updates.append((score_str, conf, article["id"]))

    if db_updates:
        conn = get_connection()
        try:
            _executemany(conn,
                "UPDATE news_sentiment SET sentiment=?, confidence=? WHERE id=?",
                db_updates,
            )
            conn.commit()
        except Exception as e:
            logger.error(f"score_and_update_news: DB update error: {e}")
            conn.rollback()
        finally:
            release_connection(conn)

    return valid


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
