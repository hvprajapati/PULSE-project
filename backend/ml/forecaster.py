import numpy as np
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Any
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_topic_from_signal(signal: dict) -> str:
    """Pull the best topic label from any signal type."""
    source = signal.get("source", "")

    if source == "reddit":
        return signal.get("subreddit", "unknown").lower()
    elif source == "news":
        return signal.get("topic", "unknown").lower()
    elif source == "github":
        return signal.get("topic") or signal.get("language") or "unknown"
    elif source == "finance":
        return signal.get("symbol", "unknown")
    elif source == "jobs":
        tags = signal.get("matched_tags", [])
        return tags[0] if tags else "unknown"
    return "unknown"


def compute_topic_velocity(signals: list[dict]) -> dict[str, dict]:
    """
    Compute how fast each topic is gaining signals.
    Velocity = signal count weighted by engagement score.
    """
    topic_data: dict[str, dict] = defaultdict(lambda: {
        "count":      0,
        "engagement": 0.0,
        "sources":    set(),
        "anomalies":  0,
    })

    for s in signals:
        topic = extract_topic_from_signal(s)
        source = s.get("source", "")

        topic_data[topic]["count"]   += 1
        topic_data[topic]["sources"].add(source)

        if s.get("is_anomaly"):
            topic_data[topic]["anomalies"] += 1

        # Engagement score per source type
        if source == "reddit":
            eng = (
                float(s.get("score", 0)) * 0.5 +
                float(s.get("num_comments", 0)) * 1.0 +
                float(s.get("upvote_ratio", 0)) * 10
            )
        elif source == "github":
            eng = (
                float(s.get("stars", 0)) * 1.0 +
                float(s.get("forks", 0)) * 2.0
            )
        elif source == "finance":
            eng = abs(float(s.get("change_pct") or 0)) * 100
        elif source == "jobs":
            eng = float(len(s.get("matched_tags", []))) * 5.0
        else:
            eng = 1.0

        topic_data[topic]["engagement"] += eng

    return dict(topic_data)


def score_topics(topic_data: dict[str, dict]) -> list[dict]:
    if not topic_data:
        return []

    # Find max values for normalization
    all_counts      = [d["count"]      for d in topic_data.values()]
    all_engagement  = [d["engagement"] for d in topic_data.values()]
    max_count       = max(all_counts)      or 1
    max_engagement  = max(all_engagement)  or 1

    scored = []
    for topic, data in topic_data.items():
        if topic == "unknown" or data["count"] < 2:
            continue

        count         = data["count"]
        source_div    = len(data["sources"])
        engagement    = data["engagement"]
        anomaly_bonus = data["anomalies"] * 0.1

        # Normalize each factor to 0-1 then weight
        norm_count      = count      / max_count
        norm_engagement = engagement / max_engagement
        norm_sources    = source_div / 5          # max 5 sources

        raw_score = (
            norm_count      * 0.35 +
            norm_sources    * 0.40 +
            norm_engagement * 0.15 +
            anomaly_bonus   * 0.10
        )

        confidence = round(min(raw_score * 100, 99.9), 1)

        # Direction
        if source_div >= 3:
            direction = "STRONG UP"
        elif source_div == 2 and count >= 5:
            direction = "UP"
        elif source_div == 2:
            direction = "MILD UP"
        elif data["anomalies"] > 0:
            direction = "SPIKE"
        else:
            direction = "NEUTRAL"

        forecast_days = 14 if source_div >= 3 else 7 if source_div == 2 else 3

        scored.append({
            "topic":          topic,
            "signal_count":   count,
            "source_count":   source_div,
            "sources":        list(data["sources"]),
            "engagement":     round(engagement, 2),
            "anomalies":      data["anomalies"],
            "confidence":     confidence,
            "direction":      direction,
            "forecast_days":  forecast_days,
            "forecast_until": (
                datetime.utcnow() + timedelta(days=forecast_days)
            ).strftime("%Y-%m-%d"),
            "scored_at":      datetime.utcnow().isoformat(),
        })

    scored.sort(key=lambda x: -x["confidence"])
    return scored


def forecast(signals: list[dict]) -> list[dict]:
    """Main entry — takes enriched signals, returns ranked topic forecasts."""
    topic_data  = compute_topic_velocity(signals)
    predictions = score_topics(topic_data)
    print(f"[forecaster] {len(predictions)} topics scored")
    return predictions


if __name__ == "__main__":
    import asyncio
    from ingestion.reddit_fetcher  import fetch_all_reddit_signals
    from ingestion.news_fetcher    import fetch_all_news_signals
    from ingestion.github_fetcher  import fetch_all_github_signals
    from ingestion.jobs_fetcher    import fetch_all_jobs_signals
    from ml.embedder               import embed_signals
    from ml.anomaly                import detect_anomalies

    async def run():
        print("[forecaster] fetching live signals...")
        reddit, news, github, jobs = await asyncio.gather(
            fetch_all_reddit_signals(),
            fetch_all_news_signals(),
            fetch_all_github_signals(),
            fetch_all_jobs_signals(),
        )
        all_signals = reddit + news + github + jobs
        print(f"[forecaster] total: {len(all_signals)} signals")

        embeddings = embed_signals(all_signals)
        enriched   = detect_anomalies(all_signals, embeddings)
        predictions = forecast(enriched)

        print("\n" + "=" * 65)
        print("  PULSE — TOP TRENDING TOPICS")
        print("=" * 65)
        print(f"  {'#':<3} {'Topic':<26} {'Direction':<12} {'Conf':>6} {'Until'}")
        print(f"  {'-'*3} {'-'*26} {'-'*12} {'-'*6} {'-'*10}")
        for i, p in enumerate(predictions[:10], 1):
            sources_str = "+".join(p["sources"])
            print(
                f"  {i:<3} "
                f"{p['topic']:<26} "
                f"{p['direction']:<12} "
                f"{p['confidence']:>5}%  "
                f"{p['forecast_until']}  "
                f"[{sources_str}]"
            )
        print("=" * 65)

    asyncio.run(run())