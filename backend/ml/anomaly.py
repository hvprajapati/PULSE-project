import numpy as np
from sklearn.ensemble import IsolationForest
from collections import Counter
from datetime import datetime
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_features(signals: list[dict]) -> np.ndarray:
    """
    Convert signals into a numeric feature matrix for anomaly detection.
    Each row = one signal, columns = numeric features.
    """
    rows = []
    for s in signals:
        source = s.get("source", "")

        if source == "reddit":
            rows.append([
                float(s.get("score", 0)),
                float(s.get("num_comments", 0)),
                float(s.get("upvote_ratio", 0)),
                0.0, 0.0
            ])
        elif source == "news":
            rows.append([0.0, 0.0, 0.0, 1.0, 0.0])

        elif source == "github":
            rows.append([
                float(s.get("stars", 0)),
                float(s.get("forks", 0)),
                0.0, 0.0, 0.0
            ])
        elif source == "finance":
            rows.append([
                0.0, 0.0,
                abs(float(s.get("change_pct", 0) or 0)),
                float(s.get("volume", 0)) / 1_000_000,
                0.0
            ])
        elif source == "jobs":
            rows.append([
                0.0, 0.0, 0.0, 0.0,
                float(len(s.get("matched_tags", [])))
            ])
        else:
            rows.append([0.0, 0.0, 0.0, 0.0, 0.0])

    return np.array(rows, dtype=float)


def detect_anomalies(
    signals:    list[dict],
    embeddings: np.ndarray,
    contamination: float = 0.05   # expect ~5% of signals to be anomalies
) -> list[dict]:
    """
    Use Isolation Forest to detect anomalous signals.
    Anomalous = unusually high engagement, volume, or activity.
    These are the signals PULSE should pay attention to most.
    """
    numeric_features = extract_features(signals)

    # Combine numeric features + embeddings for richer detection
    combined = np.hstack([numeric_features, embeddings])

    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
        n_jobs=-1
    )
    iso.fit(combined)

    # -1 = anomaly, 1 = normal
    predictions = iso.predict(combined)
    scores      = iso.decision_function(combined)  # lower = more anomalous

    enriched = []
    anomaly_count = 0
    for i, signal in enumerate(signals):
        s = signal.copy()
        is_anomaly          = bool(predictions[i] == -1)
        s["is_anomaly"]     = is_anomaly
        s["anomaly_score"]  = round(float(scores[i]), 4)
        if is_anomaly:
            anomaly_count += 1
        enriched.append(s)

    print(f"[anomaly] {anomaly_count} anomalies detected out of {len(signals)} signals")
    return enriched


def get_top_anomalies(
    signals: list[dict],
    top_n:   int = 10
) -> list[dict]:
    """Return the most anomalous signals sorted by score."""
    anomalies = [s for s in signals if s.get("is_anomaly")]
    anomalies.sort(key=lambda x: x.get("anomaly_score", 0))  # most anomalous first
    return anomalies[:top_n]


def anomaly_topic_report(anomalies: list[dict]) -> dict:
    """
    Summarize what topics/sources are driving the anomalies.
    This is the insight PULSE surfaces to the LLM.
    """
    sources  = Counter(a["source"]    for a in anomalies)
    topics   = []

    for a in anomalies:
        t = (
            a.get("topic")      or
            a.get("subreddit")  or
            a.get("symbol")     or
            a.get("language")   or
            ", ".join(a.get("matched_tags", [])[:2]) or
            "unknown"
        )
        topics.append(t)

    topic_counts = Counter(topics)

    return {
        "total_anomalies":  len(anomalies),
        "by_source":        dict(sources),
        "top_topics":       topic_counts.most_common(5),
        "generated_at":     datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    import asyncio
    from ingestion.reddit_fetcher  import fetch_all_reddit_signals
    from ingestion.news_fetcher    import fetch_all_news_signals
    from ingestion.github_fetcher  import fetch_all_github_signals
    from ingestion.jobs_fetcher    import fetch_all_jobs_signals
    from ml.embedder               import embed_signals

    async def run():
        print("[anomaly] fetching live signals...")
        reddit, news, github, jobs = await asyncio.gather(
            fetch_all_reddit_signals(),
            fetch_all_news_signals(),
            fetch_all_github_signals(),
            fetch_all_jobs_signals(),
        )
        all_signals = reddit + news + github + jobs
        print(f"[anomaly] total signals: {len(all_signals)}")

        embeddings = embed_signals(all_signals)
        enriched   = detect_anomalies(all_signals, embeddings)

        top = get_top_anomalies(enriched, top_n=5)
        print("\n[anomaly] top 5 anomalous signals:\n")
        for a in top:
            title = (
                a.get("title") or
                a.get("name")  or
                a.get("symbol") or ""
            )[:70]
            print(f"  [{a['source']:>7}] score: {a['anomaly_score']:>7} | {title}")

        report = anomaly_topic_report(top)
        print(f"\n[anomaly] report:")
        print(f"  total anomalies : {report['total_anomalies']}")
        print(f"  by source       : {report['by_source']}")
        print(f"  top topics      : {report['top_topics']}")

    asyncio.run(run())