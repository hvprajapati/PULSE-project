import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from collections import defaultdict
from typing import Any

def cluster_signals(
    signals:    list[dict],
    embeddings: np.ndarray,
    eps:        float = 0.3,
    min_samples: int  = 2
) -> list[dict]:
    """
    Group similar signals using DBSCAN clustering.
    
    eps         = how close two signals must be to be in the same cluster
                  (lower = stricter, higher = more grouped)
    min_samples = minimum signals to form a cluster
                  (2 means even pairs get clustered)
    
    Returns signals with cluster_id added to each.
    """

    # Normalize just in case (embedder already does this, belt + suspenders)
    normed = normalize(embeddings)

    # DBSCAN with cosine-like distance (1 - cosine_similarity)
    # metric='cosine' works directly on normalized vectors
    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="cosine",
        n_jobs=-1          # use all CPU cores
    ).fit(normed)

    labels = db.labels_   # -1 = noise (no cluster)

    # Attach cluster_id to each signal
    clustered = []
    for i, signal in enumerate(signals):
        enriched = signal.copy()
        enriched["cluster_id"] = int(labels[i])
        clustered.append(enriched)

    # Summary stats
    unique_clusters = set(labels) - {-1}
    noise_count     = list(labels).count(-1)
    print(f"[clusterer] {len(unique_clusters)} clusters found, "
          f"{noise_count} noise signals, "
          f"{len(signals) - noise_count} clustered")

    return clustered


def summarize_clusters(clustered_signals: list[dict]) -> list[dict]:
    """
    For each cluster, produce a summary:
    - which sources contributed
    - top titles/names
    - signal strength (size of cluster)
    """
    groups: dict[int, list[dict]] = defaultdict(list)
    for s in clustered_signals:
        cid = s.get("cluster_id", -1)
        if cid != -1:
            groups[cid].append(s)

    summaries = []
    for cid, signals in sorted(groups.items(), key=lambda x: -len(x[1])):
        sources  = list({s["source"] for s in signals})
        titles   = []
        for s in signals[:3]:
            t = s.get("title") or s.get("name") or s.get("symbol") or ""
            if t:
                titles.append(t[:80])

        # Signal strength = how many sources agree on this cluster
        source_diversity = len(sources)
        cluster_size     = len(signals)

        # Strength score: bigger + more diverse sources = stronger signal
        strength = round(
            (cluster_size * 0.6) + (source_diversity * 0.4), 2
        )

        summaries.append({
            "cluster_id":       cid,
            "size":             cluster_size,
            "sources":          sources,
            "source_diversity": source_diversity,
            "strength_score":   strength,
            "sample_titles":    titles,
        })

    # Sort by strength — strongest signal first
    summaries.sort(key=lambda x: -x["strength_score"])
    return summaries


if __name__ == "__main__":
    import asyncio
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from ingestion.reddit_fetcher  import fetch_all_reddit_signals
    from ingestion.news_fetcher    import fetch_all_news_signals
    from ingestion.github_fetcher  import fetch_all_github_signals
    from ingestion.jobs_fetcher    import fetch_all_jobs_signals
    from ml.embedder               import embed_signals

    async def run():
        print("[clusterer] fetching live signals from all sources...")
        reddit, news, github, jobs = await asyncio.gather(
            fetch_all_reddit_signals(),
            fetch_all_news_signals(),
            fetch_all_github_signals(),
            fetch_all_jobs_signals(),
        )
        all_signals = reddit + news + github + jobs
        print(f"[clusterer] total signals: {len(all_signals)}")

        print("[clusterer] embedding signals...")
        embeddings = embed_signals(all_signals)

        print("[clusterer] clustering...")
        clustered = cluster_signals(all_signals, embeddings)

        print("\n[clusterer] top clusters by strength:\n")
        summaries = summarize_clusters(clustered)

        for s in summaries[:8]:
            print(
                f"  Cluster {s['cluster_id']:>2} | "
                f"size: {s['size']:>3} | "
                f"strength: {s['strength_score']:>5} | "
                f"sources: {s['sources']}"
            )
            for t in s["sample_titles"]:
                print(f"             → {t}")
            print()

    asyncio.run(run())