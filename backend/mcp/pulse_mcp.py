import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP

from ingestion.reddit_fetcher  import fetch_all_reddit_signals
from ingestion.news_fetcher    import fetch_all_news_signals
from ingestion.github_fetcher  import fetch_all_github_signals
from ingestion.jobs_fetcher    import fetch_all_jobs_signals
from ml.embedder               import embed_signals
from ml.anomaly                import detect_anomalies
from ml.forecaster             import forecast

mcp = FastMCP("PULSE Signal Intelligence Server")


@mcp.tool()
async def get_reddit_signals(limit: int = 10) -> str:
    """
    Fetch latest Reddit posts about tech topics.
    Returns posts with scores, comment counts and subreddit.
    """
    import json
    signals = await fetch_all_reddit_signals()
    return json.dumps(signals[:limit], indent=2)


@mcp.tool()
async def get_news_signals(topic: str = "") -> str:
    """
    Fetch latest tech news articles.
    Optionally filter by topic e.g. 'artificial intelligence', 'python'.
    """
    import json
    signals = await fetch_all_news_signals()
    if topic:
        signals = [s for s in signals if topic.lower() in s.get("topic", "").lower()]
    return json.dumps(signals[:20], indent=2)


@mcp.tool()
async def get_github_signals(topic: str = "") -> str:
    """
    Fetch trending GitHub repositories created this week.
    Optionally filter by topic e.g. 'rust', 'llm', 'fastapi'.
    """
    import json
    signals = await fetch_all_github_signals()
    if topic:
        signals = [s for s in signals if topic.lower() in s.get("topic", "").lower()]
    return json.dumps(signals[:20], indent=2)


@mcp.tool()
async def get_jobs_signals() -> str:
    """
    Fetch job postings from RemoteOK and HackerNews hiring threads.
    Returns jobs with matched tech tags and salary info.
    """
    import json
    signals = await fetch_all_jobs_signals()
    return json.dumps(signals[:20], indent=2)


@mcp.tool()
async def get_pulse_forecast(top_n: int = 10) -> str:
    """
    Run the full PULSE ML pipeline.
    Fetches all sources, embeds signals, detects anomalies,
    and returns ranked topic forecasts with confidence scores
    and trend direction. This is the main PULSE intelligence output.
    """
    import json

    reddit, news, github, jobs = await asyncio.gather(
        fetch_all_reddit_signals(),
        fetch_all_news_signals(),
        fetch_all_github_signals(),
        fetch_all_jobs_signals(),
    )
    all_signals = reddit + news + github + jobs
    embeddings  = embed_signals(all_signals)
    enriched    = detect_anomalies(all_signals, embeddings)
    predictions = forecast(enriched)

    return json.dumps(predictions[:top_n], indent=2)


@mcp.tool()
async def get_anomalies(top_n: int = 5) -> str:
    """
    Return the most anomalous signals detected across all sources.
    Anomalies = sudden spikes or unusual activity worth investigating.
    """
    import json

    reddit, news, github, jobs = await asyncio.gather(
        fetch_all_reddit_signals(),
        fetch_all_news_signals(),
        fetch_all_github_signals(),
        fetch_all_jobs_signals(),
    )
    all_signals = reddit + news + github + jobs
    embeddings  = embed_signals(all_signals)
    enriched    = detect_anomalies(all_signals, embeddings)
    anomalies   = [s for s in enriched if s.get("is_anomaly")]
    anomalies.sort(key=lambda x: x.get("anomaly_score", 0))

    return json.dumps(anomalies[:top_n], indent=2)


if __name__ == "__main__":
    print("[mcp] PULSE MCP server starting on stdio...")
    mcp.run()