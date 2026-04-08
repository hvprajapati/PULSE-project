from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio

from ingestion.reddit_fetcher  import fetch_all_reddit_signals
from ingestion.news_fetcher    import fetch_all_news_signals
from ingestion.github_fetcher  import fetch_all_github_signals
from ingestion.finance_fetcher import fetch_all_finance_signals
from ingestion.jobs_fetcher    import fetch_all_jobs_signals

# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("PULSE engine starting...")
    yield
    print("PULSE engine shutting down...")

app = FastAPI(
    title       = "PULSE — Predictive Universal Live Signal Engine",
    description = "Multi-source AI signal intelligence platform",
    version     = "1.0.0",
    lifespan    = lifespan
)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name":    "PULSE",
        "version": "1.0.0",
        "status":  "running",
        "sources": ["reddit", "news", "github", "finance", "jobs"]
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/signals/reddit")
async def get_reddit_signals():
    data = await fetch_all_reddit_signals()
    return {"source": "reddit", "count": len(data), "data": data}


@app.get("/signals/news")
async def get_news_signals():
    data = await fetch_all_news_signals()
    return {"source": "news", "count": len(data), "data": data}


@app.get("/signals/github")
async def get_github_signals():
    data = await fetch_all_github_signals()
    return {"source": "github", "count": len(data), "data": data}


@app.get("/signals/finance")
async def get_finance_signals():
    data = await fetch_all_finance_signals()
    return {"source": "finance", "count": len(data), "data": data}


@app.get("/signals/jobs")
async def get_jobs_signals():
    data = await fetch_all_jobs_signals()
    return {"source": "jobs", "count": len(data), "data": data}


@app.get("/signals/all")
async def get_all_signals():
    """Fetch all sources in parallel — the main PULSE endpoint."""
    reddit_task  = fetch_all_reddit_signals()
    news_task    = fetch_all_news_signals()
    github_task  = fetch_all_github_signals()
    jobs_task    = fetch_all_jobs_signals()

    # Finance is slow (rate limited) so we run separately
    reddit, news, github, jobs = await asyncio.gather(
        reddit_task,
        news_task,
        github_task,
        jobs_task
    )

    all_signals = reddit + news + github + jobs

    return {
        "status": "ok",
        "total":  len(all_signals),
        "breakdown": {
            "reddit":  len(reddit),
            "news":    len(news),
            "github":  len(github),
            "jobs":    len(jobs),
        },
        "data": all_signals
    }