import httpx
import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
BASE_URL = "https://newsapi.org/v2/everything"

TOPICS = [
    "artificial intelligence",
    "machine learning",
    "Python programming",
    "software engineering",
    "tech startup",
    "large language model",
]

async def fetch_topic(
    client: httpx.AsyncClient,
    topic: str,
    days_back: int = 2
) -> list[dict]:
    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "q":        topic,
        "from":     from_date,
        "sortBy":   "popularity",
        "language": "en",
        "pageSize": 20,
        "apiKey":   NEWS_API_KEY,
    }

    try:
        response = await client.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        articles = []
        for article in data.get("articles", []):
            articles.append({
                "source":       "news",
                "topic":        topic,
                "title":        article.get("title", ""),
                "description":  article.get("description", "")[:500],
                "publisher":    article.get("source", {}).get("name", ""),
                "url":          article.get("url", ""),
                "published_at": article.get("publishedAt", ""),
            })
        return articles

    except Exception as e:
        print(f"[news] failed for '{topic}': {e}")
        return []


async def fetch_all_news_signals() -> list[dict]:
    async with httpx.AsyncClient() as client:
        tasks = [fetch_topic(client, topic) for topic in TOPICS]
        results = await asyncio.gather(*tasks)

    all_articles = [a for batch in results for a in batch]
    print(f"[news] fetched {len(all_articles)} articles across {len(TOPICS)} topics")
    return all_articles


if __name__ == "__main__":
    articles = asyncio.run(fetch_all_news_signals())
    for a in articles[:3]:
        print(a["publisher"], "|", a["topic"], "|", a["title"][:60])