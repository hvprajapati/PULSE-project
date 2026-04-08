import httpx
import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
BASE_URL = "https://api.github.com/search/repositories"

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept":        "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
}

TOPICS = [
    "machine-learning",
    "large-language-model",
    "fastapi",
    "artificial-intelligence",
    "llm-agent",
    "rag",
    "rust",
    "golang",
]

async def fetch_trending_repos(
    client: httpx.AsyncClient,
    topic: str,
    days_back: int = 7
) -> list[dict]:
    since_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "q":        f"topic:{topic} created:>{since_date}",
        "sort":     "stars",
        "order":    "desc",
        "per_page": 10,
    }

    try:
        response = await client.get(
            BASE_URL,
            params=params,
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        repos = []
        for repo in data.get("items", []):
            repos.append({
                "source":       "github",
                "topic":        topic,
                "name":         repo.get("full_name", ""),
                "description":  repo.get("description", "")[:300],
                "stars":        repo.get("stargazers_count", 0),
                "forks":        repo.get("forks_count", 0),
                "language":     repo.get("language", ""),
                "url":          repo.get("html_url", ""),
                "created_at":   repo.get("created_at", ""),
                "pushed_at":    repo.get("pushed_at", ""),
            })
        return repos

    except Exception as e:
        print(f"[github] failed for '{topic}': {e}")
        return []


async def fetch_all_github_signals() -> list[dict]:
    # Small delay between requests to respect rate limits
    all_repos = []
    async with httpx.AsyncClient() as client:
        for topic in TOPICS:
            repos = await fetch_trending_repos(client, topic)
            all_repos.extend(repos)
            await asyncio.sleep(1)  # 1 sec gap — stays well under 5000 req/hr

    print(f"[github] fetched {len(all_repos)} repos across {len(TOPICS)} topics")
    return all_repos


if __name__ == "__main__":
    repos = asyncio.run(fetch_all_github_signals())
    for r in repos[:3]:
        print(r["topic"], "|", r["stars"], "stars |", r["name"])