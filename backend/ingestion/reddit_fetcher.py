import httpx
import asyncio
from datetime import datetime
from typing import Optional

HEADERS = {
    "User-Agent": "PULSE-Signal-Engine/1.0 (research project)"
}

SUBREDDITS = [
    "programming",
    "MachineLearning", 
    "artificial",
    "datascience",
    "Python",
    "golang",
    "rust",
    "cscareerquestions",
]

async def fetch_subreddit(
    client: httpx.AsyncClient,
    subreddit: str,
    limit: int = 25,
    sort: str = "hot"
) -> list[dict]:
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={limit}"
    
    try:
        response = await client.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        posts = []
        for post in data["data"]["children"]:
            p = post["data"]
            posts.append({
                "source":      "reddit",
                "subreddit":   subreddit,
                "title":       p.get("title", ""),
                "text":        p.get("selftext", "")[:500],
                "score":       p.get("score", 0),
                "num_comments":p.get("num_comments", 0),
                "upvote_ratio":p.get("upvote_ratio", 0),
                "url":         p.get("url", ""),
                "created_at":  datetime.utcfromtimestamp(
                                   p.get("created_utc", 0)
                               ).isoformat(),
                "flair":       p.get("link_flair_text", ""),
            })
        return posts

    except Exception as e:
        print(f"[reddit] failed {subreddit}: {e}")
        return []


async def fetch_all_reddit_signals() -> list[dict]:
    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_subreddit(client, sub)
            for sub in SUBREDDITS
        ]
        results = await asyncio.gather(*tasks)
    
    all_posts = [post for batch in results for post in batch]
    print(f"[reddit] fetched {len(all_posts)} posts from {len(SUBREDDITS)} subreddits")
    return all_posts


if __name__ == "__main__":
    posts = asyncio.run(fetch_all_reddit_signals())
    for p in posts[:3]:
        print(p["subreddit"], "|", p["score"], "|", p["title"][:60])