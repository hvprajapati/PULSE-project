import httpx
import asyncio
from datetime import datetime

BASE_REMOTEOK    = "https://remoteok.com/api"
BASE_HN_JOBS     = "https://hacker-news.firebaseio.com/v0"

HEADERS = {
    "User-Agent": "PULSE-Signal-Engine/1.0 (research project)"
}

# Keywords we care about for signal detection
RELEVANT_TAGS = {
    "python", "machine-learning", "ai", "llm", "fastapi",
    "rust", "golang", "backend", "data-science", "ml",
    "deep-learning", "nlp", "devops", "cloud", "aws"
}

# ── RemoteOK ──────────────────────────────────────────────────────────────────

async def fetch_remoteok_jobs(client: httpx.AsyncClient) -> list[dict]:
    try:
        response = await client.get(
            BASE_REMOTEOK,
            headers=HEADERS,
            timeout=15
        )
        response.raise_for_status()
        data = response.json()

        # First item is metadata — skip it
        jobs = []
        for job in data[1:51]:   # cap at 50
            tags = [t.lower() for t in job.get("tags", [])]
            matched = [t for t in tags if t in RELEVANT_TAGS]
            if not matched:
                continue

            jobs.append({
                "source":      "jobs",
                "platform":    "remoteok",
                "title":       job.get("position", ""),
                "company":     job.get("company", ""),
                "tags":        tags,
                "matched_tags":matched,
                "salary_min":  job.get("salary_min", 0) or 0,
                "salary_max":  job.get("salary_max", 0) or 0,
                "url":         job.get("url", ""),
                "posted_at":   job.get("date", ""),
                "fetched_at":  datetime.utcnow().isoformat(),
            })
        return jobs

    except Exception as e:
        print(f"[jobs] remoteok failed: {e}")
        return []


# ── HackerNews Who's Hiring ───────────────────────────────────────────────────

async def fetch_hn_hiring_post_id(client: httpx.AsyncClient) -> int | None:
    """Get the latest 'Ask HN: Who is hiring?' post ID."""
    try:
        # Get top stories from Ask HN
        resp = await client.get(
            f"{BASE_HN_JOBS}/topstories.json",
            timeout=10
        )
        # HN monthly hiring thread is always pinned — search recent items
        search_resp = await client.get(
            "https://hn.algolia.com/api/v1/search?query=Ask+HN+Who+is+hiring&tags=story&hitsPerPage=1",
            timeout=10
        )
        search_resp.raise_for_status()
        hits = search_resp.json().get("hits", [])
        if hits:
            return int(hits[0]["objectID"])
        return None
    except Exception as e:
        print(f"[jobs] HN post ID fetch failed: {e}")
        return None


async def fetch_hn_comments(
    client: httpx.AsyncClient,
    post_id: int,
    limit: int = 30
) -> list[dict]:
    try:
        post_resp = await client.get(
            f"{BASE_HN_JOBS}/item/{post_id}.json",
            timeout=10
        )
        post_resp.raise_for_status()
        post = post_resp.json()

        comment_ids = post.get("kids", [])[:limit]

        async def fetch_comment(cid: int) -> dict | None:
            try:
                r = await client.get(
                    f"{BASE_HN_JOBS}/item/{cid}.json",
                    timeout=8
                )
                r.raise_for_status()
                c = r.json()
                text = c.get("text", "") or ""

                # Only keep comments that mention tech keywords
                text_lower = text.lower()
                matched = [t for t in RELEVANT_TAGS if t in text_lower]
                if not matched:
                    return None

                return {
                    "source":       "jobs",
                    "platform":     "hackernews",
                    "title":        text[:120].replace("<p>", " ").replace("&#x27;", "'"),
                    "company":      "",
                    "tags":         matched,
                    "matched_tags": matched,
                    "salary_min":   0,
                    "salary_max":   0,
                    "url":          f"https://news.ycombinator.com/item?id={cid}",
                    "posted_at":    datetime.utcfromtimestamp(
                                        c.get("time", 0)
                                    ).isoformat(),
                    "fetched_at":   datetime.utcnow().isoformat(),
                }
            except Exception:
                return None

        tasks   = [fetch_comment(cid) for cid in comment_ids]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r]

    except Exception as e:
        print(f"[jobs] HN comments failed: {e}")
        return []


# ── Main ──────────────────────────────────────────────────────────────────────

async def fetch_all_jobs_signals() -> list[dict]:
    async with httpx.AsyncClient() as client:
        # Run both sources in parallel
        remoteok_task = fetch_remoteok_jobs(client)

        hn_post_id = await fetch_hn_hiring_post_id(client)
        if hn_post_id:
            hn_task = fetch_hn_comments(client, hn_post_id)
        else:
            hn_task = asyncio.coroutine(lambda: [])()

        remoteok_jobs, hn_jobs = await asyncio.gather(remoteok_task, hn_task)

    all_jobs = remoteok_jobs + hn_jobs
    print(f"[jobs] fetched {len(all_jobs)} job signals "
          f"({len(remoteok_jobs)} remoteok, {len(hn_jobs)} hn)")
    return all_jobs


if __name__ == "__main__":
    jobs = asyncio.run(fetch_all_jobs_signals())
    for j in jobs[:3]:
        print(
            j["platform"], "|",
            j["matched_tags"], "|",
            j["title"][:60]
        )