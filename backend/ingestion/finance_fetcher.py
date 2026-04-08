import httpx
import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
BASE_URL = "https://www.alphavantage.co/query"

# Top tech/AI related stocks to track
SYMBOLS = [
    "NVDA",   # Nvidia — AI chips
    "MSFT",   # Microsoft — OpenAI partner
    "GOOGL",  # Google — Gemini
    "META",   # Meta — Llama
    "AMZN",   # Amazon — AWS AI
    "TSLA",   # Tesla — AI/robotics
]

async def fetch_stock_quote(
    client: httpx.AsyncClient,
    symbol: str
) -> dict | None:
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol":   symbol,
        "apikey":   ALPHA_VANTAGE_KEY,
    }

    try:
        response = await client.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        quote = data.get("Global Quote", {})
        if not quote:
            print(f"[finance] empty response for {symbol} — rate limit likely")
            return None

        return {
            "source":           "finance",
            "symbol":           symbol,
            "price":            float(quote.get("05. price", 0)),
            "change":           float(quote.get("09. change", 0)),
            "change_pct":       quote.get("10. change percent", "0%").replace("%", ""),
            "volume":           int(quote.get("06. volume", 0)),
            "high":             float(quote.get("03. high", 0)),
            "low":              float(quote.get("04. low", 0)),
            "previous_close":   float(quote.get("08. previous close", 0)),
            "fetched_at":       datetime.utcnow().isoformat(),
        }

    except Exception as e:
        print(f"[finance] failed for {symbol}: {e}")
        return None


async def fetch_all_finance_signals() -> list[dict]:
    results = []
    async with httpx.AsyncClient() as client:
        for symbol in SYMBOLS:
            quote = await fetch_stock_quote(client, symbol)
            if quote:
                results.append(quote)
            # Alpha Vantage free tier = 25 requests/day, 5/min
            # So we wait 13 seconds between each call to stay safe
            await asyncio.sleep(13)

    print(f"[finance] fetched {len(results)} stock quotes")
    return results


if __name__ == "__main__":
    print("[finance] starting — this takes ~80 seconds due to free tier rate limits...")
    quotes = asyncio.run(fetch_all_finance_signals())
    for q in quotes[:3]:
        print(
            q["symbol"], "|",
            f"${q['price']:.2f}", "|",
            f"change: {q['change_pct']}%"
        )