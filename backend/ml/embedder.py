import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Any
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer
# ... rest of file unchanged

# Free, small, fast — runs on CPU no GPU needed
MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("[embedder] loading model...")
        _model = SentenceTransformer(MODEL_NAME)
        print("[embedder] model ready")
    return _model


def signal_to_text(signal: dict) -> str:
    """Convert any signal dict into a single string for embedding."""
    source = signal.get("source", "")

    if source == "reddit":
        return f"{signal.get('title','')} {signal.get('text','')} {signal.get('subreddit','')}"

    elif source == "news":
        return f"{signal.get('title','')} {signal.get('description','')} {signal.get('topic','')}"

    elif source == "github":
        return f"{signal.get('name','')} {signal.get('description','')} {signal.get('topic','')} {signal.get('language','')}"

    elif source == "finance":
        direction = "up" if signal.get("change", 0) > 0 else "down"
        return f"{signal.get('symbol','')} stock price {direction} {signal.get('change_pct','')} percent"

    elif source == "jobs":
        tags = " ".join(signal.get("tags", []))
        return f"{signal.get('title','')} {signal.get('company','')} {tags}"

    return str(signal)


def embed_signals(signals: list[dict]) -> np.ndarray:
    """
    Take a list of signal dicts, return an (N, 384) numpy array.
    Each row is the embedding for one signal.
    """
    model  = get_model()
    texts  = [signal_to_text(s) for s in signals]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True   # cosine similarity ready
    )
    print(f"[embedder] embedded {len(signals)} signals → shape {embeddings.shape}")
    return embeddings


if __name__ == "__main__":
    # Quick test with fake signals
    test_signals = [
        {"source": "reddit",  "title": "Rust is amazing for systems programming", "text": "just rewrote my backend", "subreddit": "rust"},
        {"source": "news",    "title": "OpenAI releases GPT-5", "description": "major AI breakthrough", "topic": "artificial intelligence"},
        {"source": "github",  "name": "microsoft/phi-3", "description": "small language model", "topic": "llm", "language": "Python"},
        {"source": "jobs",    "title": "Senior ML Engineer", "company": "Groq", "tags": ["python", "ml", "llm"]},
        {"source": "finance", "symbol": "NVDA", "change": 5.2, "change_pct": "3.1"},
    ]

    embeddings = embed_signals(test_signals)
    print(f"Embedding shape : {embeddings.shape}")
    print(f"First vector    : {embeddings[0][:5]}...")  # first 5 dims