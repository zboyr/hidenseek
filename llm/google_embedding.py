"""Google Embedding API client for paraphrase quality validation."""

import os
import numpy as np
from google import genai


def get_embeddings(texts: list, model: str = "gemini-embedding-001") -> list:
    """Get embedding vectors for a batch of texts using Google's embedding API."""
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    result = client.models.embed_content(model=model, contents=texts)
    return [e.values for e in result.embeddings]


def cosine_similarity(vec1, vec2) -> float:
    """Compute cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def validate_paraphrase(
    original: str,
    paraphrase: str,
    threshold: float = 0.75,
    model: str = "gemini-embedding-001",
) -> tuple:
    """
    Validate that a paraphrase preserves semantic meaning of the original.

    Returns:
        (is_valid: bool, similarity_score: float)
    """
    embeddings = get_embeddings([original, paraphrase], model=model)
    similarity = cosine_similarity(embeddings[0], embeddings[1])
    return similarity >= threshold, similarity
