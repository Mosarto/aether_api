"""Shared infrastructure clients.

LLM access does not live here: every completion goes through app.llm (Agnes).
"""

from qdrant_client import QdrantClient

from app.config import QDRANT_API_KEY, QDRANT_URL

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
