import logging
import time
from typing import Any, Dict, List

from apps.embeddings.vector_store import get_embeddings_model

log = logging.getLogger(__name__)


def generate_embedding_and_push(text: str, metadata: Dict[str, Any], listing_id: int) -> Dict[str, Any]:
    """
    Generates an embedding for the given text using LangChain OpenAI embeddings.
    """
    if not text:
        return {}

    retries = 3
    model = get_embeddings_model()
    for attempt in range(retries):
        try:
            vector = model.embed_query(text)
            return {
                "vector_cache": vector,
                "vector_id": str(listing_id),
                "index_name": "default",
            }
        except Exception as e:
            log.warning("Embedding failed (attempt %s): %s", attempt + 1, e)
            time.sleep(1)

    raise Exception("Failed to generate embedding.")


def get_batch_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a list of texts in one batch call.
    """
    if not texts:
        return []

    try:
        return get_embeddings_model().embed_documents(texts)
    except Exception as e:
        log.error("Batch embedding failed: %s", e)
        return []
