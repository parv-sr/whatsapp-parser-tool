import logging
import time
from typing import Dict, Any
from django.conf import settings
from openai import OpenAI

log = logging.getLogger(__name__)

client = OpenAI(api_key=getattr(settings, "OPENAI_API_KEY", None))

def generate_embedding_and_push(text: str, metadata: Dict[str, Any], listing_id: int) -> Dict[str, Any]:
    """
    Generates an embedding for the given text using OpenAI.
    """
    if not text:
        return {}

    retries = 3
    for attempt in range(retries):
        try:
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=1536 
            )
            vector = resp.data[0].embedding

            return {
                "vector_cache": vector,
                "vector_id": str(listing_id),
                "index_name": "default"
            }

        except Exception as e:
            log.warning(f"OpenAI Embedding failed (attempt {attempt+1}): {e}")
            time.sleep(1)
            
    raise Exception("Failed to generate embedding.")