import logging
from typing import Any, Dict, Iterable, List, Sequence

from django.conf import settings

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

log = logging.getLogger(__name__)

QDRANT_COLLECTION = getattr(settings, "QDRANT_COLLECTION", "listing_chunks")
EMBEDDING_MODEL = getattr(settings, "EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(getattr(settings, "EMBEDDING_DIMENSIONS", 1536))


def get_embeddings_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=getattr(settings, "OPENAI_API_KEY", None),
        dimensions=EMBEDDING_DIMENSIONS,
    )


def get_qdrant_client() -> QdrantClient:
    url = getattr(settings, "QDRANT_URL", None)
    api_key = getattr(settings, "QDRANT_API_KEY", None)

    if url:
        return QdrantClient(url=url, api_key=api_key, timeout=10.0)

    path = getattr(settings, "QDRANT_LOCAL_PATH", ":memory:")
    return QdrantClient(path=path)


def _ensure_collection(client: QdrantClient) -> None:
    try:
        exists = client.collection_exists(QDRANT_COLLECTION)
    except Exception:
        exists = False

    if exists:
        return

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=qmodels.VectorParams(size=EMBEDDING_DIMENSIONS, distance=qmodels.Distance.COSINE),
    )


def get_qdrant_vectorstore() -> QdrantVectorStore:
    client = get_qdrant_client()
    _ensure_collection(client)
    return QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=get_embeddings_model(),
        content_payload_key="page_content",
        metadata_payload_key="metadata",
    )


def _safe_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, list):
            clean[key] = [str(v) for v in value if v is not None]
        elif isinstance(value, dict):
            clean[key] = {str(k): str(v) for k, v in value.items() if v is not None}
        else:
            clean[key] = str(value)
    return clean


def upsert_listing_embeddings(records: Sequence[Dict[str, Any]]) -> None:
    if not records:
        return

    client = get_qdrant_client()
    _ensure_collection(client)

    points: List[qmodels.PointStruct] = []
    for row in records:
        vector = row.get("vector")
        listing_id = row.get("listing_chunk_id")
        if not vector or listing_id is None:
            continue

        metadata = _safe_metadata(row.get("metadata") or {})
        payload = {
            "page_content": row.get("text", ""),
            "metadata": {
                **metadata,
                "listing_chunk_id": int(listing_id),
            },
            "listing_chunk_id": int(listing_id),
            "transaction_type": metadata.get("transaction_type", "UNKNOWN"),
            "property_type": metadata.get("property_type", "UNKNOWN"),
        }
        points.append(
            qmodels.PointStruct(
                id=int(listing_id),
                vector=[float(x) for x in vector],
                payload=payload,
            )
        )

    if not points:
        return

    client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=False)
    log.info("Upserted %s vectors into qdrant collection=%s", len(points), QDRANT_COLLECTION)


def build_qdrant_filter(filters: Dict[str, Any]) -> qmodels.Filter | None:
    conditions: List[qmodels.FieldCondition] = []
    for key, value in (filters or {}).items():
        if value in (None, ""):
            continue
        conditions.append(
            qmodels.FieldCondition(
                key=key,
                match=qmodels.MatchValue(value=value),
            )
        )

    if not conditions:
        return None
    return qmodels.Filter(must=conditions)
