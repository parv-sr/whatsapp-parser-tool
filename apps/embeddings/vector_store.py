import asyncio
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Sequence
from urllib.parse import quote_plus

from django.conf import settings

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

log = logging.getLogger(__name__)

VECTOR_COLLECTION_NAME = getattr(settings, "VECTOR_COLLECTION_NAME", "property_embeddings")
EMBEDDING_MODEL = getattr(settings, "EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(getattr(settings, "EMBEDDING_DIMENSIONS", 1536))
VECTOR_INSERT_BATCH_SIZE = int(getattr(settings, "VECTOR_INSERT_BATCH_SIZE", 256))


def get_embeddings_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=getattr(settings, "OPENAI_API_KEY", None),
        dimensions=EMBEDDING_DIMENSIONS,
    )


def _postgres_connection_url() -> str:
    vector_db_url = os.getenv("VECTOR_DB_URL") or getattr(settings, "VECTOR_DB_URL", None)
    if vector_db_url:
        return vector_db_url

    db = settings.DATABASES["default"]
    user = quote_plus(str(db.get("USER", "postgres")))
    password = quote_plus(str(db.get("PASSWORD", "")))
    host = db.get("HOST", "localhost")
    port = db.get("PORT", "5432")
    name = db.get("NAME", "postgres")

    base_url = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{name}"

    options = db.get("OPTIONS") or {}
    sslmode = options.get("sslmode")
    if sslmode:
        return f"{base_url}?sslmode={sslmode}"
    return base_url


def _build_vectorstore(async_mode: bool = False) -> PGVector:
    return PGVector(
        embeddings=get_embeddings_model(),
        collection_name=VECTOR_COLLECTION_NAME,
        connection=_postgres_connection_url(),
        embedding_length=EMBEDDING_DIMENSIONS,
        use_jsonb=True,
        async_mode=async_mode,
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> PGVector:
    vectorstore = _build_vectorstore(async_mode=False)
    vectorstore.create_tables_if_not_exists()
    return vectorstore


async def get_vectorstore_async() -> PGVector:
    vectorstore = _build_vectorstore(async_mode=True)
    if hasattr(vectorstore, "acreate_tables_if_not_exists"):
        await vectorstore.acreate_tables_if_not_exists()
    else:
        await asyncio.to_thread(vectorstore.create_tables_if_not_exists)
    return vectorstore


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


def _prepare_embedding_payload(records: Sequence[Dict[str, Any]]) -> tuple[List[str], List[List[float]], List[Dict[str, Any]], List[str]]:
    texts: List[str] = []
    embeddings: List[List[float]] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for row in records:
        vector = row.get("vector")
        listing_id = row.get("listing_chunk_id")
        if not vector or listing_id is None:
            continue

        metadata = _safe_metadata(row.get("metadata") or {})
        metadata["listing_chunk_id"] = int(listing_id)
        metadata.setdefault("transaction_type", metadata.get("transaction_type", "UNKNOWN"))
        metadata.setdefault("property_type", metadata.get("property_type", "UNKNOWN"))

        texts.append(row.get("text", ""))
        embeddings.append([float(x) for x in vector])
        metadatas.append(metadata)
        ids.append(str(listing_id))

    return texts, embeddings, metadatas, ids


def upsert_listing_embeddings(records: Sequence[Dict[str, Any]]) -> None:
    if not records:
        return

    vectorstore = get_vectorstore()
    texts, embeddings, metadatas, ids = _prepare_embedding_payload(records)

    if not ids:
        return

    for start in range(0, len(ids), VECTOR_INSERT_BATCH_SIZE):
        end = start + VECTOR_INSERT_BATCH_SIZE
        vectorstore.add_embeddings(
            texts=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )

    log.info("Upserted %s vectors into pgvector collection=%s", len(ids), VECTOR_COLLECTION_NAME)


async def aupsert_listing_embeddings(records: Sequence[Dict[str, Any]]) -> None:
    if not records:
        return

    vectorstore = await get_vectorstore_async()
    texts, embeddings, metadatas, ids = _prepare_embedding_payload(records)

    if not ids:
        return

    if hasattr(vectorstore, "aadd_embeddings"):
        for start in range(0, len(ids), VECTOR_INSERT_BATCH_SIZE):
            end = start + VECTOR_INSERT_BATCH_SIZE
            await vectorstore.aadd_embeddings(
                texts=texts[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )
    else:
        await asyncio.to_thread(upsert_listing_embeddings, records)


def build_pg_filter(filters: Dict[str, Any]) -> Dict[str, Any]:
    pg_filter: Dict[str, Any] = {}
    for key, value in (filters or {}).items():
        if value in (None, ""):
            continue
        pg_filter[key] = {"$eq": value}
    return pg_filter
