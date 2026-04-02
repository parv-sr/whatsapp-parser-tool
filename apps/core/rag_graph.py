import asyncio
import hashlib
import json
import logging
import re
from collections import defaultdict
from contextlib import asynccontextmanager
from functools import lru_cache
import sys
from typing import Annotated, Any, AsyncIterator, Dict, List, Optional, TypedDict

from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.cache import cache
from django.db import connection
from django.db import connections
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from apps.core.tools import search_property_listings
from apps.embeddings.vector_store import build_pg_filter, get_vectorstore, get_vectorstore_async
from apps.preprocessing.models import ListingChunk

log = logging.getLogger(__name__)

DEFAULT_TOP_K = 10
CHAT_MODEL = getattr(settings, "CHAT_MODEL", "gpt-4o-mini")
ALLOWED_CHAT_MODELS = getattr(settings, "OPENAI_CHAT_MODELS", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
ENABLE_LLM_GRADING = bool(getattr(settings, "RAG_ENABLE_LLM_GRADING", True))

REAL_ESTATE_TERMS = {
    "office", "shop", "villa", "bhk", "sqft", "carpet", "price", "budget", "listing", "ownership", "commercial",
    "residential", "tenant", "furnish", "deposit", "location", "area", "tower", "building", "layout", "apartment",
    "flat", "lease", "rent", "sale", "property",
}
OWNERSHIP_TERMS = {"buy", "sale", "sell", "ownership", "own", "purchase", "resale"}
LEASE_TERMS = {"lease", "leave and license", "l&l", "license"}
RENT_TERMS = {"rent", "rental", "tenant", "let out"}
STOPWORDS = {"the", "in", "at", "for", "with", "and", "or", "to", "of", "a", "an", "near"}
MUST_HAVE_FEATURE_TERMS = {
    "parking", "lift", "elevator", "furnished", "semi-furnished", "unfurnished", "balcony", "terrace",
    "pool", "gym", "garden", "corner", "vaastu", "vastu", "pet-friendly", "pet", "servant", "duplex",
}


class RealEstateClassifier(BaseModel):
    is_real_estate_query: bool = Field(description="True when user asks about property/listing/rent/sale/lease intent.")
    reason: str = Field(default="")


class QueryRewrite(BaseModel):
    rewritten_query: str = Field(description="Expanded and normalized query for retrieval")


class DocumentGrade(BaseModel):
    score: int = Field(ge=1, le=10)
    reason: str = Field(default="")


class FinalAnswer(BaseModel):
    answer: str
    has_relevant_data: bool
    sources: List[int]


class RAGState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    model: str
    temperature: float
    thread_id: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float


def _to_plain_data(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {k: _to_plain_data(v) for k, v in value.model_dump().items()}
    if isinstance(value, dict):
        return {k: _to_plain_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain_data(v) for v in value]
    return value


async def _ainvoke_structured_output_plain(prompt: Any, llm: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    result = await (prompt | llm).ainvoke(payload)
    plain = _to_plain_data(result)
    return plain if isinstance(plain, dict) else {}


def _escape_html(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _build_html_table_rows(rows: List[Dict[str, Any]]) -> str:
    rendered: List[str] = []
    for item in rows:
        meta = _with_metadata_defaults(item.get("metadata") or {})
        rendered.append(
            "<tr>"
            f"<td>{_escape_html(item.get('id', ''))}</td>"
            f"<td>{_escape_html(meta.get('transaction_type', 'Not specified'))}</td>"
            f"<td>{_escape_html(meta.get('property_type', 'Not specified'))}</td>"
            f"<td>{_escape_html(meta.get('location', 'Not specified'))}</td>"
            f"<td>{_escape_html(meta.get('building_name', 'Not specified'))}</td>"
            f"<td>{_escape_html(meta.get('bhk', 'Not specified'))}</td>"
            f"<td>{_escape_html(meta.get('sqft', 'Not specified'))}</td>"
            f"<td>{_escape_html(meta.get('price', 'Not specified'))}</td>"
            "</tr>"
        )
    return "".join(rendered)


def _compose_answer_html(state: RAGState, final_text: str, source_ids: List[int]) -> str:
    contexts = state.get("graded_contexts") or state.get("contexts") or []
    source_set = set(source_ids or [])
    rows = [ctx for ctx in contexts if ctx.get("id") in source_set]
    if not rows:
        rows = contexts[: min(8, len(contexts))]
    rows = rows[:8]

    top_matches = []
    for idx, row in enumerate(rows[:3], start=1):
        meta = _with_metadata_defaults(row.get("metadata") or {})
        top_matches.append(
            f"<li><strong>Listing ID: {_escape_html(row.get('id'))}</strong> - "
            f"{_escape_html(meta.get('bhk', 'Not specified'))}, "
            f"{_escape_html(meta.get('location', 'Not specified'))}, "
            f"{_escape_html(meta.get('transaction_type', 'Not specified'))}.</li>"
        )

    intro_text = final_text or "Here are the best matches I found in your uploaded WhatsApp listings."
    table_html = (
        "<h4>Details</h4>"
        "<table><thead><tr>"
        "<th>ID</th><th>Transaction</th><th>Property</th><th>Location</th>"
        "<th>Building</th><th>BHK</th><th>SQFT</th><th>Price</th>"
        "</tr></thead><tbody>"
        f"{_build_html_table_rows(rows)}"
        "</tbody></table>"
    )
    if not rows:
        table_html = "<p><em>No listing rows available yet. Try broadening your query.</em></p>"

    top_html = f"<h4>Top Matches</h4><ol>{''.join(top_matches)}</ol>" if top_matches else ""
    return f"<p>{_escape_html(intro_text)}</p>{top_html}{table_html}"


def _escape_html(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _build_html_table_rows(rows: List[Dict[str, Any]]) -> str:
    rendered: List[str] = []
    for item in rows:
        meta = _with_metadata_defaults(item.get("metadata") or {})
        rendered.append(
            "<tr>"
            f"<td>{_escape_html(item.get('id', ''))}</td>"
            f"<td>{_escape_html(meta.get('transaction_type', 'Not specified'))}</td>"
            f"<td>{_escape_html(meta.get('property_type', 'Not specified'))}</td>"
            f"<td>{_escape_html(meta.get('location', 'Not specified'))}</td>"
            f"<td>{_escape_html(meta.get('building_name', 'Not specified'))}</td>"
            f"<td>{_escape_html(meta.get('bhk', 'Not specified'))}</td>"
            f"<td>{_escape_html(meta.get('sqft', 'Not specified'))}</td>"
            f"<td>{_escape_html(meta.get('price', 'Not specified'))}</td>"
            "</tr>"
        )
    return "".join(rendered)


def _compose_answer_html(state: RAGState, final_text: str, source_ids: List[int]) -> str:
    contexts = state.get("graded_contexts") or state.get("contexts") or []
    source_set = set(source_ids or [])
    preferred_rows = [ctx for ctx in contexts if ctx.get("id") in source_set]
    scored_rows = [ctx for ctx in contexts if (ctx.get("relevance_score") or 0) >= 6]
    dynamic_limit = min(10, max(3, len(scored_rows) or len(preferred_rows) or len(contexts)))

    rows = preferred_rows
    if not rows:
        rows = scored_rows or contexts
    rows = rows[:dynamic_limit]

    top_matches = []
    for idx, row in enumerate(rows[: min(5, len(rows))], start=1):
        meta = _with_metadata_defaults(row.get("metadata") or {})
        top_matches.append(
            f"<li><strong>Listing ID: {_escape_html(row.get('id'))}</strong> - "
            f"{_escape_html(meta.get('bhk', 'Not specified'))}, "
            f"{_escape_html(meta.get('location', 'Not specified'))}, "
            f"{_escape_html(meta.get('transaction_type', 'Not specified'))}.</li>"
        )

    intro_text = final_text or "Here are the best matches I found in your uploaded WhatsApp listings."
    table_html = (
        "<h4>Details</h4>"
        "<table><thead><tr>"
        "<th>ID</th><th>Transaction</th><th>Property</th><th>Location</th>"
        "<th>Building</th><th>BHK</th><th>SQFT</th><th>Price</th>"
        "</tr></thead><tbody>"
        f"{_build_html_table_rows(rows)}"
        "</tbody></table>"
    )
    if not rows:
        table_html = "<p><em>No listing rows available yet. Try broadening your query.</em></p>"

    top_html = f"<h4>Top Matches</h4><ol>{''.join(top_matches)}</ol>" if top_matches else ""
    return f"<p>{_escape_html(intro_text)}</p>{top_html}{table_html}"


def _build_messages(query: str, contexts: List[Dict[str, Any]], memory: str = "") -> List[Dict[str, str]]:
    """
    Backward-compatible helper kept for tests and diagnostics.
    """
    snippet_blocks = []
    for idx, item in enumerate(contexts, start=1):
        meta = dict(item.get("metadata", {}))
        meta["_id"] = item.get("id")
        meta["_rank_score"] = round(float(item.get("hybrid_score", 0.0)), 6)
        snippet_blocks.append(f"[SNIPPET {idx}]\n{json.dumps(meta, ensure_ascii=False)}\n[/SNIPPET {idx}]")

    system_prompt = (
        "You are a helpful real-estate assistant. Answer based on the provided context from uploaded WhatsApp chats. "
        "If no exact match is found, be helpful and show the closest available listings instead of refusing."
    )
    messages = [{"role": "system", "content": system_prompt}]
    if memory:
        messages.append({"role": "system", "content": f"Conversation history:\n{memory}"})
    messages.append({"role": "user", "content": f"Query: {query}\n\nContext snippets:\n" + "\n".join(snippet_blocks)})
    return messages


def _cache_key(prefix: str, payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"rag:{prefix}:{digest}"


async def _aget_or_set_cache(cache_key: str, ttl_seconds: int, compute_fn):
    cached = await sync_to_async(cache.get)(cache_key)
    if cached is not None:
        return cached
    value = await compute_fn()
    await sync_to_async(cache.set)(cache_key, value, timeout=ttl_seconds)
    return value


def _safe_model(model: Optional[str]) -> str:
    return model if model in ALLOWED_CHAT_MODELS else CHAT_MODEL


def _chat_llm(state: RAGState, temperature: Optional[float] = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=_safe_model(state.get("model")),
        api_key=getattr(settings, "OPENAI_API_KEY", None),
        temperature=float(state.get("temperature", 0.0) if temperature is None else temperature),
        max_tokens=900,
    )


def _is_domain_query(query: str) -> bool:
    if any(ch.isdigit() for ch in query or ""):
        return True
    tokens = [t for t in re.split(r"\W+", (query or "").lower()) if t]
    if len(tokens) < 2:
        return False
    matches = sum(1 for t in tokens if t in REAL_ESTATE_TERMS)
    return matches > 0


def _with_metadata_defaults(metadata: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "listing_type": "UNKNOWN",
        "transaction_type": "UNKNOWN",
        "property_type": "UNKNOWN",
        "location": "Not specified",
        "building_name": "Not specified",
        "furnishing": "Not specified",
        "price": "Not specified",
        "sqft": "Not specified",
        "bhk": "Not specified",
        "parking": "Not specified",
        "contact_numbers": [],
        "features": [],
    }
    normalized = dict(defaults)
    normalized.update(metadata or {})
    for key, value in list(normalized.items()):
        if value in [None, "", [], {}] and key not in {"contact_numbers", "features"}:
            normalized[key] = "Not specified"
    return normalized




def _normalize_real_estate_shorthand(query: str) -> str:
    normalized = (query or "").strip()
    if not normalized:
        return ""

    substitutions = [
        (r"\bbkc\b", "bandra kurla complex mumbai"),
        (r"\bandheri\s*w\b", "andheri west mumbai"),
        (r"\blokhandwala\b", "lokhandwala andheri west mumbai"),
        (r"\b([123])\s*[-/]?\s*bhk\b", r"\1 bhk apartment"),
        (r"\bl&l\b", "leave and license lease"),
    ]
    for pattern, replacement in substitutions:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    lowered = normalized.lower()
    if "lease" in lowered and "rent" not in lowered:
        normalized = f"{normalized} for lease"
    elif "rent" in lowered and "lease" not in lowered:
        normalized = f"{normalized} for rent"

    return re.sub(r"\s+", " ", normalized).strip()


def _extract_hard_constraints(query: str) -> List[str]:
    lowered = (query or "").lower()
    constraints: List[str] = []
    if "balcony" in lowered:
        constraints.append("must have balcony")
    if "attached bath" in lowered or "attached bathroom" in lowered:
        constraints.append("attached bath")
    if "road view" in lowered:
        constraints.append("road view")
    return constraints


def _ensure_constraints_in_rewrite(original_query: str, rewritten_query: str) -> str:
    final_query = (rewritten_query or "").strip()
    missing = [c for c in _extract_hard_constraints(original_query) if c not in final_query.lower()]
    if missing:
        suffix = ", ".join(missing)
        final_query = f"{final_query}; {suffix}" if final_query else suffix
    return final_query.strip()
def _extract_filters(query: str) -> Dict[str, Any]:
    lowered = (query or "").lower()
    filters: Dict[str, Any] = {}
    if any(term in lowered for term in {"buy", "purchase", "want", "sale"}):
        filters["transaction_type"] = "SALE"
    elif any(term in lowered for term in {"rent", "lease", "tenant"}):
        filters["transaction_type"] = "RENT"

    if any(term in lowered for term in {"flat", "apartment", "villa", "bhk"}):
        filters["property_type"] = "RESIDENTIAL"
    elif any(term in lowered for term in {"office", "shop", "commercial"}):
        filters["property_type"] = "COMMERCIAL"
    return filters


def _extract_query_preferences(query: str) -> Dict[str, str]:
    lowered = (query or "").lower()
    prefs: Dict[str, str] = {}
    if any(t in lowered for t in LEASE_TERMS):
        prefs["transaction_type"] = "LEASE"
    elif any(t in lowered for t in OWNERSHIP_TERMS):
        prefs["transaction_type"] = "SALE"
    elif any(t in lowered for t in RENT_TERMS):
        prefs["transaction_type"] = "RENT"

    if any(t in lowered for t in {"office", "commercial", "shop"}):
        prefs["property_type"] = "COMMERCIAL"
    elif any(t in lowered for t in {"flat", "apartment", "villa", "residential", "bhk"}):
        prefs["property_type"] = "RESIDENTIAL"
    return prefs


def _query_tokens(query: str) -> List[str]:
    return [
        token for token in re.split(r"\W+", (query or "").lower())
        if len(token) > 2 and token not in STOPWORDS
    ]


def _extract_must_have_terms(query: str) -> set[str]:
    tokens = set(_query_tokens(query))
    return {token for token in tokens if token in MUST_HAVE_FEATURE_TERMS}


def _load_listing_briefs(listing_ids: List[int]) -> Dict[int, ListingChunk]:
    return {
        l.id: l
        for l in ListingChunk.objects.filter(id__in=listing_ids).only(
            "id", "transaction_type", "category", "metadata"
        )
    }


def _deterministic_rerank(
    query: str,
    fused_scores: Dict[int, float],
    rank_data: Dict[int, Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    listing_ids = list(fused_scores.keys())
    listing_map = _load_listing_briefs(listing_ids)
    prefs = _extract_query_preferences(query)
    query_token_set = set(_query_tokens(query))
    must_have_terms = _extract_must_have_terms(query)

    scored_rows: List[Dict[str, Any]] = []
    for lid in listing_ids:
        listing = listing_map.get(lid)
        if not listing:
            continue

        metadata = _with_metadata_defaults(listing.metadata or {})
        location_tokens = set(_query_tokens(str(metadata.get("location", ""))))
        feature_values = " ".join(
            [
                str(metadata.get("furnishing", "")),
                str(metadata.get("parking", "")),
                str(metadata.get("building_name", "")),
                " ".join(str(item) for item in (metadata.get("features") or [])),
            ]
        ).lower()

        transaction_match = 1.0 if prefs.get("transaction_type") and listing.transaction_type == prefs["transaction_type"] else 0.0
        property_match = 1.0 if prefs.get("property_type") and listing.category == prefs["property_type"] else 0.0
        token_overlap = (len(query_token_set & location_tokens) / max(1, len(query_token_set))) if query_token_set else 0.0
        must_have_match = (
            sum(1 for term in must_have_terms if term in feature_values) / max(1, len(must_have_terms))
            if must_have_terms else 0.0
        )

        weighted = (
            transaction_match * 0.35
            + property_match * 0.25
            + token_overlap * 0.25
            + must_have_match * 0.15
        )
        deterministic_score = weighted + (float(fused_scores.get(lid, 0.0)) * 0.20)
        scored_rows.append(
            {
                "listing_chunk_id": lid,
                **rank_data.get(lid, {}),
                "hybrid_score": fused_scores.get(lid, 0.0),
                "deterministic_score": deterministic_score,
                "feature_scores": {
                    "transaction_match": transaction_match,
                    "property_match": property_match,
                    "location_overlap": token_overlap,
                    "must_have_match": must_have_match,
                },
            }
        )

    scored_rows.sort(
        key=lambda row: (
            row.get("deterministic_score", 0.0),
            row.get("hybrid_score", 0.0),
            row.get("listing_chunk_id", 0),
        ),
        reverse=True,
    )
    return scored_rows[:top_k]


def _fetch_top_k_by_vector(query: str, filters: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    vectorstore = get_vectorstore()
    pg_filter = build_pg_filter(filters)
    docs = vectorstore.similarity_search_with_score(query=query, k=top_k, filter=pg_filter or None)
    out: List[Dict[str, Any]] = []
    for doc, score in docs:
        meta = doc.metadata or {}
        lid = meta.get("listing_chunk_id") or (meta.get("metadata") or {}).get("listing_chunk_id")
        if lid is None:
            continue
        out.append({"listing_chunk_id": int(lid), "distance": float(score) if score is not None else None})
    return out


async def _afetch_top_k_by_vector(query: str, filters: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    cache_key = _cache_key("vec", {"q": query, "filters": filters, "k": top_k})

    async def _compute() -> List[Dict[str, Any]]:
        vectorstore = await get_vectorstore_async()
        pg_filter = build_pg_filter(filters)
        if hasattr(vectorstore, "asimilarity_search_with_score"):
            docs = await vectorstore.asimilarity_search_with_score(query=query, k=top_k, filter=pg_filter or None)
        else:
            docs = await asyncio.to_thread(vectorstore.similarity_search_with_score, query, top_k, pg_filter or None)
        out: List[Dict[str, Any]] = []
        for doc, score in docs:
            meta = doc.metadata or {}
            lid = meta.get("listing_chunk_id") or (meta.get("metadata") or {}).get("listing_chunk_id")
            if lid is None:
                continue
            out.append({"listing_chunk_id": int(lid), "distance": float(score) if score is not None else None})
        return out

    return await _aget_or_set_cache(cache_key, 120, _compute)


def _fetch_top_k_by_keyword(query: str, top_k: int) -> List[Dict[str, Any]]:
    terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
    if not terms:
        return []

    where_clause = " OR ".join(["LOWER(lc.text) LIKE %s OR LOWER(CAST(lc.metadata AS TEXT)) LIKE %s" for _ in terms])
    score_clause = " + ".join([
        "(CASE WHEN LOWER(lc.text) LIKE %s THEN 1 ELSE 0 END + CASE WHEN LOWER(CAST(lc.metadata AS TEXT)) LIKE %s THEN 1 ELSE 0 END)"
        for _ in terms
    ])

    params: List[Any] = []
    for t in terms:
        like = f"%{t}%"
        params.extend([like, like])

    sql = f"""
        SELECT lc.id, ({score_clause})::float AS keyword_score
        FROM preprocessing_listingchunk lc
        WHERE {where_clause}
        GROUP BY lc.id
        ORDER BY keyword_score DESC, lc.last_seen DESC
        LIMIT %s
    """

    for t in terms:
        like = f"%{t}%"
        params.extend([like, like])
    params.append(top_k)

    with connection.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [{"listing_chunk_id": r[0], "keyword_score": float(r[1])} for r in rows]


def _hybrid_retrieve(query: str, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    candidate_pool = max(48, top_k * 8)
    effective_filters = filters if filters is not None else _extract_filters(query)
    vec_hits = _fetch_top_k_by_vector(query, filters=effective_filters, top_k=candidate_pool)
    kw_hits = _fetch_top_k_by_keyword(query, top_k=candidate_pool)

    fused_scores: Dict[int, float] = defaultdict(float)
    rank_data: Dict[int, Dict[str, Any]] = {}
    max_kw = max((item.get("keyword_score", 0.0) for item in kw_hits), default=0.0)

    for rank, item in enumerate(vec_hits, start=1):
        lid = item.get("listing_chunk_id")
        if lid:
            fused_scores[lid] += 1.0 / (40 + rank)
            rank_data[lid] = {**rank_data.get(lid, {}), **item}

    for rank, item in enumerate(kw_hits, start=1):
        lid = item.get("listing_chunk_id")
        if lid:
            kw_score = float(item.get("keyword_score") or 0.0)
            normalized_kw = (kw_score / max_kw) if max_kw > 0 else 0.0
            fused_scores[lid] += normalized_kw * 0.3
            rank_data[lid] = {**rank_data.get(lid, {}), **item}

    return _deterministic_rerank(query=query, fused_scores=fused_scores, rank_data=rank_data, top_k=top_k)


async def _hybrid_retrieve_async(
    query: str,
    top_k: int,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    cache_key = _cache_key("hybrid", {"q": query, "k": top_k, "filters": filters or {}})

    async def _compute() -> List[Dict[str, Any]]:
        return await asyncio.to_thread(_hybrid_retrieve, query, top_k, filters)

    return await _aget_or_set_cache(cache_key, 90, _compute)


def _load_contexts(nearest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    listing_ids = [item["listing_chunk_id"] for item in nearest if item.get("listing_chunk_id")]
    listing_map = {l.id: l for l in ListingChunk.objects.filter(id__in=listing_ids)}
    contexts: List[Dict[str, Any]] = []
    for item in nearest:
        lid = item.get("listing_chunk_id")
        listing = listing_map.get(lid)
        if not listing:
            continue
        contexts.append(
            {
                "id": listing.id,
                "metadata": _with_metadata_defaults(listing.metadata or {}),
                "distance": item.get("distance"),
                "keyword_score": item.get("keyword_score"),
                "hybrid_score": item.get("hybrid_score", 0.0),
            }
        )
    return contexts


async def _load_contexts_async(nearest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(_load_contexts, nearest)


async def _classify_query_node(state: RAGState) -> RAGState:
    query = state.get("query", "").strip()
    if _is_domain_query(query):
        return {"is_real_estate_query": True, "reject_reason": "heuristic"}

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Classify if the query is about real-estate search, listing advice, rent/sale/lease, pricing, area, or property features."),
            ("human", "Query: {query}"),
        ]
    )
    try:
        llm = _chat_llm(state, temperature=0.0).with_structured_output(RealEstateClassifier)
        result = await _ainvoke_structured_output_plain(prompt, llm, {"query": query})
        return {
            "is_real_estate_query": bool(result.get("is_real_estate_query")),
            "reject_reason": str(result.get("reason") or ""),
        }
    except Exception as exc:
        log.warning("Classifier LLM failed; falling back to heuristic classifier: %s", exc)
        return {"is_real_estate_query": _is_domain_query(query), "reject_reason": "heuristic_fallback"}


async def _rewrite_query_node(state: RAGState) -> RAGState:
    query = state.get("query", "")
    normalized_query = _normalize_real_estate_shorthand(query) or query
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the query for real-estate retrieval. Keep output concise and retrieval-oriented. "
                "Preserve user hard constraints exactly (for example: must-have balcony, attached bath, road view). "
                "Few-shot transformations: "
                "BKC -> bandra kurla complex mumbai; Andheri W -> andheri west mumbai; "
                "Lokhandwala -> lokhandwala andheri west mumbai; "
                "1BHK/1 bhk/1-bhk -> 1 bhk apartment; 2BHK/2 bhk/2-bhk -> 2 bhk apartment; "
                "3BHK/3 bhk/3-bhk -> 3 bhk apartment; "
                "rent intent -> for rent; lease/L&L intent -> for lease. Return concise text only.",
            ),
            ("human", "Original query: {query}"),
        ]
    )
    try:
        llm = _chat_llm(state, temperature=0.0).with_structured_output(QueryRewrite)
        rewritten = await (prompt | llm).ainvoke({"query": normalized_query})
        rewritten_query = _ensure_constraints_in_rewrite(query, rewritten.rewritten_query.strip() or normalized_query)
        return {"rewritten_query": rewritten_query}
    except Exception as exc:
        log.warning("Rewrite LLM failed; using normalized query: %s", exc)
        return {"rewritten_query": _ensure_constraints_in_rewrite(query, normalized_query)}


AGENT_SYSTEM_PROMPT = (
    "You are an intelligent real estate agent. "
    "You MUST use the search_property_listings tool to find properties. "
    "For follow-up questions, combine the user's new request with their previous constraints from the conversation history "
    "and pass the COMBINED constraints to the tool. "
    "Review the tool results carefully before answering. "
    "Output a markdown table containing between 3 and 10 of the best matches. "
    "Below the table, briefly explain exactly WHY you selected these listings based on the user's specific constraints."
)


async def _agent_node(state: RAGState) -> RAGState:
    llm = _chat_llm(state).bind_tools([search_property_listings])
    messages = state.get("messages") or []
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT), *messages]
    response = await llm.ainvoke(messages)
    return {"messages": [response]}


async def _grade_documents_node(state: RAGState) -> RAGState:
    contexts = state.get("contexts", [])
    query = state.get("rewritten_query") or state.get("query", "")
    if not contexts:
        return {"graded_contexts": []}

    llm_grading_enabled = bool(state.get("use_llm_grading", ENABLE_LLM_GRADING))
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Score relevance of a listing chunk to a real-estate query from 1-10."),
            ("human", "Query: {query}\nListing JSON: {listing_json}"),
        ]
    )
    llm = _chat_llm(state, temperature=0.0).with_structured_output(DocumentGrade) if llm_grading_enabled else None

    async def _grade_one(ctx: Dict[str, Any]) -> Dict[str, Any]:
        listing_json = json.dumps(ctx.get("metadata", {}), ensure_ascii=False)
        merged = dict(ctx)
        deterministic_score = float(merged.get("deterministic_score", merged.get("hybrid_score", 0.0)))
        deterministic_grade = max(1, min(10, int(round(deterministic_score * 10))))
        merged["deterministic_relevance_score"] = deterministic_grade
        try:
            if llm:
                grade = await (prompt | llm).ainvoke({"query": query, "listing_json": listing_json})
                llm_score = int(grade.score)
                merged["llm_relevance_score"] = llm_score
                merged["relevance_score"] = max(1, min(10, int(round(deterministic_grade * 0.7 + llm_score * 0.3))))
                merged["relevance_reason"] = grade.reason or "llm_plus_deterministic"
            else:
                merged["relevance_score"] = deterministic_grade
                merged["relevance_reason"] = "deterministic_only"
        except Exception:
            merged["relevance_score"] = deterministic_grade
            merged["relevance_reason"] = "deterministic_fallback"
        return merged

    graded = await asyncio.gather(*[_grade_one(ctx) for ctx in contexts])
    graded.sort(
        key=lambda x: (
            x.get("relevance_score", 0),
            x.get("deterministic_score", x.get("hybrid_score", 0.0)),
            x.get("hybrid_score", 0.0),
        ),
        reverse=True,
    )
    return {"graded_contexts": graded}


def _route_after_classify(state: RAGState) -> str:
    return "rewrite_query" if state.get("is_real_estate_query") else "fallback"


def _route_after_grading(state: RAGState) -> str:
    graded = state.get("graded_contexts", [])
    if any((ctx.get("relevance_score") or 0) >= 7 for ctx in graded):
        return "generate"
    return "fallback"


async def _generate_node(state: RAGState) -> RAGState:
    query = state.get("query", "")
    prior_messages = state.get("messages") or []
    memory_lines: List[str] = []
    for message in prior_messages[:-1]:
        role = getattr(message, "type", "system").upper()
        content = getattr(message, "content", "")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
            )
        memory_lines.append(f"{role}: {content}")
    memory = "\n".join(memory_lines) or state.get("memory", "")
    graded_contexts = [ctx for ctx in state.get("graded_contexts", []) if (ctx.get("relevance_score") or 0) >= 7]
    snippet_blocks = []
    for idx, item in enumerate(graded_contexts, start=1):
        meta = dict(item.get("metadata", {}))
        meta["_id"] = item.get("id")
        meta["_score"] = item.get("relevance_score", 0)
        snippet_blocks.append(f"[SNIPPET {idx}]\\n{json.dumps(meta, ensure_ascii=False)}\\n[/SNIPPET {idx}]")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful real-estate assistant. Answer based on the provided context from uploaded WhatsApp chats. "
                "If no exact match is found, be helpful and show the closest available listings instead of refusing.",
            ),
            ("system", "Conversation history:\n{memory}"),
            ("human", "User query: {query}\n\nContext snippets:\n{snippets}"),
        ]
    )
    try:
        llm = _chat_llm(state).with_structured_output(FinalAnswer)
        result = await _ainvoke_structured_output_plain(
            prompt,
            llm,
            {"query": query, "memory": memory or "None", "snippets": "\n".join(snippet_blocks)},
        )
        return {"final": result}
    except Exception as exc:
        log.warning("Generate LLM failed; returning deterministic fallback summary: %s", exc)
        top_ids = [int(item.get("id")) for item in graded_contexts[:3] if item.get("id")]
        if top_ids:
            answer = "I found relevant listings from your uploaded chats. Top matches: " + ", ".join(f"#{lid}" for lid in top_ids)
        else:
            answer = "I couldn't find enough listing evidence to answer that reliably."
        return {"final": {"answer": answer, "has_relevant_data": bool(top_ids), "sources": top_ids}}


async def _fallback_node(state: RAGState) -> RAGState:
    graded = state.get("graded_contexts") or state.get("contexts") or []
    top = graded[:3]
    previews = []
    source_ids: List[int] = []
    for item in top:
        meta = item.get("metadata", {})
        source_ids.append(int(item.get("id")))
        previews.append(
            f"- #{item.get('id')} | {meta.get('location', 'Not specified')} | "
            f"{meta.get('bhk', 'Not specified')} | {meta.get('price', 'Not specified')}"
        )

    if not state.get("is_real_estate_query", True):
        answer = (
            "I can help with real-estate queries based on your uploaded WhatsApp chats. "
            "Please ask about locations, budgets, BHK, rent/sale/lease, or listing features."
        )
        return {"final": {"answer": answer, "has_relevant_data": False, "sources": []}}

    if previews:
        answer = (
            "I couldn't find an exact match in your uploaded chats, but here are the closest listings I have:\n"
            + "\n".join(previews)
        )
    else:
        answer = "I couldn't find an exact match in your uploaded chats yet. Try adding area, budget, and property type for better matches."

    return {"final": {"answer": answer, "has_relevant_data": False, "sources": source_ids}}


async def _format_output_node(state: RAGState) -> RAGState:
    final = state.get("final") if isinstance(state.get("final"), dict) else {}

    if not final:
        graded = state.get("graded_contexts") or []
        fallback_ids = [int(ctx.get("id")) for ctx in graded[:3] if str(ctx.get("id", "")).isdigit()]
        if fallback_ids:
            final = {
                "answer": "I couldn't find enough listing evidence to answer that reliably.",
                "sources": fallback_ids,
                "confidence": 0.0,
            }
        else:
            final = {
                "answer": "I couldn't find enough listing evidence to answer that reliably.",
                "sources": [],
                "confidence": 0.0,
            }

    answer_text = (final.get("answer") or "I couldn't find enough listing evidence to answer that reliably.").strip()
    raw_sources = final.get("sources")
    source_ids: List[int] = []
    if isinstance(raw_sources, list):
        for item in raw_sources:
            candidate = item.get("id") if isinstance(item, dict) else item
            if str(candidate).isdigit():
                source_ids.append(int(candidate))

    contexts = state.get("graded_contexts") or state.get("contexts") or []
    source_set = set(source_ids)
    sources = []
    for ctx in contexts:
        if ctx.get("id") not in source_set:
            continue
        normalized = dict(ctx)
        normalized["id"] = int(ctx.get("id"))
        normalized["metadata"] = _with_metadata_defaults(ctx.get("metadata") or {})
        sources.append(normalized)

    answer = _compose_answer_html(state, answer_text, source_ids)
    output: RAGState = {"answer": answer, "sources": sources, "messages": [AIMessage(content=answer_text)]}

    model = final.get("model") or state.get("model")
    if isinstance(model, str) and model:
        output["model"] = _safe_model(model)

    confidence = final.get("confidence")
    if isinstance(confidence, (int, float)):
        output["confidence"] = float(confidence)

    return output


@lru_cache(maxsize=1)
def _checkpoint_dsn() -> Optional[str]:
    db = connections["default"].settings_dict
    if db.get("ENGINE", "").endswith("sqlite3"):
        return None
    return (
        getattr(settings, "LANGGRAPH_CHECKPOINT_DSN", None)
        or db.get("OPTIONS", {}).get("dsn")
        or f"postgresql://{db.get('USER')}:{db.get('PASSWORD')}@{db.get('HOST') or 'localhost'}:{db.get('PORT') or 5432}/{db.get('NAME')}"
    )


@lru_cache(maxsize=1)
def _checkpoint_backend_config() -> Dict[str, str]:
    dsn = _checkpoint_dsn()
    backend = "none"
    reason = "No checkpoint DSN configured."

    if not dsn:
        pass
    elif sys.platform == "win32":
        backend = "memory"
        reason = "Windows runtime detected; skipping async PostgresSaver and using in-memory saver."
    else:
        backend = "postgres"
        reason = "Using async PostgresSaver backend."

    log.info("Checkpoint backend selected: %s (%s)", backend, reason)
    return {"backend": backend, "reason": reason}


@asynccontextmanager
async def _checkpointer_context():
    config = _checkpoint_backend_config()
    backend = config["backend"]

    if backend == "none":
        yield None
        return

    if backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        yield MemorySaver()
        return

    dsn = _checkpoint_dsn()
    if not dsn:
        yield None
        return

    if sys.platform == "win32" and isinstance(asyncio.get_running_loop(), asyncio.ProactorEventLoop):
        from langgraph.checkpoint.memory import MemorySaver

        yield MemorySaver()
        return

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ModuleNotFoundError:
        log.info("LangGraph postgres checkpoint package unavailable; using in-memory saver.")
        from langgraph.checkpoint.memory import MemorySaver

        yield MemorySaver()
        return

    try:
        async with AsyncPostgresSaver.from_conn_string(dsn) as checkpointer:
            yield checkpointer
    except Exception as exc:
        log.warning("Falling back to in-memory checkpointer after PostgresSaver failure: %s", exc)
        from langgraph.checkpoint.memory import MemorySaver

        yield MemorySaver()


def _build_graph_definition():
    graph = StateGraph(RAGState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", ToolNode([search_property_listings]))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph


GRAPH_DEFINITION = _build_graph_definition()
RAG_WORKFLOW = GRAPH_DEFINITION.compile()


def _extract_tool_contexts(messages: List[Any]) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    for message in messages:
        msg_type = getattr(message, "type", None) or (message.get("type") if isinstance(message, dict) else None)
        if msg_type != "tool":
            continue
        content = getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")
        if not isinstance(content, str):
            continue
        try:
            parsed = json.loads(content)
        except Exception:
            continue
        if isinstance(parsed, list):
            contexts.extend([item for item in parsed if isinstance(item, dict)])
    return contexts


def _latest_ai_text(messages: List[Any]) -> str:
    for message in reversed(messages):
        msg_type = getattr(message, "type", None) or (message.get("type") if isinstance(message, dict) else None)
        if msg_type != "ai":
            continue

        tool_calls = getattr(message, "tool_calls", None) if not isinstance(message, dict) else message.get("tool_calls")
        content = getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")

        if isinstance(content, str):
            if tool_calls and not content.strip():
                continue
            if content.strip():
                return content
            continue

        if isinstance(content, list):
            text_content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
            ).strip()
            if tool_calls and not text_content:
                continue
            if text_content:
                return text_content
    return ""


async def run_rag(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    model: str = CHAT_MODEL,
    temperature: float = 0.0,
    memory: str = "",
    thread_id: str = "default",
) -> RAGState:
    state: RAGState = {
        "messages": [HumanMessage(content=query)],
        "model": model,
        "temperature": temperature,
        "thread_id": thread_id,
    }
    config = {"configurable": {"thread_id": thread_id}}
    async with _checkpointer_context() as checkpointer:
        workflow = GRAPH_DEFINITION.compile(checkpointer=checkpointer) if checkpointer else RAG_WORKFLOW
        result = _to_plain_data(await workflow.ainvoke(state, config=config))
        messages = result.get("messages") or []
        answer = _latest_ai_text(messages).strip()
        sources = _extract_tool_contexts(messages)
        result["answer"] = answer
        result["sources"] = sources
        return result


async def stream_rag_events(
    query: str,
    top_k: int,
    model: str,
    temperature: float,
    memory: str,
    thread_id: str,
) -> AsyncIterator[Dict[str, Any]]:
    state: RAGState = {
        "messages": [HumanMessage(content=query)],
        "model": model,
        "temperature": temperature,
        "thread_id": thread_id,
    }
    config = {"configurable": {"thread_id": thread_id}}

    async with _checkpointer_context() as checkpointer:
        workflow = GRAPH_DEFINITION.compile(checkpointer=checkpointer) if checkpointer else RAG_WORKFLOW
        final_state: Dict[str, Any] = {}
        emitted_tokens = False
        async for event in workflow.astream_events(state, config=config, version="v2"):
            if event.get("event") == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                delta = ""
                if chunk is not None:
                    delta = getattr(chunk, "content", "") or ""
                    if isinstance(delta, list):
                        delta = "".join(
                            part.get("text", "") for part in delta if isinstance(part, dict) and part.get("type") == "text"
                        )
                if delta:
                    emitted_tokens = True
                    yield {"type": "token", "delta": delta}
            if event.get("event") == "on_chain_end" and event.get("name") == "agent":
                maybe_state = event.get("data", {}).get("output") or {}
                if isinstance(maybe_state, dict):
                    final_state = _to_plain_data(maybe_state)

        messages = final_state.get("messages") or []
        answer = _latest_ai_text(messages).strip()
        sources = _extract_tool_contexts(messages)
        if answer and not emitted_tokens:
            yield {"type": "token", "delta": answer}
        yield {"type": "final", "answer": answer, "sources": sources, "model": _safe_model(model)}


async def get_recent_messages_text(user, limit: int = 6) -> str:
    def _load() -> str:
        msgs = ChatMessage.objects.filter(user=user).order_by("-timestamp")[:limit]
        msgs = list(reversed(msgs))
        return "\n".join(f"{m.role.upper()}: {m.content}" for m in msgs)

    from apps.core.models import ChatMessage

    return await sync_to_async(_load)()
