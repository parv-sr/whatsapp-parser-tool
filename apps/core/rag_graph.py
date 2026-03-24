import asyncio
import json
import hashlib
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, TypedDict

from django.conf import settings
from django.db import connection
from django.core.cache import cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from apps.embeddings.vector_store import build_pg_filter, get_vectorstore, get_vectorstore_async
from apps.preprocessing.models import ListingChunk

log = logging.getLogger(__name__)

DEFAULT_TOP_K = 8
REAL_ESTATE_TERMS = {
    "office", "shop", "villa", "bhk", "sqft", "carpet", "price", "budget", "listing", "ownership", "commercial",
    "residential", "tenant", "furnish", "deposit", "location", "area", "tower", "building", "layout",
}
OWNERSHIP_TERMS = {"buy", "sale", "sell", "ownership", "own", "purchase", "resale"}
LEASE_TERMS = {"lease", "leave and license", "l&l", "license"}
RENT_TERMS = {"rent", "rental", "tenant", "let out"}
CHAT_MODEL = getattr(settings, "CHAT_MODEL", "gpt-4o-mini")
ALLOWED_CHAT_MODELS = getattr(settings, "OPENAI_CHAT_MODELS", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])


def _cache_key(prefix: str, payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"rag:{prefix}:{digest}"


async def _aget_or_set_cache(cache_key: str, ttl_seconds: int, compute_fn):
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    value = await compute_fn()
    cache.set(cache_key, value, timeout=ttl_seconds)
    return value


class RAGState(TypedDict, total=False):
    query: str
    top_k: int
    model: str
    temperature: float
    nearest: List[Dict[str, Any]]
    contexts: List[Dict[str, Any]]
    reject: bool
    answer: str
    messages: List[Any]


def _infer_transaction_type(text: str, current_value: str = "") -> str:
    candidate = (current_value or "").upper().strip()
    if candidate in {"RENT", "LEASE", "SALE", "OWNERSHIP"}:
        return "SALE" if candidate == "OWNERSHIP" else candidate

    lowered = (text or "").lower()
    if any(t in lowered for t in LEASE_TERMS):
        return "LEASE"
    if any(t in lowered for t in OWNERSHIP_TERMS):
        return "SALE"
    if any(t in lowered for t in RENT_TERMS):
        return "RENT"
    return "UNKNOWN"


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

    raw_text = " ".join(str(normalized.get(k, "")) for k in ["listing_type", "transaction_type", "cleaned_text", "location", "features"])
    inferred = _infer_transaction_type(raw_text, str(normalized.get("transaction_type", "")))
    normalized["transaction_type"] = inferred
    if normalized.get("listing_type") in ["UNKNOWN", "Not specified"]:
        normalized["listing_type"] = inferred
    return normalized


def _expanded_query(query: str) -> str:
    q = (query or "").strip()
    lower = q.lower()
    if any(term in lower for term in {"rent", "rental", "tenant"}):
        q += " rent rental tenant"
    if any(term in lower for term in {"lease", "leave and license", "l&l", "license"}):
        q += " lease leave and license"
    if any(term in lower for term in {"buy", "sale", "ownership", "own", "purchase"}):
        q += " sale ownership resale buy purchase"
    return q


def _extract_filters(query: str) -> Dict[str, Any]:
    lowered = (query or "").lower()
    filters: Dict[str, Any] = {}

    if any(term in lowered for term in {"buy", "purchase", "want"}):
        filters["transaction_type"] = "SALE"
    elif any(term in lowered for term in {"rent", "lease"}):
        filters["transaction_type"] = "RENT"

    if any(term in lowered for term in {"flat", "apartment"}):
        filters["property_type"] = "RESIDENTIAL"
    elif any(term in lowered for term in {"office", "shop"}):
        filters["property_type"] = "COMMERCIAL"

    return filters


def _extract_query_preferences(query: str) -> Dict[str, str]:
    lowered = (query or "").lower()
    prefs: Dict[str, str] = {}
    txn = _infer_transaction_type(lowered)
    if txn != "UNKNOWN":
        prefs["transaction_type"] = txn

    if any(t in lowered for t in {"office", "commercial", "shop"}):
        prefs["property_type"] = "COMMERCIAL"
    elif any(t in lowered for t in {"flat", "apartment", "villa", "residential", "bhk"}):
        prefs["property_type"] = "RESIDENTIAL"
    return prefs


def _is_domain_query(query: str) -> bool:
    if any(ch.isdigit() for ch in query):
        return True
    tokens = [t for t in re.split(r"\W+", query.lower()) if t]
    if len(tokens) < 2:
        return False
    matches = sum(1 for t in tokens if t in REAL_ESTATE_TERMS)
    return matches > 0


def _is_gibberish(query: str) -> bool:
    stripped = re.sub(r"\s+", "", query or "")
    if not stripped:
        return True
    alnum = sum(ch.isalnum() for ch in stripped)
    vowels = sum(ch.lower() in "aeiou" for ch in stripped if ch.isalpha())
    return alnum < 4 or (vowels == 0 and len(stripped) >= 6)


def _fetch_top_k_by_vector(query: str, filters: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    vectorstore = get_vectorstore()
    pg_filter = build_pg_filter(filters)
    docs = vectorstore.similarity_search_with_score(query=query, k=top_k, filter=pg_filter or None)

    results: List[Dict[str, Any]] = []
    for doc, score in docs:
        meta = doc.metadata or {}
        listing_id = meta.get("listing_chunk_id")
        if listing_id is None:
            listing_id = (meta.get("metadata") or {}).get("listing_chunk_id")
        if listing_id is None:
            continue
        results.append(
            {
                "listing_chunk_id": int(listing_id),
                "distance": float(score) if score is not None else None,
            }
        )
    return results




async def _afetch_top_k_by_vector(query: str, filters: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    cache_key = _cache_key("vec", {"q": query, "filters": filters, "k": top_k})

    async def _compute() -> List[Dict[str, Any]]:
        vectorstore = await get_vectorstore_async()
        pg_filter = build_pg_filter(filters)

        if hasattr(vectorstore, "asimilarity_search_with_score"):
            docs = await vectorstore.asimilarity_search_with_score(query=query, k=top_k, filter=pg_filter or None)
        else:
            docs = await asyncio.to_thread(
                vectorstore.similarity_search_with_score,
                query,
                top_k,
                pg_filter or None,
            )

        results: List[Dict[str, Any]] = []
        for doc, score in docs:
            meta = doc.metadata or {}
            listing_id = meta.get("listing_chunk_id")
            if listing_id is None:
                listing_id = (meta.get("metadata") or {}).get("listing_chunk_id")
            if listing_id is None:
                continue
            results.append(
                {
                    "listing_chunk_id": int(listing_id),
                    "distance": float(score) if score is not None else None,
                }
            )
        return results

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


def _hybrid_retrieve(query: str, top_k: int) -> List[Dict[str, Any]]:
    expanded = _expanded_query(query)
    candidate_pool = max(32, top_k * 5)
    filters = _extract_filters(query)
    vec_hits = _fetch_top_k_by_vector(expanded, filters=filters, top_k=candidate_pool)
    kw_hits = _fetch_top_k_by_keyword(expanded, top_k=candidate_pool)

    fused_scores: Dict[int, float] = defaultdict(float)
    rank_data: Dict[int, Dict[str, Any]] = {}
    query_prefs = _extract_query_preferences(query)
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

    listing_ids = list(fused_scores.keys())
    listing_map = {
        l.id: l
        for l in ListingChunk.objects.filter(id__in=listing_ids).only("id", "transaction_type", "category")
    }

    for lid in listing_ids:
        listing = listing_map.get(lid)
        if not listing:
            continue
        if query_prefs.get("transaction_type") and listing.transaction_type == query_prefs["transaction_type"]:
            fused_scores[lid] += 0.2

        property_pref = query_prefs.get("property_type")
        if property_pref == "COMMERCIAL" and listing.category == "COMMERCIAL":
            fused_scores[lid] += 0.15
        if property_pref == "RESIDENTIAL" and listing.category == "RESIDENTIAL":
            fused_scores[lid] += 0.15

    ranked_ids = [lid for lid, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]
    return [{"listing_chunk_id": lid, **rank_data.get(lid, {}), "hybrid_score": fused_scores[lid]} for lid in ranked_ids]


def _load_contexts(nearest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    listing_ids = [item["listing_chunk_id"] for item in nearest if item.get("listing_chunk_id")]
    listing_map = {l.id: l for l in ListingChunk.objects.filter(id__in=listing_ids)}

    contexts = []
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


def _build_messages(query: str, contexts: List[Dict[str, Any]], memory: str = "") -> List[Any]:
    snippet_blocks = []
    for idx, item in enumerate(contexts, start=1):
        meta = dict(item.get("metadata", {}))
        meta["_id"] = item.get("id")
        meta["_rank_score"] = round(float(item.get("hybrid_score", 0.0)), 6)
        snippet_blocks.append(f"[SNIPPET {idx}]\n{json.dumps(meta, ensure_ascii=False)}\n[/SNIPPET {idx}]")

    system_prompt = (
        "You are a real-estate listings assistant for retrieval QA. "
        "Only use snippet JSON facts. If query is irrelevant, gibberish, or outside real-estate listings, reply exactly: "
        "'I can only help with real-estate listing queries based on uploaded WhatsApp data.'\n\n"
        "When relevant:\n"
        "1) Rank listings by relevance to the user query giving priority to explicit constraints (location, budget, BHK, transaction type, furnishing, parking).\n"
        "2) Output concise markdown with sections: 'Top Matches' then 'Details'.\n"
        "3) In 'Top Matches', include numbered bullets with listing id and why relevant.\n"
        "4) In 'Details', provide a markdown table with columns: id, transaction_type, property_type, location, bhk, sqft, price, furnishing, parking, contact_numbers.\n"
        "5) Never leave blank values: write 'Not specified'.\n"
        "6) Distinguish RENT vs LEASE vs SALE clearly.\n"
        "7) If query asks for filtering, return only matching listings from snippets when possible.\n"
    )

    messages: List[Any] = [SystemMessage(content=system_prompt)]
    if memory:
        messages.append(SystemMessage(content=f"Conversation history:\n{memory}"))

    user_prompt = f"Query: {query}\n\nContext snippets:\n" + "\n".join(snippet_blocks)
    messages.append(HumanMessage(content=user_prompt))
    return messages


def _reject_text() -> str:
    return "I can only help with real-estate listing queries based on uploaded WhatsApp data."


def _route_after_retrieve(state: RAGState) -> str:
    return "reject" if state.get("reject") else "respond"


async def _hybrid_retrieve_async(query: str, top_k: int) -> List[Dict[str, Any]]:
    cache_key = _cache_key("hybrid", {"q": query, "k": top_k})

    async def _compute() -> List[Dict[str, Any]]:
        return await asyncio.to_thread(_hybrid_retrieve, query, top_k)

    return await _aget_or_set_cache(cache_key, 90, _compute)


async def _load_contexts_async(nearest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cache_key = _cache_key("ctx", nearest)

    async def _compute() -> List[Dict[str, Any]]:
        return await asyncio.to_thread(_load_contexts, nearest)

    return await _aget_or_set_cache(cache_key, 90, _compute)


async def _retrieve_node(state: RAGState) -> RAGState:
    query = state.get("query", "")
    top_k = max(1, min(30, int(state.get("top_k") or DEFAULT_TOP_K)))
    nearest = await _hybrid_retrieve_async(query, top_k=top_k)
    contexts = await _load_contexts_async(nearest)
    reject = _is_gibberish(query) or (not contexts and not _is_domain_query(query))
    return {"nearest": nearest, "contexts": contexts, "reject": reject}


async def _reject_node(_: RAGState) -> RAGState:
    cache_key = _cache_key("reject", "static")

    async def _compute() -> RAGState:
        return {"answer": _reject_text()}

    return await _aget_or_set_cache(cache_key, 3600, _compute)


async def _respond_node(state: RAGState) -> RAGState:
    model = state.get("model") if state.get("model") in ALLOWED_CHAT_MODELS else CHAT_MODEL
    temperature = float(state.get("temperature", 0.0))
    query = state.get("query", "")
    contexts = state.get("contexts", [])
    memory = state.get("memory", "")

    cache_key = _cache_key(
        "resp",
        {"model": model, "temperature": temperature, "query": query, "contexts": contexts, "memory": memory},
    )

    async def _compute() -> RAGState:
        messages = _build_messages(query, contexts, memory=memory)
        llm = ChatOpenAI(
            model=model,
            api_key=getattr(settings, "OPENAI_API_KEY", None),
            temperature=temperature,
            max_tokens=900,
        )
        response = await llm.ainvoke(messages)
        answer = (response.content or "").strip() if response else ""
        if not answer:
            answer = "I couldn't find enough listing evidence to answer that reliably."
        return {"answer": answer, "messages": messages, "model": model}

    return await _aget_or_set_cache(cache_key, 60, _compute)


def build_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("reject", _reject_node)
    graph.add_node("respond", _respond_node)
    graph.set_entry_point("retrieve")
    graph.add_conditional_edges("retrieve", _route_after_retrieve, {"reject": "reject", "respond": "respond"})
    graph.add_edge("reject", END)
    graph.add_edge("respond", END)
    return graph.compile()


RAG_WORKFLOW = build_rag_graph()


async def run_rag(query: str, top_k: int = DEFAULT_TOP_K, model: str = CHAT_MODEL, temperature: float = 0.0, memory: str = "") -> RAGState:
    state: RAGState = {
        "query": query,
        "top_k": top_k,
        "model": model,
        "temperature": temperature,
        "memory": memory,
    }
    return await RAG_WORKFLOW.ainvoke(state)


def stream_llm_from_state(state: RAGState):
    model = state.get("model") if state.get("model") in ALLOWED_CHAT_MODELS else CHAT_MODEL
    llm = ChatOpenAI(
        model=model,
        api_key=getattr(settings, "OPENAI_API_KEY", None),
        temperature=float(state.get("temperature", 0.0)),
        max_tokens=900,
        streaming=True,
    )
    for chunk in llm.stream(state.get("messages", [])):
        text = chunk.content or ""
        if text:
            yield text
