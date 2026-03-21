# apps/core/views.py
import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db import connection
from django.http import HttpRequest, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.utils.html import escape

from openai import OpenAI

from apps.core.utils.html_sanitiser import clean_html
from apps.preprocessing.models import ListingChunk

from .models import ChatMessage

log = logging.getLogger(__name__)

EMBEDDING_MODEL = getattr(settings, "EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = getattr(settings, "CHAT_MODEL", "gpt-4o-mini")
DEFAULT_TOP_K = 8
ALLOWED_CHAT_MODELS = getattr(
    settings,
    "OPENAI_CHAT_MODELS",
    ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
)

REAL_ESTATE_TERMS = {
    "office", "shop", "villa", "bhk", "sqft", "carpet", "price", "budget", "listing", "ownership", "commercial",
    "residential", "tenant", "furnish", "deposit", "location", "area", "tower", "building", "layout",
}
OWNERSHIP_TERMS = {"buy", "sale", "sell", "ownership", "own", "purchase", "resale"}
LEASE_TERMS = {"lease", "leave and license", "l&l", "license"}
RENT_TERMS = {"rent", "rental", "tenant", "let out"}


def _get_client() -> OpenAI:
    return OpenAI(api_key=getattr(settings, "OPENAI_API_KEY", None))


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

    raw_text = " ".join(
        str(normalized.get(k, ""))
        for k in ["listing_type", "transaction_type", "cleaned_text", "location", "features"]
    )
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



def _extract_query_preferences(query: str) -> Dict[str, str]:
    """
    Infer simple hard preferences from the query so reranking can prioritize
    listings that explicitly match intent.
    """
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


def _embed_query(text: str) -> List[float]:
    if not text:
        return []
    resp = _get_client().embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def _vector_to_pg_literal(vec: List[float]) -> str:
    inner = ",".join(f"{float(x):.6f}" for x in vec)
    return f"'[{inner}]'::vector"


def _extract_filters(query: str) -> Dict[str, Any]:
    lowered = (query or "").lower()
    filters: Dict[str, Any] = {}

    if any(term in lowered for term in {"buy", "purchase", "want"}):
        filters["listing_intent"] = "OFFER"
        filters["transaction_type"] = "SALE"
    elif any(term in lowered for term in {"rent", "lease"}):
        filters["listing_intent"] = "OFFER"
        filters["transaction_type"] = "RENT"

    if any(term in lowered for term in {"flat", "apartment"}):
        filters["property_type"] = "RESIDENTIAL"
    elif any(term in lowered for term in {"office", "shop"}):
        filters["property_type"] = "COMMERCIAL"

    return filters


def _fetch_top_k_by_vector(vec: List[float], filters: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    if not vec:
        return []
    
    where_clauses = ["er.embedding_vector IS NOT NULL"]
    params: List[Any] = []

    for key, value in (filters or {}).items():
        where_clauses.append("lc.metadata->>%s = %s")
        params.extend([key, str(value)])


    sql = f"""
        SELECT er.id, er.listing_chunk_id, er.embedding_vector <-> {_vector_to_pg_literal(vec)} AS distance
        FROM preprocessing_embeddingrecord er
        JOIN preprocessing_listingchunk lc ON lc.id = er.listing_chunk_id
        WHERE {' AND '.join(where_clauses)}
        ORDER BY distance
        LIMIT %s
    """
    params.append(top_k)

    with connection.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [
        {
            "embedding_record_id": r[0],
            "listing_chunk_id": r[1],
            "distance": float(r[2]) if r[2] is not None else None,
        }
        for r in rows
    ]


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


def _hybrid_retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    expanded = _expanded_query(query)
    candidate_pool = max(32, top_k * 5)
    filters = _extract_filters(query)
    vec_hits = _fetch_top_k_by_vector(_embed_query(expanded), filters=filters, top_k=candidate_pool)
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
        for l in ListingChunk.objects.filter(id__in=listing_ids).only(
            "id", "transaction_type", "category", "intent", "metadata"
        )
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


def _get_recent_messages(user, limit=6) -> str:
    msgs = ChatMessage.objects.filter(user=user).order_by("-timestamp")[:limit]
    msgs = list(reversed(msgs))
    return "\n".join(f"{m.role.upper()}: {m.content}" for m in msgs)


def _build_prompt(query: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
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

    user_prompt = (
        f"Query: {query}\n\n"
        "Context snippets:\n"
        + "\n".join(snippet_blocks)
    )

    messages = [{"role": "system", "content": system_prompt}]
    return messages + [{"role": "user", "content": user_prompt}]


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


def _reject_text() -> str:
    return "I can only help with real-estate listing queries based on uploaded WhatsApp data."


def _parse_request(request: HttpRequest) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(request.body.decode("utf-8"))
    except Exception:
        return None


def _should_reject(query: str, contexts: List[Dict[str, Any]]) -> bool:
    if _is_gibberish(query):
        return True
    if contexts:
        return False
    return not _is_domain_query(query)


def _stream_chat_completion(messages: List[Dict[str, str]], model: str, temperature: float) -> Iterable[str]:
    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=900,
        stream=True,
    )
    for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield json.dumps({"type": "token", "delta": delta}) + "\n"


@login_required
def get_listing_source(request, pk: int):
    try:
        listing = ListingChunk.objects.get(pk=pk)
        if not listing.raw_chunk:
            return JsonResponse({"error": "No source file linked to this listing."}, status=404)

        return JsonResponse(
            {
                "id": listing.id,
                "raw_text": listing.raw_chunk.raw_text,
                "sender": listing.raw_chunk.sender or "Unknown",
                "timestamp": listing.raw_chunk.message_start.isoformat() if listing.raw_chunk.message_start else None,
            }
        )
    except ListingChunk.DoesNotExist:
        return JsonResponse({"error": "Listing not found."}, status=404)
    except Exception:
        log.exception("Error fetching source for listing %s", pk)
        return JsonResponse({"error": "Internal server error"}, status=500)


@login_required
def chat_query(request: HttpRequest):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    body = _parse_request(request)
    if body is None:
        return JsonResponse({"error": "invalid json"}, status=400)

    query = (body.get("query") or body.get("q") or "").strip()
    if not query:
        return JsonResponse({"error": "query is required"}, status=400)

    top_k = int(body.get("top_k", DEFAULT_TOP_K))
    temperature = float(body.get("temperature", 0.0))
    selected_model = body.get("model") if body.get("model") in ALLOWED_CHAT_MODELS else CHAT_MODEL

    ChatMessage.objects.create(user=request.user, role="user", content=query)

    try:
        nearest = _hybrid_retrieve(query, top_k=top_k)
        contexts = _load_contexts(nearest)

        if _should_reject(query, contexts):
            answer = _reject_text()
        else:
            messages = _build_prompt(query, contexts)
            memory = _get_recent_messages(request.user, limit=6)
            if memory:
                messages.insert(1, {"role": "system", "content": "Conversation history:\n" + memory})
            resp = _get_client().chat.completions.create(
                model=selected_model,
                messages=messages,
                temperature=temperature,
                max_tokens=900,
            )
            answer = (resp.choices[0].message.content or "").strip() if resp.choices else ""
            if not answer:
                answer = "I couldn't find enough listing evidence to answer that reliably."

        ChatMessage.objects.create(user=request.user, role="assistant", content=answer)
        return JsonResponse({"answer": clean_html(answer), "sources": contexts, "model": selected_model})
    except Exception as e:
        log.exception("chat_query failed")
        return JsonResponse({"error": "model_call_failed", "details": str(e)}, status=500)


@login_required
def chat_stream(request: HttpRequest):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    body = _parse_request(request)
    if body is None:
        return JsonResponse({"error": "invalid json"}, status=400)

    query = (body.get("query") or "").strip()
    if not query:
        return JsonResponse({"error": "query is required"}, status=400)

    top_k = int(body.get("top_k", DEFAULT_TOP_K))
    temperature = float(body.get("temperature", 0.0))
    selected_model = body.get("model") if body.get("model") in ALLOWED_CHAT_MODELS else CHAT_MODEL

    ChatMessage.objects.create(user=request.user, role="user", content=query)

    try:
        nearest = _hybrid_retrieve(query, top_k=top_k)
        contexts = _load_contexts(nearest)
    except Exception as e:
        log.exception("stream retrieval failed")
        return JsonResponse({"error": "hybrid_lookup_failed", "details": str(e)}, status=500)

    if _should_reject(query, contexts):
        msg = _reject_text()
        ChatMessage.objects.create(user=request.user, role="assistant", content=msg)

        def reject_iter():
            yield json.dumps({"type": "token", "delta": msg}) + "\n"
            yield json.dumps({"type": "done", "model": selected_model, "sources": contexts}) + "\n"

        return StreamingHttpResponse(reject_iter(), content_type="application/x-ndjson")

    messages = _build_prompt(query, contexts)
    memory = _get_recent_messages(request.user, limit=6)
    if memory:
        messages.insert(1, {"role": "system", "content": "Conversation history:\n" + memory})

    def stream_iter():
        parts: List[str] = []
        try:
            for line in _stream_chat_completion(messages, selected_model, temperature):
                payload = json.loads(line)
                if payload.get("type") == "token":
                    parts.append(payload.get("delta", ""))
                yield line
        except Exception as e:
            yield json.dumps({"type": "error", "message": f"Streaming error: {escape(str(e))}"}) + "\n"
        finally:
            final_answer = "".join(parts).strip() or "I couldn't find enough listing evidence to answer that reliably."
            try:
                ChatMessage.objects.create(user=request.user, role="assistant", content=final_answer)
            except Exception:
                log.exception("Failed to store streamed assistant message")
            yield json.dumps({"type": "done", "model": selected_model, "sources": contexts}) + "\n"

    return StreamingHttpResponse(stream_iter(), content_type="application/x-ndjson")


@login_required
def chat_view(request):
    qs = ChatMessage.objects.filter(user=request.user).order_by("timestamp")
    history = [{"role": m.role, "content": m.content} for m in qs]
    return render(
        request,
        "core/chat.html",
        {"history": history, "available_models": ALLOWED_CHAT_MODELS, "default_model": CHAT_MODEL},
    )