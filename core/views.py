# apps/core/views.py
import json
import logging
import re
from collections import defaultdict
from typing import List, Dict, Any
from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.db import connection
from django.utils import timezone
from django.shortcuts import render
from django.contrib.auth.decorators import login_required

from openai import OpenAI

from apps.preprocessing.models import ListingChunk, EmbeddingRecord
from apps.core.utils.html_sanitiser import clean_html

from apps.core.models import ChatMessage

log = logging.getLogger(__name__)

#HELPERS

# === Structured snippet helpers ===

def format_snippet_block(snippet: Dict[str, Any], idx: int) -> str:
    meta = snippet.get("metadata", {})
    meta["_id"] = snippet.get("id")  # <-- critical stable reference
    
    # We dump the whole metadata dict as JSON so the LLM can parse it reliably
    json_text = json.dumps(meta, indent=2)
    return (
        f"[SNIPPET {idx}]\n"
        f"{json_text}\n"
        f"[/SNIPPET {idx}]\n"
    )


def build_chat_prompt(user_query: str, snippet_metas: List[Dict[str, Any]]) -> str:
    """Construct final user prompt with strict JSON blocks."""
    blocks = []
    for i, meta in enumerate(snippet_metas, start=1):
        blocks.append(format_snippet_block(meta, i))

    return (
        f"Query: {user_query}\n\n"
        f"Use ONLY the JSON in the snippets.\n"
        f"If a field doesn't exist OR is null, OMIT it entirely from the table.\n\n"
        + "\n".join(blocks)
    )


# init OpenAI client (uses OPENAI_API_KEY from settings or env)
client = OpenAI(api_key=getattr(settings, "OPENAI_API_KEY", None))

EMBEDDING_MODEL = getattr(settings, "EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = getattr(settings, "CHAT_MODEL", "gpt-4o-mini")
DEFAULT_TOP_K = 5
ALLOWED_CHAT_MODELS = getattr(
    settings,
    "OPENAI_CHAT_MODELS",
    ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
)

OWNERSHIP_TERMS = {"buy", "sale", "sell", "ownership", "own", "purchase", "resale"}
LEASE_TERMS = {"lease", "leave and license", "l&l", "ll"}
RENT_TERMS = {"rent", "rental", "tenant", "let out"}


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
    if any(term in lower for term in {"lease", "leave and license", "l&l"}):
        q += " lease leave and license"
    if any(term in lower for term in {"buy", "sale", "ownership", "own"}):
        q += " sale ownership resale buy"
    return q

def _embed_query(text: str) -> List[float]:
    """Return embedding vector (list of floats) for a query."""
    if not text:
        return []
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    vector = resp.data[0].embedding
    return vector

def _vector_to_pg_literal(vec: List[float]) -> str:
    """
    Convert a python list to a Postgres pgvector literal: '[0.1,0.2,...]'::vector
    (We use this in raw SQL ordering with <-> operator).
    """
    # keep a consistent float formatting to avoid extremely long strings
    inner = ",".join(f"{float(x):.6f}" for x in vec)
    return f"'[{inner}]'::vector"

def _fetch_top_k_by_vector(vec: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts: {embedding_id, listing_chunk_id, distance}
    Performs similarity search on preprocessing_embeddingrecord.
    Assumes your table is named 'preprocessing_embeddingrecord' and column 'embedding_vector'.
    """
    if not vec:
        return []

    vector_literal = _vector_to_pg_literal(vec)
    # Raw SQL: use the <-> operator for pgvector distance (Euclidean/inner product depending on index)
    sql = f"""
        SELECT id, listing_chunk_id, embedding_vector <-> {vector_literal} AS distance
        FROM preprocessing_embeddingrecord
        WHERE embedding_vector IS NOT NULL
        ORDER BY distance
        LIMIT %s
    """

    with connection.cursor() as cur:
        cur.execute(sql, [top_k])
        rows = cur.fetchall()
    results = []
    for r in rows:
        results.append({
            "embedding_record_id": r[0],
            "listing_chunk_id": r[1],
            "distance": float(r[2]) if r[2] is not None else None,
        })
    return results


def _fetch_top_k_by_keyword(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    if not query:
        return []

    terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
    if not terms:
        return []

    where_clause = " OR ".join(["LOWER(lc.text) LIKE %s" for _ in terms])
    params = [f"%{t}%" for t in terms]
    sql = f"""
        SELECT lc.id, COUNT(*)::float AS keyword_score
        FROM preprocessing_listingchunk lc
        WHERE {where_clause}
        GROUP BY lc.id
        ORDER BY keyword_score DESC, lc.last_seen DESC
        LIMIT %s
    """
    params.append(top_k)
    with connection.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [{"listing_chunk_id": r[0], "keyword_score": float(r[1])} for r in rows]


def _hybrid_retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    q = _expanded_query(query)
    vec_hits = _fetch_top_k_by_vector(_embed_query(q), top_k=max(20, top_k * 4))
    kw_hits = _fetch_top_k_by_keyword(q, top_k=max(20, top_k * 4))

    fused_scores: Dict[int, float] = defaultdict(float)
    rank_data: Dict[int, Dict[str, Any]] = {}
    for rank, item in enumerate(vec_hits, start=1):
        lid = item.get("listing_chunk_id")
        if not lid:
            continue
        fused_scores[lid] += 1 / (50 + rank)
        rank_data[lid] = {**rank_data.get(lid, {}), **item}

    for rank, item in enumerate(kw_hits, start=1):
        lid = item.get("listing_chunk_id")
        if not lid:
            continue
        fused_scores[lid] += 1 / (50 + rank)
        rank_data[lid] = {**rank_data.get(lid, {}), **item}

    ranked_ids = [lid for lid, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]
    return [{"listing_chunk_id": lid, **rank_data.get(lid, {}), "hybrid_score": fused_scores[lid]} for lid in ranked_ids]


def _get_recent_messages(user, limit=6) -> str:
    msgs = ChatMessage.objects.filter(user=user).order_by("-timestamp")[:limit]
    msgs = list(reversed(msgs))
    return "\n".join(f"{m.role.upper()}: {m.content}" for m in msgs)



def _build_rag_prompt(query: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Build messages for the chat model. Return messages array.
    contexts: list of dicts containing "text" and "metadata" fields
    We'll produce a concise system prompt and then supply each context as a numbered snippet.
    """
    system = {
        "role": "system",
        "content": (
            "You answer strictly using the JSON inside the snippets. Always include contact number."
            "Each snippet includes a unique numeric _id. "
            "When creating tables, always include a column named 'id' populated with this _id. "
            "For follow-up questions, ALWAYS use the id to retrieve information. "
            "Never guess, never use outside knowledge, never invent missing values. "
            "Differentiate transaction type clearly: RENT vs LEASE vs SALE/OWNERSHIP. "
            "Never leave important values blank; use 'Not specified' when metadata is unavailable."
        )

    }

    # build user message using structured snippets
    snippet_metas = contexts
    user_message = build_chat_prompt(query, snippet_metas)

    return [
        system,
        {"role": "user", "content": user_message}
    ]

@login_required
def get_listing_source(request, pk: int):
    """
    API to fetch the original raw message text for a specific ListingChunk.
    Used for the 'View Source' feature in the chat.
    """
    try:
        listing = ListingChunk.objects.get(pk=pk)
        
        # Follow the relationship to the raw chunk
        if not listing.raw_chunk:
            return JsonResponse({"error": "No source file linked to this listing."}, status=404)
            
        return JsonResponse({
            "text": listing.raw_chunk.text,
            "filename": listing.raw_chunk.source_file or "Unknown Source"
        })
    except ListingChunk.DoesNotExist:
        return JsonResponse({"error": "Listing not found"}, status=404)
    except Exception as e:
        log.exception("Error fetching source: %s", e)
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def chat_query(request: HttpRequest):
    """
    POST endpoint: accepts JSON {"query": "...", "top_k": 5, "temperature": 0.0}
    Returns: {"answer": "...", "sources": [...], "raw_model_response": ...}
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "invalid json"}, status=400)

    query = body.get("query") or body.get("q")
    if not query or not isinstance(query, str) or not query.strip():
        return JsonResponse({"error": "query is required"}, status=400)
    
    ChatMessage.objects.create(
        user=request.user,
        role="user",
        content=query
    )

    top_k = int(body.get("top_k", DEFAULT_TOP_K))
    temperature = float(body.get("temperature", 0.0))
    selected_model = body.get("model") or CHAT_MODEL
    if selected_model not in ALLOWED_CHAT_MODELS:
        selected_model = CHAT_MODEL

    # 1) Embed the query
    try:
        nearest = _hybrid_retrieve(query, top_k=top_k)
    except Exception as e:
        log.exception("Hybrid lookup failed: %s", e)
        return JsonResponse({"error": "hybrid_lookup_failed", "details": str(e)}, status=500)

    # 3) Load ListingChunk objects and prepare context
    listing_ids = [r["listing_chunk_id"] for r in nearest if r.get("listing_chunk_id")]
    listings = ListingChunk.objects.filter(id__in=listing_ids)
    # preserve original order by id list
    listing_map = {l.id: l for l in listings}
    contexts = []
    for r in nearest:
        lid = r.get("listing_chunk_id")
        if not lid:
            continue
        l = listing_map.get(lid)
        if not l:
            continue
        contexts.append({
            "id": l.id,
            "metadata": _with_metadata_defaults(l.metadata or {}),
            "distance": r.get("distance"),
            "keyword_score": r.get("keyword_score"),
            "hybrid_score": r.get("hybrid_score"),
        })

    # 4) Build prompt and call chat model

    # Add short user-specific memory
    memory_text = _get_recent_messages(request.user, limit=6)
    messages = _build_rag_prompt(query, contexts)

    if memory_text.strip():
        messages.insert(1, {
            "role": "system",
            "content": "Conversation history:\n" + memory_text
        })

    try:
        draft_resp = client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=temperature,
            max_tokens=800,
        )
        draft_answer = draft_resp.choices[0].message.content if draft_resp.choices else ""

        refine_messages = messages + [
            {"role": "assistant", "content": draft_answer or ""},
            {
                "role": "user",
                "content": (
                    "Review your previous answer for accuracy and completeness using only snippets. "
                    "Improve formatting and ensure rent/lease/sale are not mixed. "
                    "Do not add hallucinated facts."
                ),
            },
        ]
        resp = client.chat.completions.create(
            model=selected_model,
            messages=refine_messages,
            temperature=0.0,
            max_tokens=800,
        )
        answer = ""
        if resp.choices and len(resp.choices) > 0:
            answer = resp.choices[0].message.content or draft_answer
            try:
                ChatMessage.objects.create(
                    user = request.user,
                    role="assistant",
                    content=answer
                )
            except Exception as e:
                log.exception(f"Can't store response in DB:{e}")
        else:
            answer = "No response from model."
    except Exception as e:
        log.exception("Chat model call failed: %s", e)
        return JsonResponse({"error": "model_call_failed", "details": str(e)}, status=500)

    return JsonResponse({
        "answer": clean_html(answer),
        "sources": contexts,
        "model": selected_model,
        "raw_model_response": getattr(resp, "to_dict", lambda: {})() if resp else {}
    })

@login_required
def chat_view(request):
    qs = ChatMessage.objects.filter(user=request.user).order_by("timestamp")

    history = [
        {
            "role": m.role,
            "content": m.content
        }
        for m in qs
    ]

    return render(
        request,
        "core/chat.html",
        {"history": history, "available_models": ALLOWED_CHAT_MODELS, "default_model": CHAT_MODEL},
    )