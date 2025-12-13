# apps/core/views.py
import json
import logging
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

from .models import ChatMessage

log = logging.getLogger(__name__)

#HELPERS

# === Structured snippet helpers ===

def format_snippet_block(snippet: Dict[str, Any], idx: int) -> str:
    meta = snippet.get("metadata", {})
    meta["_id"] = snippet.get("id")  # <-- critical stable reference
    
    json_text = json.dumps(meta, ensure_ascii=False)

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
            "If a field is absent or null, omit it entirely from the table."
        )

    }

    # build user message using structured snippets
    snippet_metas = contexts
    user_message = build_chat_prompt(query, snippet_metas)

    return [
        system,
        {"role": "user", "content": user_message}
    ]

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

    # 1) Embed the query
    try:
        q_vec = _embed_query(query)
    except Exception as e:
        log.exception("Embedding failure: %s", e)
        return JsonResponse({"error": "embedding_failed", "details": str(e)}, status=500)

    # 2) Find top-k vectors in DB
    try:
        nearest = _fetch_top_k_by_vector(q_vec, top_k=top_k)
    except Exception as e:
        log.exception("Vector DB lookup failed: %s", e)
        return JsonResponse({"error": "vector_lookup_failed", "details": str(e)}, status=500)

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
            "metadata": l.metadata or {},
            "distance": r.get("distance"),
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
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=800,
        )
        # pick the first assistant message
        answer = ""
        if resp.choices and len(resp.choices) > 0:
            answer = resp.choices[0].message.content
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

    return render(request, "core/chat.html", {"history": history})
