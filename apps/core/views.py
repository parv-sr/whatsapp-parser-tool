import json
import logging
from typing import Any, Dict, List, Optional

from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.utils.html import escape

from apps.core.rag_graph import CHAT_MODEL, ALLOWED_CHAT_MODELS, run_rag, stream_llm_from_state
from apps.core.utils.html_sanitiser import clean_html
from apps.preprocessing.models import ListingChunk

from .models import ChatMessage

log = logging.getLogger(__name__)
DEFAULT_TOP_K = 8


def _get_recent_messages(user, limit=6) -> str:
    msgs = ChatMessage.objects.filter(user=user).order_by("-timestamp")[:limit]
    msgs = list(reversed(msgs))
    return "\n".join(f"{m.role.upper()}: {m.content}" for m in msgs)


def _parse_request(request: HttpRequest) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(request.body.decode("utf-8"))
    except Exception:
        return None


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
        memory = _get_recent_messages(request.user, limit=6)
        rag_state = run_rag(query=query, top_k=top_k, model=selected_model, temperature=temperature, memory=memory)
        answer = rag_state.get("answer") or "I couldn't find enough listing evidence to answer that reliably."
        contexts = rag_state.get("contexts", [])

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
        memory = _get_recent_messages(request.user, limit=6)
        rag_state = run_rag(query=query, top_k=top_k, model=selected_model, temperature=temperature, memory=memory)
        contexts = rag_state.get("contexts", [])
    except Exception as e:
        log.exception("stream retrieval failed")
        return JsonResponse({"error": "hybrid_lookup_failed", "details": str(e)}, status=500)

    if rag_state.get("reject"):
        msg = rag_state.get("answer") or "I can only help with real-estate listing queries based on uploaded WhatsApp data."
        ChatMessage.objects.create(user=request.user, role="assistant", content=msg)

        def reject_iter():
            yield json.dumps({"type": "token", "delta": msg}) + "\n"
            yield json.dumps({"type": "done", "model": selected_model, "sources": contexts}) + "\n"

        return StreamingHttpResponse(reject_iter(), content_type="application/x-ndjson")

    def stream_iter():
        parts: List[str] = []
        try:
            for delta in stream_llm_from_state(rag_state):
                parts.append(delta)
                yield json.dumps({"type": "token", "delta": delta}) + "\n"
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
