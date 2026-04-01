import json
import logging
from asgiref.sync import async_to_sync
from asgiref.sync import sync_to_async

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, JsonResponse, StreamingHttpResponse
from django.shortcuts import render

from apps.core.utils.html_sanitiser import clean_html
from apps.preprocessing.models import ListingChunk

from .models import ChatMessage
from .rag_graph import get_recent_messages_text, run_rag, stream_rag_events

log = logging.getLogger(__name__)

CHAT_MODEL = getattr(settings, "CHAT_MODEL", "gpt-4o-mini")
DEFAULT_TOP_K = 10
ALLOWED_CHAT_MODELS = getattr(settings, "OPENAI_CHAT_MODELS", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])


def _parse_request(request: HttpRequest):
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
async def chat_query(request: HttpRequest):
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

    await sync_to_async(ChatMessage.objects.create)(user=request.user, role="user", content=query)

    try:
        memory = await get_recent_messages_text(request.user, limit=6)
        state = await run_rag(
            query=query,
            top_k=top_k,
            model=selected_model,
            temperature=temperature,
            memory=memory,
            thread_id=f"user-{request.user.id}",
        )
        answer = (state.get("answer") or "I couldn't find enough listing evidence to answer that reliably.").strip()
        sources = state.get("sources") or []

        await sync_to_async(ChatMessage.objects.create)(user=request.user, role="assistant", content=answer)
        return JsonResponse({"answer": clean_html(answer), "sources": sources, "model": selected_model})
    except Exception as e:
        log.exception("chat_query failed")
        return JsonResponse({"error": "model_call_failed", "details": str(e)}, status=500)


@login_required
async def chat_stream(request: HttpRequest):
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

    await sync_to_async(ChatMessage.objects.create)(user=request.user, role="user", content=query)
    thread_id = f"user-{request.user.id}"
    is_asgi_request = hasattr(request, "scope")
    memory = await get_recent_messages_text(request.user, limit=6)

    if is_asgi_request:
        async def stream_iter():
            parts = []
            final_sources = []
            final_model = selected_model
            try:
                async for payload in stream_rag_events(
                    query=query,
                    top_k=top_k,
                    model=selected_model,
                    temperature=temperature,
                    memory=memory,
                    thread_id=thread_id,
                ):
                    if payload.get("type") == "token":
                        delta = payload.get("delta", "")
                        if delta:
                            parts.append(delta)
                            yield json.dumps({"type": "token", "delta": delta}) + "\n"
                    elif payload.get("type") == "final":
                        final_sources = payload.get("sources") or []
                        final_model = payload.get("model") or selected_model
                        if not parts and payload.get("answer"):
                            answer_text = payload.get("answer")
                            parts.append(answer_text)
                            yield json.dumps({"type": "token", "delta": answer_text}) + "\n"
            except Exception as e:
                yield json.dumps({"type": "error", "message": f"Streaming error: {str(e)}"}) + "\n"
            finally:
                final_answer = "".join(parts).strip() or "I couldn't find enough listing evidence to answer that reliably."
                try:
                    await sync_to_async(ChatMessage.objects.create)(user=request.user, role="assistant", content=final_answer)
                except Exception:
                    log.exception("Failed to store streamed assistant message")
                yield json.dumps({"type": "done", "model": final_model, "sources": final_sources}) + "\n"

        return StreamingHttpResponse(stream_iter(), content_type="application/x-ndjson")

    try:
        state = await run_rag(
            query=query,
            top_k=top_k,
            model=selected_model,
            temperature=temperature,
            memory=memory,
            thread_id=thread_id,
        )
        final_answer = (state.get("answer") or "").strip() or "I couldn't find enough listing evidence to answer that reliably."
        final_sources = state.get("sources") or []
        final_model = state.get("model") or selected_model
    except Exception as e:
        final_answer = "I couldn't find enough listing evidence to answer that reliably."
        final_sources = []
        final_model = selected_model
        error_chunk = json.dumps({"type": "error", "message": f"Streaming error: {str(e)}"}) + "\n"
    else:
        error_chunk = None

    try:
        await sync_to_async(ChatMessage.objects.create)(user=request.user, role="assistant", content=final_answer)
    except Exception:
        log.exception("Failed to store streamed assistant message")

    def sync_stream_iter():
        if error_chunk:
            yield error_chunk
        yield json.dumps({"type": "token", "delta": final_answer}) + "\n"
        yield json.dumps({"type": "done", "model": final_model, "sources": final_sources}) + "\n"

    return StreamingHttpResponse(sync_stream_iter(), content_type="application/x-ndjson")


@login_required
def chat_view(request):
    qs = ChatMessage.objects.filter(user=request.user).order_by("timestamp")
    history = [{"role": m.role, "content": m.content} for m in qs]
    return render(
        request,
        "core/chat.html",
        {"history": history, "available_models": ALLOWED_CHAT_MODELS, "default_model": CHAT_MODEL},
    )
