# apps/ingestion/pipeline.py
import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import re
import hashlib
import logging
import os
import json
import subprocess
import tempfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from asgiref.sync import sync_to_async
import multiprocessing
import threading

from django.utils import timezone
from django.db import transaction, connection, close_old_connections
from django.core.cache import cache

# Models
from apps.ingestion.models import RawFile, RawMessageChunk
from apps.ingestion.dedupe.pre_llm_dedupe import PreLLMDedupe
from apps.ingestion.dedupe.dupe_tracker import DupeTracker
from apps.preprocessing.models import ListingChunk, EmbeddingRecord
from apps.embeddings.vector_store import aupsert_listing_embeddings

# Extraction
from apps.preprocessing.extractor import BatchItemResult, extract_listings_from_batch

# Vectoriser
try:
    from apps.embeddings.vectoriser import get_batch_embeddings
except ImportError:
    get_batch_embeddings = None

from config import settings

log = logging.getLogger(__name__)

# --- CONFIGURATION ---
BATCH_SIZE = 1000
NUM_CORES = multiprocessing.cpu_count()
RUST_PARSER_BIN = os.path.join(
    settings.BASE_DIR, 
    "rust_parser", "whatsapp-parser", "target", "release", "whatsapp-parser.exe"
)

LLM_BATCH_SIZE = 60
MAX_WORKERS = max(4, int(os.getenv("INGESTION_MAX_WORKERS", "6")))
VECTOR_DEDUPE_DISTANCE_THRESHOLD = 0.05
RUNTIME_LOG_LIMIT = 150

log.info(f"Pipeline initialized: {NUM_CORES} cores -> LLM_BATCH_SIZE={LLM_BATCH_SIZE}, MAX_WORKERS={MAX_WORKERS}")

# --- REGEX ---
MSG_START_RE = re.compile(
    r"^\[(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s+(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]\s+(?:~?\s?)(?P<sender>[^:]+):\s+",
    re.MULTILINE,
)
KEYWORDS_RE = re.compile(r"(bhk|rent|sale|lease|price|cr\b|lacs?|sqft|carpet|furnished|office|shop|flat|buy|sell|want)", re.IGNORECASE)
JUNK_RE = re.compile(r"(security code changed|waiting for this message|message was deleted|encrypted|joined|left|added|null|media omitted)", re.IGNORECASE)


# ---------------HELPER FUNCS---------------
def _runtime_log_key(raw_file_id: int) -> str:
    return f"runtime_logs:{raw_file_id}"


def _dedupe_stats_key(raw_file_id: int) -> str:
    return f"dedupe_stats:{raw_file_id}"


def _append_runtime_log(raw_file_id: int, level: str, message: str) -> None:
    ts = timezone.now().strftime("%H:%M:%S")
    line = f"{ts} [{level.upper()}] {message}"
    key = _runtime_log_key(raw_file_id)
    logs = cache.get(key, []) or []
    logs.append(line)
    if len(logs) > RUNTIME_LOG_LIMIT:
        logs = logs[-RUNTIME_LOG_LIMIT:]
    cache.set(key, logs, timeout=24 * 3600)


def _store_dedupe_stats(raw_file_id: int, tracker: DupeTracker) -> None:
    stats_payload = tracker.as_dict()
    cache.set(_dedupe_stats_key(raw_file_id), stats_payload, timeout=24 * 3600)
    RawFile.objects.filter(pk=raw_file_id).update(dedupe_stats=stats_payload)


def _remove_from_active_processing(raw_file_id: int) -> None:
    active = cache.get("processing_files", [])
    if raw_file_id in active:
        active.remove(raw_file_id)
        cache.set("processing_files", active, timeout=3600)
        log.info("Removed raw_file_id=%s from active processing list", raw_file_id)


def _cancel_requested(raw_file_id: int) -> bool:
    cancel_key = bool(cache.get(f"cancel:{raw_file_id}", False))
    db_cancelled = RawFile.objects.filter(pk=raw_file_id, status="CANCELLED").exists()
    cancelled = cancel_key or db_cancelled
    if cancelled:
        log.warning("Cancellation requested for raw_file_id=%s (cache=%s db=%s)", raw_file_id, cancel_key, db_cancelled)
    return cancelled


def _generate_dedupe_hash(
    cleaned_text: str,
    sender: str,
    location: str = "",
    transaction_type: str = "",
    listing_intent: str = "",
) -> str:
    norm_text = re.sub(r"\s+", "", (cleaned_text or "")).lower()
    norm_sender = re.sub(r"\s+", "", (sender or "")).lower()
    if not any([location, transaction_type, listing_intent]):
        # Backward-compatible behavior for callsites/fixtures that only provide text+sender.
        raw_key = f"{norm_text}|{norm_sender}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    norm_location = re.sub(r"\s+", "", (location or "")).lower()
    norm_transaction = re.sub(r"\s+", "", (transaction_type or "")).lower()
    norm_intent = re.sub(r"\s+", "", (listing_intent or "")).lower()
    raw_key = f"{norm_text}|{norm_sender}|{norm_location}|{norm_transaction}|{norm_intent}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _generate_legacy_dedupe_hash(cleaned_text: str, sender: str) -> str:
    """
    Backward-compatible dedupe hash used by older rows/tests:
    hash(normalized_text + normalized_sender).
    """
    norm_text = re.sub(r"\s+", "", (cleaned_text or "")).lower()
    norm_sender = re.sub(r"\s+", "", (sender or "")).lower()
    raw_key = f"{norm_text}|{norm_sender}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()




def _async_cache_key(prefix: str, payload: str) -> str:
    return f"pipeline:{prefix}:{payload}"


async def _aget_or_set_pipeline_cache(cache_key: str, ttl_seconds: int, compute_fn):
    cached = await sync_to_async(cache.get)(cache_key)
    if cached is not None:
        return cached
    value = await compute_fn()
    await sync_to_async(cache.set)(cache_key, value, timeout=ttl_seconds)
    return value


def _parse_whatsapp_datetime(date_raw: str | None, time_raw: str | None):
    if not date_raw:
        return None

    combined = f"{(date_raw or '').strip()} {(time_raw or '').strip()}".strip()
    fmts = [
        "%d/%m/%y %I:%M %p",
        "%d/%m/%y %I:%M:%S %p",
        "%d/%m/%y %H:%M",
        "%d/%m/%y %H:%M:%S",
        "%d/%m/%Y %I:%M %p",
        "%d/%m/%Y %I:%M:%S %p",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(combined, fmt)
        except Exception:
            continue
    return None


def _materialize_raw_file_to_temp_path(raw_file) -> str:
    with raw_file.file.open("rb") as src:
        with tempfile.NamedTemporaryFile(prefix="whatsapp_", suffix=".txt", delete=False) as tmp:
            for chunk in src.chunks():
                tmp.write(chunk)
            return tmp.name


def iter_rust_parsed_messages(raw_file):
    temp_input_path = _materialize_raw_file_to_temp_path(raw_file)
    try:
        cmd = [RUST_PARSER_BIN, "--input", temp_input_path]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if not proc.stdout:
            raise RuntimeError("Rust parser stdout was not available.")

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

        stderr = proc.stderr.read() if proc.stderr else ""
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Rust parser failed with exit code {rc}: {stderr.strip()}")
    finally:
        try:
            os.remove(temp_input_path)
        except OSError:
            pass

def stream_chat_messages(file_obj):
    buffer = []
    current_match = None
    for line in file_obj:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.replace("\r\n", "\n").replace("\r", "\n").replace("\0", "")
        match = MSG_START_RE.match(line)
        if match:
            if current_match and buffer:
                yield _parse_buffered_message(current_match, buffer)
            current_match = match
            buffer = [line]
        else:
            if buffer:
                buffer.append(line)
    if current_match and buffer:
        yield _parse_buffered_message(current_match, buffer)


def _parse_buffered_message(match, buffer):
    raw_date, raw_time = match.group("date"), match.group("time")
    sender = match.group("sender").strip()
    full_text = "".join(buffer)
    text_body = full_text[match.end():].strip().replace("\0", "")
    dt = None
    try:
        if len(raw_date.split("/")[-1]) == 2:
            fmt_base = "%d/%m/%y, %I:%M:%S %p"
        else:
            fmt_base = "%d/%m/%Y, %I:%M:%S %p"
        if len(raw_time.split(":")) == 2:
            fmt = fmt_base.replace(":%S", "")
        else:
            fmt = fmt_base
        ts_str = f"{raw_date}, {raw_time}"
        dt = datetime.strptime(ts_str, fmt)
    except Exception:
        dt = None
    return {"timestamp": dt, "sender": sender, "text": text_body, "raw_full": full_text}


def process_single_llm_batch(batch_data):
    """
    THREAD WORKER: Processes a batch of message chunks.
    CRITICAL: Manages DB connections explicitly to prevent deadlocks.
    """
    manage_db_connection = threading.current_thread() is not threading.main_thread()
    if manage_db_connection:
        close_old_connections()

    batch_idx, chunk_ids, raw_file_id = batch_data

    try:
        chunks = RawMessageChunk.objects.filter(id__in=chunk_ids).order_by("id")
        if not chunks.exists():
            return {"chunk_count": 0, "dupe": DupeTracker().as_dict()}

        deduper = PreLLMDedupe()
        tracker = DupeTracker()

        texts_to_extract = []
        map_idx_to_chunk = {}
        processed_chunks = []
        duplicate_local_chunks = []

        # 1) local in-chat dedupe
        for chunk in chunks:
            if deduper.should_keep(chunk.raw_text):
                map_idx_to_chunk[len(texts_to_extract)] = chunk
                texts_to_extract.append(chunk.raw_text)
            else:
                chunk.status = "DUPLICATE_LOCAL"
                duplicate_local_chunks.append(chunk)
                tracker.add_in_chat(chunk.raw_text)

        if not texts_to_extract:
            return {"chunk_count": len(chunk_ids), "dupe": tracker.as_dict()}

        try:
            log.info("Batch %s: Extracting %s texts", batch_idx, len(texts_to_extract))
            results = extract_listings_from_batch(texts_to_extract)
        except Exception as e:
            log.error("Batch %s LLM Failed: %s", batch_idx, e)
            chunks.update(status="ERROR")
            return {"chunk_count": 0, "dupe": tracker.as_dict()}

        listing_buffer = []
        listing_candidates = []
        batch_seen_hashes = set()

        for res in results:
            if isinstance(res, dict):
                try:
                    res = BatchItemResult(**res)
                except Exception as schema_err:
                    log.warning("Batch %s: dropped invalid result payload (%s)", batch_idx, schema_err)
                    continue

            idx = getattr(res, "message_index", None)
            if idx is None:
                continue
            chunk = map_idx_to_chunk.get(idx)
            if not chunk:
                continue

            chunk.status = "PROCESSED"
            chunk.split_into = len(res.listings)
            processed_chunks.append(chunk)

            if res.is_irrelevant or not res.listings:
                continue

            for listing_data in res.listings:
                tracker.add_candidate()
                text_for_hash = listing_data.cleaned_text
                composite_key = f"{chunk.sender}|{listing_data.location}|{listing_data.transaction_type}|{text_for_hash[:200]}"
                composite_hash = _generate_dedupe_hash(
                    text_for_hash,
                    chunk.sender,
                    listing_data.location,
                    listing_data.transaction_type,
                    listing_data.listing_intent,
                )
                legacy_hash = _generate_legacy_dedupe_hash(text_for_hash, chunk.sender)

                # 2) in-batch dedupe
                if composite_hash in batch_seen_hashes:
                    tracker.add_in_batch(composite_hash)
                    continue
                batch_seen_hashes.add(composite_hash)

                # 3) in-db dedupe
                vector_text = f"{listing_data.cleaned_text} {listing_data.location} {listing_data.transaction_type}"

                lc = ListingChunk(
                    raw_chunk=chunk,
                    text=vector_text,
                    date_seen=chunk.message_start,
                    composite_key=composite_key,
                    composite_hash=composite_hash,
                    metadata=listing_data.model_dump(exclude={"cleaned_text"}),
                    intent="LISTING" if listing_data.listing_intent == "OFFER" else "REQUIREMENT",
                    category=listing_data.property_type,
                    transaction_type=listing_data.transaction_type,
                    confidence=0.9,
                    status="ACTIVE",
                )
                listing_candidates.append(
                    {
                        "composite_hash": composite_hash,
                        "legacy_hash": legacy_hash,
                        "listing_chunk": lc,
                    }
                )

        if duplicate_local_chunks:
            RawMessageChunk.objects.filter(id__in=[c.id for c in duplicate_local_chunks]).update(status="DUPLICATE_LOCAL")

        if processed_chunks:
            RawMessageChunk.objects.bulk_update(processed_chunks, ["status", "split_into"], batch_size=500)

        if listing_candidates:
            all_hashes = {entry["composite_hash"] for entry in listing_candidates}
            all_hashes.update(entry["legacy_hash"] for entry in listing_candidates)
            existing_hashes = set(
                ListingChunk.objects.filter(composite_hash__in=all_hashes).values_list("composite_hash", flat=True)
            )

            for entry in listing_candidates:
                if entry["composite_hash"] in existing_hashes or entry["legacy_hash"] in existing_hashes:
                    tracker.add_in_db(entry["composite_hash"])
                    continue
                listing_buffer.append(entry["listing_chunk"])

        if listing_buffer:
            with transaction.atomic():
                hashes = [lc.composite_hash for lc in listing_buffer]
                existing_before = set(ListingChunk.objects.filter(composite_hash__in=hashes).values_list("composite_hash", flat=True))

                ListingChunk.objects.bulk_create(listing_buffer, ignore_conflicts=True)

                saved_chunks = list(ListingChunk.objects.filter(composite_hash__in=hashes))
                inserted_hashes = {lc.composite_hash for lc in saved_chunks} - existing_before
                tracker.add_inserted(len(inserted_hashes))

                log.info("Batch %s: Syncing %s listings", batch_idx, len(saved_chunks))

                if get_batch_embeddings and saved_chunks:
                    try:
                        texts = [c.text for c in saved_chunks]
                        embeddings = get_batch_embeddings(texts)
                        embedding_objs = []

                        for lc, vec in zip(saved_chunks, embeddings):
                            if vec:
                                embedding_objs.append(
                                    EmbeddingRecord(
                                        listing_chunk=lc,
                                        embedding_vector=vec,
                                        vector_db_id=str(lc.id),
                                        embedded_at=timezone.now(),
                                    )
                                )

                        if embedding_objs:
                            EmbeddingRecord.objects.bulk_create(embedding_objs, ignore_conflicts=True)

                            qdrant_rows = []
                            for lc, vec in zip(saved_chunks, embeddings):
                                if not vec:
                                    continue
                                qdrant_rows.append(
                                    {
                                        "listing_chunk_id": lc.id,
                                        "text": lc.text,
                                        "metadata": lc.metadata or {},
                                        "vector": vec,
                                    }
                                )
                            asyncio.run(aupsert_listing_embeddings(qdrant_rows))

                    except Exception as e:
                        log.error("Batch %s Embedding Error: %s", batch_idx, e)

        return {"chunk_count": len(chunk_ids), "dupe": tracker.as_dict()}

    except Exception as e:
        log.error("Thread worker failed: %s", e)
        RawMessageChunk.objects.filter(id__in=chunk_ids).update(status="ERROR")
        return {"chunk_count": 0, "dupe": DupeTracker().as_dict()}
    finally:
        if manage_db_connection:
            try:
                connection.close()
            except Exception:
                close_old_connections()


def _process_file_in_background_sync(raw_file_id: int):
    """
    Orchestrator: Chunk -> ThreadPool
    """
    raw_file = None
    try:
        raw_file = RawFile.objects.get(pk=raw_file_id)
        # REMOVED strict path check: file_path = getattr(raw_file.file, "path", None)
        
        file_tracker = DupeTracker()

        cache.set(_runtime_log_key(raw_file_id), [], timeout=24 * 3600)
        _store_dedupe_stats(raw_file_id, file_tracker)

        log.info("Starting Multi-Threaded Pipeline for File %s", raw_file_id)
        _append_runtime_log(raw_file_id, "info", "Starting processing pipeline")

        if _cancel_requested(raw_file_id):
            raw_file.status = "CANCELLED"
            raw_file.notes = "Cancelled by user"
            raw_file.process_finished_at = timezone.now()
            raw_file.save(update_fields=["status", "notes", "process_finished_at"])
            cache.set(f"progress:{raw_file_id}", 0, timeout=3600)
            _append_runtime_log(raw_file_id, "warning", "Upload cancelled before processing")
            return

        # We trust the file exists in storage if we can fetch the model.
        # If the actual content is missing, the open() call below will catch it.
        
        raw_file.status = "PROCESSING"
        raw_file.process_started_at = timezone.now()
        raw_file.save()
        cache.set(f"progress:{raw_file_id}", 1, timeout=3600)
        log.info("raw_file_id=%s status=PROCESSING progress=1", raw_file_id)
        _append_runtime_log(raw_file_id, "info", "Status moved to PROCESSING")

        chunk_buffer = []
        all_chunk_ids = []
        parser_used = "rust"

        try:
            for msg_data in iter_rust_parsed_messages(raw_file):
                if _cancel_requested(raw_file_id):
                    raise RuntimeError("CANCELLED BY USER")

                rt = msg_data.get("cleaned_text") or ""
                if not rt or len(rt) < 15 or JUNK_RE.search(rt) or not KEYWORDS_RE.search(rt):
                    continue

                parsed_dt = _parse_whatsapp_datetime(msg_data.get("date_raw"), msg_data.get("time_raw"))
                ts = None
                if parsed_dt:
                    try:
                        ts = timezone.make_aware(parsed_dt)
                    except Exception:
                        ts = None

                chunk_buffer.append(
                    RawMessageChunk(
                        rawfile=raw_file,
                        message_start=ts,
                        sender=msg_data.get("sender"),
                        raw_text=(msg_data.get("raw_text") or "")[:10000],
                        cleaned_text=rt,
                        status="PENDING",
                        user=getattr(raw_file, "owner", None),
                    )
                )

                if len(chunk_buffer) >= BATCH_SIZE:
                    objs = RawMessageChunk.objects.bulk_create(chunk_buffer)
                    all_chunk_ids.extend([o.id for o in objs])
                    chunk_buffer = []
        except FileNotFoundError:
            parser_used = "python-fallback"
            _append_runtime_log(raw_file_id, "warning", f"Rust parser binary not found at {RUST_PARSER_BIN}. Using Python fallback parser.")
            f = None
            should_close = False
            if hasattr(raw_file.file, "open"):
                raw_file.file.open("rb")
                f = raw_file.file
                should_close = True
            elif hasattr(raw_file.file, "path") and os.path.exists(raw_file.file.path):
                f = open(raw_file.file.path, "rb")
                should_close = True
            else:
                raise FileNotFoundError("Could not open file: Storage backend does not support direct access.")

            try:
                for msg_data in stream_chat_messages(f):
                    if _cancel_requested(raw_file_id):
                        raise RuntimeError("CANCELLED BY USER")
                    rt = msg_data["text"]
                    if not rt or len(rt) < 15 or JUNK_RE.search(rt) or not KEYWORDS_RE.search(rt):
                        continue

                    ts = None
                    if msg_data["timestamp"]:
                        try:
                            ts = timezone.make_aware(msg_data["timestamp"])
                        except Exception:
                            ts = None

                    chunk_buffer.append(
                        RawMessageChunk(
                            rawfile=raw_file,
                            message_start=ts,
                            sender=msg_data["sender"],
                            raw_text=rt,
                            cleaned_text=rt,
                            status="PENDING",
                            user=getattr(raw_file, "owner", None),
                        )
                    )

                    if len(chunk_buffer) >= BATCH_SIZE:
                        objs = RawMessageChunk.objects.bulk_create(chunk_buffer)
                        all_chunk_ids.extend([o.id for o in objs])
                        chunk_buffer = []
            finally:
                if should_close and f:
                    f.close()

            if chunk_buffer:
                objs = RawMessageChunk.objects.bulk_create(chunk_buffer)
                all_chunk_ids.extend([o.id for o in objs])
                chunk_buffer = []

            if rust_rows_seen == 0:
                parser_used = "python-fallback-rust-empty-output"
                _append_runtime_log(raw_file_id, "warning", "Rust parser returned 0 message rows. Falling back to Python parser.")
                raise FileNotFoundError("Rust parser produced zero rows")

            if not all_chunk_ids:
                parser_used = "python-fallback-rust-zero-candidates"
                _append_runtime_log(raw_file_id, "warning", "Rust parser produced rows but 0 candidates after filtering. Falling back to Python parser.")
                raise FileNotFoundError("Rust parser produced zero candidates")
        except FileNotFoundError:
            if parser_used == "rust":
                parser_used = "python-fallback"
                _append_runtime_log(raw_file_id, "warning", f"Rust parser binary not found at {RUST_PARSER_BIN}. Using Python fallback parser.")
            f = None
            should_close = False
            if hasattr(raw_file.file, "open"):
                raw_file.file.open("rb")
                f = raw_file.file
                should_close = True
            elif hasattr(raw_file.file, "path") and os.path.exists(raw_file.file.path):
                f = open(raw_file.file.path, "rb")
                should_close = True
            else:
                raise FileNotFoundError("Could not open file: Storage backend does not support direct access.")

            try:
                for msg_data in stream_chat_messages(f):
                    if _cancel_requested(raw_file_id):
                        raise RuntimeError("CANCELLED BY USER")
                    rt = msg_data["text"]
                    if not rt or len(rt) < 15 or JUNK_RE.search(rt) or not KEYWORDS_RE.search(rt):
                        continue

                    ts = None
                    if msg_data["timestamp"]:
                        try:
                            ts = timezone.make_aware(msg_data["timestamp"])
                        except Exception:
                            ts = None

                    chunk_buffer.append(
                        RawMessageChunk(
                            rawfile=raw_file,
                            message_start=ts,
                            sender=msg_data["sender"],
                            raw_text=rt,
                            cleaned_text=rt,
                            status="PENDING",
                            user=getattr(raw_file, "owner", None),
                        )
                    )

                    if len(chunk_buffer) >= BATCH_SIZE:
                        objs = RawMessageChunk.objects.bulk_create(chunk_buffer)
                        all_chunk_ids.extend([o.id for o in objs])
                        chunk_buffer = []
            finally:
                if should_close and f:
                    f.close()

            if chunk_buffer:
                objs = RawMessageChunk.objects.bulk_create(chunk_buffer)
                all_chunk_ids.extend([o.id for o in objs])

        except (FileNotFoundError, ValueError) as e:
             # Capture missing file errors here specifically
             log.error(f"File access error for {raw_file_id}: {e}")
             raw_file.status = "FAILED"
             raw_file.notes = "File could not be opened (missing or corrupt)."
             raw_file.save()
             _append_runtime_log(raw_file_id, "error", f"File access failed: {e}")
             return
        total_msgs = len(all_chunk_ids)
        _append_runtime_log(raw_file_id, "info", f"Parsing complete: {total_msgs} candidate messages (parser={parser_used})")

        if total_msgs == 0:
            raw_file.status = "COMPLETED"
            raw_file.notes = "No valid messages."
            raw_file.save()
            cache.set(f"progress:{raw_file_id}", 100)
            _append_runtime_log(raw_file_id, "warning", "No valid messages found in file")
            return

        llm_batches = []
        for i in range(0, total_msgs, LLM_BATCH_SIZE):
            batch_ids = all_chunk_ids[i : i + LLM_BATCH_SIZE]
            llm_batches.append((i // LLM_BATCH_SIZE, batch_ids, raw_file_id))

        processed_count = 0
        _append_runtime_log(raw_file_id, "info", f"Dispatching {len(llm_batches)} extraction batches")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {executor.submit(lambda payload: asyncio.run(process_single_llm_batch_async(payload)), b): b for b in llm_batches}

            for future in as_completed(future_map):
                if _cancel_requested(raw_file_id):
                    raise RuntimeError("CANCELLED BY USER")
                batch_idx = future_map[future][0]
                try:
                    batch_result = future.result(timeout=300)
                    processed_count += batch_result.get("chunk_count", len(future_map[future][1]))
                    file_tracker.merge(DupeTracker.from_dict(batch_result.get("dupe")))
                    _store_dedupe_stats(raw_file_id, file_tracker)

                    prog = int((processed_count / total_msgs) * 98)
                    if prog < 1:
                        prog = 1
                    cache.set(f"progress:{raw_file_id}", prog, timeout=3600)
                    _append_runtime_log(raw_file_id, "info", f"Batch {batch_idx} completed ({prog}%)")
                except TimeoutError:
                    log.error("Batch TIMED OUT - Possible API or DB hang. Skipping batch.")
                    _append_runtime_log(raw_file_id, "error", f"Batch {batch_idx} timed out")
                except Exception as exc:
                    log.error("Batch Exception: %s", exc)
                    _append_runtime_log(raw_file_id, "error", f"Batch {batch_idx} failed: {exc}")

        raw_file.status = "COMPLETED"
        raw_file.processed = True
        raw_file.notes = f"Processed {total_msgs} messages."
        raw_file.process_finished_at = timezone.now()
        raw_file.save()
        cache.set(f"progress:{raw_file_id}", 100, timeout=3600)
        _store_dedupe_stats(raw_file_id, file_tracker)

        summary = file_tracker.summary().strip().splitlines()
        for line in summary:
            _append_runtime_log(raw_file_id, "info", line)

        _append_runtime_log(raw_file_id, "info", "Processing finished")
        _remove_from_active_processing(raw_file_id)

    except Exception as e:
        if raw_file is None:
            log.exception("Orchestrator failed before rawfile lookup for raw_file_id=%s", raw_file_id)
            return

        if str(e) == "CANCELLED BY USER" or _cancel_requested(raw_file_id):
            log.warning("Orchestrator cancelled for raw_file_id=%s", raw_file_id)
            raw_file.status = "CANCELLED"
            raw_file.notes = "Cancelled by user"
            raw_file.process_finished_at = timezone.now()
            raw_file.save(update_fields=["status", "notes", "process_finished_at"])
            cache.set(f"progress:{raw_file_id}", 0, timeout=3600)
            _append_runtime_log(raw_file_id, "warning", "Processing cancelled by user")
            _remove_from_active_processing(raw_file_id)
            return

        log.exception("Orchestrator Error: %s", e)
        raw_file.status = "FAILED"
        raw_file.notes = str(e)
        raw_file.process_finished_at = timezone.now()
        raw_file.save()
        cache.set(f"progress:{raw_file_id}", 0)
        _append_runtime_log(raw_file_id, "error", f"Pipeline failed: {e}")
        _remove_from_active_processing(raw_file_id)

async def process_single_llm_batch_async(batch_data):
    batch_idx, chunk_ids, raw_file_id = batch_data
    cache_key = _async_cache_key("batch", f"{raw_file_id}:{batch_idx}:{len(chunk_ids)}")

    async def _compute():
        return await asyncio.to_thread(process_single_llm_batch, batch_data)

    return await _aget_or_set_pipeline_cache(cache_key, 120, _compute)


async def process_file_in_background_async(raw_file_id: int):
    cache_key = _async_cache_key("file", str(raw_file_id))

    async def _compute():
        await asyncio.to_thread(_process_file_in_background_sync, raw_file_id)
        return {"raw_file_id": raw_file_id, "status": "completed"}

    return await _aget_or_set_pipeline_cache(cache_key, 30, _compute)


def process_file_in_background(raw_file_id: int):
    return asyncio.run(process_file_in_background_async(raw_file_id))
