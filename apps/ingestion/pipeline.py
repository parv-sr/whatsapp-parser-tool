import re
import hashlib
import logging
import os
import io
from datetime import datetime
from django.utils import timezone
from django.db import connection
from django.db import transaction
from django.core.cache import cache
from pgvector.django import CosineDistance 

# Models
from apps.ingestion.models import RawFile, RawMessageChunk
from apps.ingestion.dedupe.pre_llm_dedupe import PreLLMDedupe
from apps.ingestion.dedupe.dupe_tracker import DupeTracker
from apps.preprocessing.models import ListingChunk, EmbeddingRecord

# Extractor
from apps.preprocessing.extractor import extract_listings_from_batch

# Vectoriser
try:
    from apps.embeddings.vectoriser import get_batch_embeddings
except ImportError:
    get_batch_embeddings = None

log = logging.getLogger(__name__)

# --- Configurable constants ---
# Increased to 1000 to minimize Redis commands (10k limit on Upstash)
BATCH_SIZE = 1000 
# Chunk size for sending to LLM (still 50 to avoid HTTP timeouts/large payloads)
LLM_BATCH_SIZE = 12
EMBEDDING_DB_FIELD = "embedding_vector"
VECTOR_DEDUPE_DISTANCE_THRESHOLD = 0.05

# --- 1. Strict WhatsApp Splitting Regex ---
MSG_START_RE = re.compile(
    r"^\[(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s+(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]\s+(?:~?\s?)(?P<sender>[^:]+):\s+",
    re.MULTILINE
)

# --- 2. Gatekeeper Regexes ---
KEYWORDS_RE = re.compile(r"(bhk|rent|sale|lease|price|cr\b|lacs?|sqft|carpet|furnished|office|shop|flat|buy|sell|want)", re.IGNORECASE)
JUNK_RE = re.compile(r"(security code changed|waiting for this message|message was deleted|encrypted|joined|left|added|null|media omitted)", re.IGNORECASE)


def _generate_dedupe_hash(cleaned_text: str, sender: str) -> str:
    """
    Deterministic hash for deduplication.
    """
    norm_text = re.sub(r"\s+", "", (cleaned_text or "")).lower()
    norm_sender = re.sub(r"\s+", "", (sender or "")).lower()
    raw_key = f"{norm_text}|{norm_sender}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _vector_to_pg_literal(vec):
    inner = ",".join(f"{float(x):.6f}" for x in vec)
    return f"'[{inner}]'::vector"


def stream_chat_messages(file_obj):
    """
    Generator that reads a file line-by-line and yields structured messages.
    Does NOT load the whole file into memory.
    """
    buffer = []
    current_match = None

    for line in file_obj:
        # Decode if bytes
        if isinstance(line, bytes):
            line = line.decode('utf-8', errors='replace')
        
        # Normalize newlines
        line = line.replace("\r\n", "\n").replace("\r", "\n")
        
        match = MSG_START_RE.match(line)
        if match:
            # Found new message start -> Yield previous message if exists
            if current_match and buffer:
                yield _parse_buffered_message(current_match, buffer)
            
            current_match = match
            buffer = [line]
        else:
            # Continuation of previous message
            if buffer:
                buffer.append(line)

    # Yield the last one
    if current_match and buffer:
        yield _parse_buffered_message(current_match, buffer)


def _parse_buffered_message(match, buffer):
    raw_date, raw_time = match.group("date"), match.group("time")
    sender = match.group("sender").strip()
    full_text = "".join(buffer)
    # Body is text after the header match
    text_body = full_text[match.end():].strip()

    # Parse Timestamp
    dt = None
    try:
        # heuristics on year / seconds
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


# --- ORCHESTRATOR: Reads Stream, Batches, Dispatches ---
def process_file_in_background(raw_file_id: int):
    """
    Acts as the 'Streamer'. Reads file line-by-line, creates DB rows, 
    and dispatches batches to Celery workers.
    """
    # Import here to avoid circular dependency
    from apps.ingestion.tasks import process_batch_task

    try:
        raw_file = RawFile.objects.get(pk=raw_file_id)
        
        file_path = getattr(raw_file.file, "path", None)
        log.info("Processing RawFile id=%s (Streaming Mode)", raw_file_id)

        if not file_path or not os.path.exists(file_path):
            raw_file.status = "FAILED"
            raw_file.notes = "File not found"
            raw_file.save()
            return

        raw_file.status = "PROCESSING"
        raw_file.process_started_at = timezone.now()
        raw_file.save()

        # Initialize progress
        cache.set(f"progress:{raw_file_id}", 1)

        batch_buffer = []
        total_dispatched = 0

        # Open file as a stream
        with open(file_path, "rb") as f:
            for msg_data in stream_chat_messages(f):
                raw_text = msg_data["text"]
                
                # Fast pre-filter (Gatekeeper)
                if not raw_text or len(raw_text) < 20:
                    continue
                if JUNK_RE.search(raw_text) or not KEYWORDS_RE.search(raw_text):
                    continue

                # Prepare object
                aware_dt = None
                if msg_data["timestamp"]:
                    try:
                        aware_dt = timezone.make_aware(msg_data["timestamp"])
                    except Exception:
                        pass

                obj = RawMessageChunk(
                    rawfile=raw_file,
                    message_start=aware_dt,
                    sender=msg_data["sender"],
                    raw_text=raw_text, # storing body only
                    cleaned_text=raw_text,
                    status="PENDING", # Mark pending for worker
                    user=getattr(raw_file, "owner", None),
                )
                batch_buffer.append(obj)

                # Dispatch if buffer full
                if len(batch_buffer) >= BATCH_SIZE:
                    _dispatch_buffer(batch_buffer, process_batch_task)
                    total_dispatched += len(batch_buffer)
                    batch_buffer = []
                    
                    # Update rough progress
                    cache.incr(f"progress:{raw_file_id}", 1)

            # Flush remainder
            if batch_buffer:
                _dispatch_buffer(batch_buffer, process_batch_task)
                total_dispatched += len(batch_buffer)

        log.info("Finished streaming file %s. Dispatched %s messages.", raw_file_id, total_dispatched)
        
        raw_file.notes = f"Ingested {total_dispatched} messages for background processing."
        raw_file.processed = True 
        raw_file.save()

    except Exception as e:
        log.exception("Orchestrator Failed: %s", e)
        if "raw_file" in locals():
            raw_file.status = "FAILED"
            raw_file.notes = str(e)
            raw_file.save()

def _dispatch_buffer(buffer_list, task_func):
    """
    Bulk creates RawMessageChunks and calls the worker task.
    """
    if not buffer_list:
        return
    
    # 1. Bulk Create
    created_objs = RawMessageChunk.objects.bulk_create(buffer_list)
    
    # 2. Extract IDs
    ids = [obj.id for obj in created_objs]
    
    # 3. Fire & Forget Task (Pass IDs)
    task_func.delay(ids)


# --- WORKER: Processes a Chunk of IDs ---
def process_raw_chunk_batch(chunk_ids):
    """
    Worker logic:
    1. Reads batch of RawMessageChunks from DB
    2. Local Dedupe (PreLLMDedupe)
    3. Calls LLM Extraction (Batched)
    4. DB Dedupe (Check Composite Hash)
    5. Selective Embedding & Save
    """
    try:
        # Fetch fresh data
        chunks = RawMessageChunk.objects.filter(id__in=chunk_ids)
        if not chunks.exists():
            return

        # Initialize Tracking Tools
        deduper = PreLLMDedupe()
        tracker = DupeTracker()

        texts_to_extract = []
        map_index_to_raw_chunk = {}

        # 1. Local Filter & Dedupe
        for chunk in chunks:
            raw_text = chunk.raw_text
            # SimHash / Exact Match Check
            if deduper.should_keep(raw_text):
                map_index_to_raw_chunk[len(texts_to_extract)] = chunk
                texts_to_extract.append(raw_text)
            else:
                log.info("[IN-CHAT DUPE] Raw message duplicate detected (sender=%s)", chunk.sender)
                tracker.add_in_chat(raw_text)
                chunk.status = "DUPLICATE_LOCAL"
                chunk.save(update_fields=["status"])

        if not texts_to_extract:
            log.info(tracker.summary())
            return

        # 2. Extraction (LLM) - Process in sub-batches of 50 to avoid timeouts
        extraction_results = []
        
        # Simple sub-batching loop
        for i in range(0, len(texts_to_extract), LLM_BATCH_SIZE):
            sub_texts = texts_to_extract[i : i + LLM_BATCH_SIZE]
            sub_map_start_index = i
            
            try:
                batch_res = extract_listings_from_batch(sub_texts)
                
                # Map back to raw_chunk using offset
                for res in batch_res:
                    # Adjust index to global list
                    relative_idx = getattr(res, "message_index", 0) # Assumes extractor uses relative 0-N index
                    
                    # Safety check
                    # The extractor returns indices 0..49 for the sub-batch.
                    # We map that to texts_to_extract[i + relative_idx]
                    global_idx = sub_map_start_index + relative_idx
                    
                    if global_idx in map_index_to_raw_chunk:
                        extraction_results.append({
                            "raw_chunk": map_index_to_raw_chunk[global_idx],
                            "result": res
                        })
            except Exception as e:
                log.error("LLM Sub-batch failed: %s", e)
                continue

        # 3. Processing & Selective DB Writing
        items_to_embed = [] # list of dicts
        
        for entry in extraction_results:
            r_chunk = entry["raw_chunk"]
            res = entry["result"]
            
            listings = getattr(res, "listings", [])
            
            # Update chunk status
            r_chunk.split_into = len(listings)
            r_chunk.status = "PROCESSED"
            r_chunk.save(update_fields=["split_into", "status"])

            for item in listings:
                # Generate Hash
                cleaned_text = getattr(item, "cleaned_text", "")
                sender = r_chunk.sender or ""
                comp_hash = _generate_dedupe_hash(cleaned_text, sender)

                # --- DB DEDUPE (CRITICAL FOR FREE TIER) ---
                # Check if hash exists. If so, update last_seen and SKIP embedding/creation.
                # This saves DB rows and Embedding API costs.
                if ListingChunk.objects.filter(composite_hash=comp_hash).exists():
                    ListingChunk.objects.filter(composite_hash=comp_hash).update(last_seen=timezone.now())
                    log.info("[DB DUPE - HASH] Duplicate listing found for hash=%s...", comp_hash[:10])
                    tracker.add_in_db(comp_hash)
                    continue
                
                # Prepare for embedding
                listing_type = getattr(item, "listing_type", "UNKNOWN")
                location = getattr(item, "location", "")
                
                vector_text = f"{cleaned_text} | {location} | {listing_type}"
                
                items_to_embed.append({
                    "raw_chunk": r_chunk,
                    "item": item,
                    "hash": comp_hash,
                    "vector_text": vector_text
                })

        # 4. Generate Embeddings & Save (Only for NEW unique items)
        if items_to_embed and get_batch_embeddings:
            # We can send all unique texts to OpenAI
            vec_inputs = [x["vector_text"] for x in items_to_embed]
            try:
                vectors = get_batch_embeddings(vec_inputs)
            except Exception:
                vectors = []

            for idx, data in enumerate(items_to_embed):
                vector = vectors[idx] if idx < len(vectors) else None
                item_data = data["item"]
                
                # Create Listing
                try:
                    listing_type = getattr(item_data, "listing_type", "UNKNOWN")
                    property_type = getattr(item_data, "property_type", "UNKNOWN")
                    
                    db_intent = "UNKNOWN"
                    if listing_type == "REQUIREMENT":
                        db_intent = "REQUIREMENT"
                    elif listing_type in ("RENT", "SALE"):
                        db_intent = "LISTING"
                    
                    db_cat = "LAND" if property_type == "PLOT" else (property_type or "UNKNOWN")

                    lc = ListingChunk.objects.create(
                        raw_chunk=data["raw_chunk"],
                        text=getattr(item_data, "cleaned_text", ""),
                        date_seen=data["raw_chunk"].message_start,
                        metadata=item_data.model_dump() if hasattr(item_data, "model_dump") else {},
                        intent=db_intent,
                        category=db_cat,
                        transaction_type=listing_type if listing_type != "REQUIREMENT" else "UNKNOWN",
                        composite_hash=data["hash"],
                        status="ACTIVE"
                    )

                    log.info("Created ListingChunk id=%s (sender=%s, hash=%s...)",
                                lc.id,
                                data["raw_chunk"].sender,
                                data["hash"][:8]
                    )

                    if vector:
                        EmbeddingRecord.objects.create(
                            listing_chunk=lc,
                            embedding_vector=vector, # PGVector field
                            vector_db_id=str(lc.id),
                            embedded_at=timezone.now()
                        )
                        log.info("   ↳ Saved embedding for listing id=%s", lc.id)
                    else:
                        log.info("   ↳ Skipped embedding (no vector)")

                except Exception as e:
                    log.error("Error saving listing: %s", e)
        
        # Log Dupetracker summary for this batch
        log.info(tracker.summary())

    except Exception as e:
        log.exception("Worker Batch Failed: %s", e)