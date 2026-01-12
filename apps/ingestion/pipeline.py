# Updated apps/ingestion/pipeline.py
import re
import hashlib
import logging
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

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

# Updated Extractor (now with async support)
from apps.preprocessing.extractor import extract_listings_from_batch  # Sync wrapper

# Vectoriser
try:
    from apps.embeddings.vectoriser import get_batch_embeddings
except ImportError:
    get_batch_embeddings = None

log = logging.getLogger(__name__)

# --- Configurable constants (Dynamic for max utilization) ---
BATCH_SIZE = 1000  # Database write batch size (unchanged)

# Dynamic based on CPU cores (over-subscribe for I/O-bound work)
NUM_CORES = multiprocessing.cpu_count()
LLM_BATCH_SIZE = 4
MAX_WORKERS = 4
EMBEDDING_DB_FIELD = "embedding_vector"
VECTOR_DEDUPE_DISTANCE_THRESHOLD = 0.05

log.info(f"Pipeline initialized: {NUM_CORES} cores detected -> LLM_BATCH_SIZE={LLM_BATCH_SIZE}, MAX_WORKERS={MAX_WORKERS}")

# --- 1. Strict WhatsApp Splitting Regex --- (Unchanged)
MSG_START_RE = re.compile(
    r"^\[(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s+(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]\s+(?:~?\s?)(?P<sender>[^:]+):\s+",
    re.MULTILINE
)

# --- 2. Gatekeeper Regexes --- (Unchanged)
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
        
        # Normalize newlines and remove NUL
        line = line.replace("\r\n", "\n").replace("\r", "\n").replace("\0", "")  # Clean NUL
        
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
    text_body = full_text[match.end():].strip().replace("\0", "")  # Extra NUL clean

    # Parse Timestamp (unchanged)
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

# --- THREAD WORKER: Processes a single batch of IDs (Now with Async LLM) ---
def process_single_llm_batch(batch_data):
    """
    Executed by ThreadPoolExecutor.
    Processes a small batch of raw texts (extraction -> dedupe -> embedding -> save).
    Now uses async extraction via sync wrapper for seamless integration.
    """
    batch_idx, chunk_ids, raw_file_id = batch_data
    
    # Explicitly close DB connection at end of thread execution to be safe with Django
    # But for now we rely on Django's auto-management.
    
    try:
        # Fetch objects (must refetch in thread)
        chunks = RawMessageChunk.objects.filter(id__in=chunk_ids)
        if not chunks.exists():
            return 0

        deduper = PreLLMDedupe()
        tracker = DupeTracker() # Note: Tracker is local to thread here
        
        texts_to_extract = []
        map_idx_to_chunk = {}

        # 1. Local Dedupe
        for chunk in chunks:
            if deduper.should_keep(chunk.raw_text):
                map_idx_to_chunk[len(texts_to_extract)] = chunk
                texts_to_extract.append(chunk.raw_text)
            else:
                chunk.status = "DUPLICATE_LOCAL"
                chunk.save(update_fields=["status"])
                # Log to tracker and console
                tracker.add_in_chat(chunk.raw_text) 
                log.info(f"[IN-CHAT DUPE] Skipped duplicate message from {chunk.sender}")

        if not texts_to_extract:
            log.info(tracker.summary())
            return len(chunk_ids)

        # 2. Extract (Async via Sync Wrapper - Handles Parallelism Internally)
        try:
            log.info(f"Batch {batch_idx}: Extracting {len(texts_to_extract)} texts")
            for i, txt in enumerate(texts_to_extract):
                log.debug(f"Text {i}: {txt[:200]}...")

            results = extract_listings_from_batch(texts_to_extract)
        except Exception as e:
            log.error(f"Batch {batch_idx} LLM Failed: {e}")
            # Mark chunks as error
            for chunk in chunks:
                chunk.status = "ERROR"
                chunk.save(update_fields=["status"])
            return 0

        # 3. Process Results (For each extracted listing)
        listing_buffer = []
        embedding_buffer = []  # If embeddings enabled

        for res in results:
            idx = res.message_index
            chunk = map_idx_to_chunk.get(idx)
            if not chunk:
                continue

            chunk.status = "PROCESSED"
            chunk.split_into = len(res.listings)
            chunk.save(update_fields=["status", "split_into"])

            if res.is_irrelevant or not res.listings:
                continue

            for listing_data in res.listings:
                # Prepare composite keys
                text_for_hash = listing_data.cleaned_text
                composite_key = f"{chunk.sender}|{listing_data.location}|{listing_data.listing_type}|{text_for_hash[:200]}"
                composite_hash = _generate_dedupe_hash(text_for_hash, chunk.sender)

                # Check hash dedupe
                existing_lc = ListingChunk.objects.filter(composite_hash=composite_hash).first()
                if existing_lc:
                    existing_lc.last_seen = timezone.now()
                    existing_lc.save(update_fields=["last_seen"])
                    
                    # Log DB duplicate
                    tracker.add_in_db(composite_hash)
                    log.info(f"[DB DUPE - HASH] Skipped existing listing {existing_lc.id} (hash={composite_hash[:8]})")
                    continue  # Skip create

                # Vector prep text (for embedding)
                vector_text = f"{listing_data.cleaned_text} {listing_data.location} {listing_data.listing_type}"

                # Create ListingChunk
                lc = ListingChunk(
                    raw_chunk=chunk,
                    text=vector_text,
                    date_seen=chunk.message_start,
                    composite_key=composite_key,
                    composite_hash=composite_hash,
                    metadata=listing_data.model_dump(exclude={"cleaned_text"}),  # Exclude text (stored separately)
                    intent="LISTING" if listing_data.listing_type in ["RENT", "SALE"] else "REQUIREMENT",
                    category=listing_data.property_type,
                    transaction_type=listing_data.listing_type,
                    confidence=0.9,  # Default; could use LLM confidence if added
                    status="ACTIVE"
                )
                listing_buffer.append(lc)

        # Bulk create listings
        if listing_buffer:
            with transaction.atomic():
                ListingChunk.objects.bulk_create(listing_buffer, ignore_conflicts=True)
                log.info(f"Batch {batch_idx}: Saved {len(listing_buffer)} new listings to DB.")

                # 4. Embeddings (If available; batch them)
                if get_batch_embeddings:
                    unique_texts = [lc.text for lc in listing_buffer]
                    try:
                        embeddings = get_batch_embeddings(unique_texts)
                        for i, (lc, vec) in enumerate(zip(listing_buffer, embeddings)):
                            if vec:
                                EmbeddingRecord.objects.create(
                                    listing_chunk=lc,
                                    embedding_vector=vec,
                                    vector_db_id=str(lc.id),
                                    embedded_at=timezone.now()
                                )
                    except Exception as e:
                        log.error(f"Batch embedding failed: {e}")

                # Vector Dedupe (Post-Embed; Query for similars)
                for lc in listing_buffer:
                    er = getattr(lc, 'embedding', None)  # Access related EmbeddingRecord safely
                    # Note: Since we just created EmbeddingRecord manually above, the reverse relation 
                    # might not be populated on 'lc' object instance without refresh.
                    # We can fetch the embedding we just created if needed, 
                    # but typically vector dedupe is done BEFORE insert or via a background job.
                    # Here we do a quick check if we just inserted a duplicate vector.
                    
                    # NOTE: Re-fetching for correctness in this linear flow
                    er = EmbeddingRecord.objects.filter(listing_chunk=lc).first()

                    if not er or not er.embedding_vector:
                        continue

                    # Raw SQL for cosine similarity check
                    vec_literal = _vector_to_pg_literal(er.embedding_vector)
                    sql = f"""
                        SELECT id, listing_chunk_id, {vec_literal} <=> embedding_vector AS distance
                        FROM preprocessing_embeddingrecord
                        WHERE embedding_vector IS NOT NULL AND id != %s
                        ORDER BY distance
                        LIMIT 1
                    """
                    with connection.cursor() as cur:
                        cur.execute(sql, [er.id])
                        row = cur.fetchone()
                        if row and row[2] < VECTOR_DEDUPE_DISTANCE_THRESHOLD:
                            # Similar exists; mark as duplicate
                            lc.status = "STALE"
                            lc.save(update_fields=["status"])
                            
                            tracker.add_in_db(f"vector:{lc.id}")
                            log.info(f"[DB DUPE - VECTOR] Marked {lc.id} as duplicate (dist={row[2]:.3f})")

        # Update chunk count (post-processing)
        processed_chunks = len([r for r in results if r.listings])
        
        # Log Summary for this batch
        log.info(tracker.summary())
        
        return processed_chunks  # Return actual processed, not total

    except Exception as e:
        log.error(f"Thread worker failed: {e}")
        # Mark all as error
        RawMessageChunk.objects.filter(id__in=chunk_ids).update(status="ERROR")
        return 0
    finally:
        # Ensure connection is clean
        connection.close()


# --- ORCHESTRATOR --- (Minor tweaks for higher throughput)
def process_file_in_background(raw_file_id: int):
    """
    1. Streams file -> Creates DB Chunks (Bulk)
    2. Spawns Threads -> Processes Chunks in parallel (Now with async inside)
    3. Updates Status -> COMPLETED
    """
    try:
        raw_file = RawFile.objects.get(pk=raw_file_id)
        
        file_path = getattr(raw_file.file, "path", None)
        log.info("Processing RawFile id=%s (Async-Threaded Mode, max_workers=%d)", raw_file_id, MAX_WORKERS)

        if not file_path or not os.path.exists(file_path):
            raw_file.status = "FAILED"
            raw_file.notes = "File not found"
            raw_file.save()
            return

        raw_file.status = "PROCESSING"
        raw_file.process_started_at = timezone.now()
        raw_file.save()

        cache.set(f"progress:{raw_file_id}", 1)

        # --- STEP 1: STREAM & BUFFER TO DB --- (Unchanged)
        chunk_buffer = []
        all_chunk_ids = []
        
        f = None
        should_close = False
        try:
            if hasattr(raw_file.file, "open"):
                f = raw_file.file.open("rb")
                should_close = True
            else:
                f = open(file_path, "rb")
                should_close = True

            for msg_data in stream_chat_messages(f):
                rt = msg_data["text"]
                # Gatekeeper
                if not rt or len(rt) < 15 or JUNK_RE.search(rt) or not KEYWORDS_RE.search(rt):
                    continue
                
                ts = None
                if msg_data["timestamp"]:
                    try:
                        ts = timezone.make_aware(msg_data["timestamp"])
                    except Exception:
                        pass
                
                chunk_buffer.append(RawMessageChunk(
                    rawfile=raw_file,
                    message_start=ts,
                    sender=msg_data["sender"],
                    raw_text=rt,
                    cleaned_text=rt,
                    status="PENDING",
                    user=getattr(raw_file, "owner", None)
                ))

                if len(chunk_buffer) >= BATCH_SIZE:
                    objs = RawMessageChunk.objects.bulk_create(chunk_buffer)
                    all_chunk_ids.extend([o.id for o in objs])
                    chunk_buffer = []
            
            # Flush remainder
            if chunk_buffer:
                objs = RawMessageChunk.objects.bulk_create(chunk_buffer)
                all_chunk_ids.extend([o.id for o in objs])

            log.info(f"Orchestrator: Streamed & buffered {len(all_chunk_ids)} messages from file {raw_file_id}")

        finally:
            if should_close and f:
                f.close()

        # --- STEP 2: THREADED PROCESSING (Dynamic Workers, Larger Batches) ---
        total_msgs = len(all_chunk_ids)
        if total_msgs == 0:
            raw_file.status = "COMPLETED"
            raw_file.notes = "No valid messages found."
            raw_file.save()
            return

        # Create batches for threads
        llm_batches = []
        for i in range(0, total_msgs, LLM_BATCH_SIZE):
            batch_ids = all_chunk_ids[i : i + LLM_BATCH_SIZE]
            llm_batches.append((i // LLM_BATCH_SIZE, batch_ids, raw_file_id))  # Batch index

        processed_count = 0
        
        # Parallel Execution (Dynamic max workers for compute utilization)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all
            future_map = {executor.submit(process_single_llm_batch, b): b for b in llm_batches}
            
            for future in as_completed(future_map):
                cnt = future.result()
                processed_count += cnt
                
                # Update Progress (1-99%)
                # We cap at 99 so 100 is reserved for completion
                if total_msgs > 0:
                    prog = int((processed_count / total_msgs) * 98) + 1
                    cache.set(f"progress:{raw_file_id}", prog)

       
        # --- STEP 3: FINALIZE --- (Unchanged)
        # If we reach here, threads are done.
        raw_file.status = "COMPLETED"
        raw_file.process_finished_at = timezone.now()
        raw_file.processed = True
        raw_file.notes = f"Processed {processed_count} messages successfully."
        raw_file.save()
        
        # Force 100%
        cache.set(f"progress:{raw_file_id}", 100)
        log.info(f"Orchestrator: File {raw_file_id} completed. {processed_count} valid listings processed.")

        active = cache.get("processing_files", [])
        if raw_file_id in active:
            active.remove(raw_file_id)
            cache.set("processing_files", active)

        cache.delete(f"progress:{raw_file_id}")

    except Exception as e:
        log.exception("Orchestrator Failed: %s", e)
        if "raw_file" in locals():
            raw_file.status = "FAILED"
            raw_file.notes = str(e)
            raw_file.save()

        active = cache.get("processing_files", [])
        if raw_file_id in active:
            active.remove(raw_file_id)
            cache.set("processing_files", active)
        cache.delete(f"progress:{raw_file_id}")