# apps/ingestion/pipeline.py
import re
import hashlib
import logging
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import multiprocessing

from django.utils import timezone
from django.db import connection, transaction, close_old_connections
from django.core.cache import cache

# Models
from apps.ingestion.models import RawFile, RawMessageChunk
from apps.ingestion.dedupe.pre_llm_dedupe import PreLLMDedupe
from apps.ingestion.dedupe.dupe_tracker import DupeTracker
from apps.preprocessing.models import ListingChunk, EmbeddingRecord

# Extraction
from apps.preprocessing.extractor import extract_listings_from_batch

# Vectoriser
try:
    from apps.embeddings.vectoriser import get_batch_embeddings
except ImportError:
    get_batch_embeddings = None

log = logging.getLogger(__name__)

# --- CONFIGURATION ---
BATCH_SIZE = 1000
NUM_CORES = multiprocessing.cpu_count()

LLM_BATCH_SIZE = 60  
MAX_WORKERS = 4  
VECTOR_DEDUPE_DISTANCE_THRESHOLD = 0.05

log.info(f"Pipeline initialized: {NUM_CORES} cores -> LLM_BATCH_SIZE={LLM_BATCH_SIZE}, MAX_WORKERS={MAX_WORKERS}")

# --- REGEX ---
MSG_START_RE = re.compile(
    r"^\[(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s+(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]\s+(?:~?\s?)(?P<sender>[^:]+):\s+",
    re.MULTILINE
)
KEYWORDS_RE = re.compile(r"(bhk|rent|sale|lease|price|cr\b|lacs?|sqft|carpet|furnished|office|shop|flat|buy|sell|want)", re.IGNORECASE)
JUNK_RE = re.compile(r"(security code changed|waiting for this message|message was deleted|encrypted|joined|left|added|null|media omitted)", re.IGNORECASE)


def _generate_dedupe_hash(cleaned_text: str, sender: str) -> str:
    norm_text = re.sub(r"\s+", "", (cleaned_text or "")).lower()
    norm_sender = re.sub(r"\s+", "", (sender or "")).lower()
    raw_key = f"{norm_text}|{norm_sender}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

def stream_chat_messages(file_obj):
    buffer = []
    current_match = None
    for line in file_obj:
        if isinstance(line, bytes):
            line = line.decode('utf-8', errors='replace')
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
    # 1. FIX: Close unusable connections inherited from parent
    close_old_connections()
    
    batch_idx, chunk_ids, raw_file_id = batch_data
    
    try:
        # Re-fetch objects safely
        chunks = RawMessageChunk.objects.filter(id__in=chunk_ids)
        if not chunks.exists():
            return 0

        deduper = PreLLMDedupe()
        tracker = DupeTracker()
        
        texts_to_extract = []
        map_idx_to_chunk = {}

        # Local Dedupe
        for chunk in chunks:
            if deduper.should_keep(chunk.raw_text):
                map_idx_to_chunk[len(texts_to_extract)] = chunk
                texts_to_extract.append(chunk.raw_text)
            else:
                chunk.status = "DUPLICATE_LOCAL"
                chunk.save(update_fields=["status"])
                tracker.add_in_chat(chunk.raw_text) 

        if not texts_to_extract:
            return len(chunk_ids)

        # Extract (LLM)
        try:
            log.info(f"Batch {batch_idx}: Extracting {len(texts_to_extract)} texts")
            results = extract_listings_from_batch(texts_to_extract)
        except Exception as e:
            log.error(f"Batch {batch_idx} LLM Failed: {e}")
            chunks.update(status="ERROR")
            return 0

        listing_buffer = []

        # Prepare Data
        for res in results:
            idx = res.message_index
            chunk = map_idx_to_chunk.get(idx)
            if not chunk: continue

            chunk.status = "PROCESSED"
            chunk.split_into = len(res.listings)
            chunk.save(update_fields=["status", "split_into"])

            if res.is_irrelevant or not res.listings:
                continue

            for listing_data in res.listings:
                text_for_hash = listing_data.cleaned_text
                composite_key = f"{chunk.sender}|{listing_data.location}|{listing_data.listing_type}|{text_for_hash[:200]}"
                composite_hash = _generate_dedupe_hash(text_for_hash, chunk.sender)

                # Lightweight check to avoid DB hit if possible
                if ListingChunk.objects.filter(composite_hash=composite_hash).exists():
                    tracker.add_in_db(composite_hash)
                    continue 

                vector_text = f"{listing_data.cleaned_text} {listing_data.location} {listing_data.listing_type}"

                lc = ListingChunk(
                    raw_chunk=chunk,
                    text=vector_text,
                    date_seen=chunk.message_start,
                    composite_key=composite_key,
                    composite_hash=composite_hash,
                    metadata=listing_data.model_dump(exclude={"cleaned_text"}),
                    intent="LISTING" if listing_data.listing_type in ["RENT", "SALE"] else "REQUIREMENT",
                    category=listing_data.property_type,
                    transaction_type=listing_data.listing_type,
                    confidence=0.9,
                    status="ACTIVE"
                )
                listing_buffer.append(lc)

        # Save to DB (Atomic)
        if listing_buffer:
            with transaction.atomic():
                # A. Bulk Create (Fast, ignore conflicts)
                ListingChunk.objects.bulk_create(listing_buffer, ignore_conflicts=True)
                
                # B. Re-fetch IDs (Crucial step)
                hashes = [lc.composite_hash for lc in listing_buffer]
                saved_chunks = list(ListingChunk.objects.filter(composite_hash__in=hashes))
                
                log.info(f"Batch {batch_idx}: Syncing {len(saved_chunks)} listings")

                # C. Embeddings
                if get_batch_embeddings and saved_chunks:
                    try:
                        texts = [c.text for c in saved_chunks]
                        embeddings = get_batch_embeddings(texts)
                        embedding_objs = []
                        
                        for lc, vec in zip(saved_chunks, embeddings):
                            if vec:
                                embedding_objs.append(EmbeddingRecord(
                                    listing_chunk=lc,
                                    embedding_vector=vec,
                                    vector_db_id=str(lc.id),
                                    embedded_at=timezone.now()
                                ))
                        
                        if embedding_objs:
                            EmbeddingRecord.objects.bulk_create(embedding_objs, ignore_conflicts=True)
                            
                    except Exception as e:
                        log.error(f"Batch {batch_idx} Embedding Error: {e}")

        processed_chunks = len([r for r in results if r.listings])
        return processed_chunks

    except Exception as e:
        log.error(f"Thread worker failed: {e}")
        RawMessageChunk.objects.filter(id__in=chunk_ids).update(status="ERROR")
        return 0
    finally:
        # 2. FIX: Force close connection to release it back to pool or destroy it
        connection.close()


def process_file_in_background(raw_file_id: int):
    """
    Orchestrator: Chunk -> ThreadPool
    """
    try:
        raw_file = RawFile.objects.get(pk=raw_file_id)
        file_path = getattr(raw_file.file, "path", None)
        log.info(f"Starting Multi-Threaded Pipeline for File {raw_file_id}")

        if not file_path or not os.path.exists(file_path):
            raw_file.status = "FAILED"
            raw_file.notes = "File not found"
            raw_file.save()
            return

        raw_file.status = "PROCESSING"
        raw_file.process_started_at = timezone.now()
        raw_file.save()
        cache.set(f"progress:{raw_file_id}", 1, timeout=3600)

        # 1. Parsing (Main Thread)
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
                if not rt or len(rt) < 15 or JUNK_RE.search(rt) or not KEYWORDS_RE.search(rt):
                    continue
                
                ts = None
                if msg_data["timestamp"]:
                    try: ts = timezone.make_aware(msg_data["timestamp"])
                    except: pass
                
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
            
            if chunk_buffer:
                objs = RawMessageChunk.objects.bulk_create(chunk_buffer)
                all_chunk_ids.extend([o.id for o in objs])

        finally:
            if should_close and f: f.close()

        total_msgs = len(all_chunk_ids)
        if total_msgs == 0:
            raw_file.status = "COMPLETED"
            raw_file.notes = "No valid messages."
            raw_file.save()
            cache.set(f"progress:{raw_file_id}", 100)
            return

        # 2. Execution
        llm_batches = []
        for i in range(0, total_msgs, LLM_BATCH_SIZE):
            batch_ids = all_chunk_ids[i : i + LLM_BATCH_SIZE]
            llm_batches.append((i // LLM_BATCH_SIZE, batch_ids, raw_file_id))

        processed_count = 0
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {executor.submit(process_single_llm_batch, b): b for b in llm_batches}
            
            for future in as_completed(future_map):
                try:
                    # FIX: Timeout ensures we don't hang forever
                    future.result(timeout=300) 
                    
                    processed_count += len(future_map[future][1])
                    prog = int((processed_count / total_msgs) * 98)
                    if prog < 1: prog = 1
                    cache.set(f"progress:{raw_file_id}", prog, timeout=3600)
                except TimeoutError:
                    log.error("Batch TIMED OUT - Possible API or DB hang. Skipping batch.")
                except Exception as exc:
                    log.error(f"Batch Exception: {exc}")

        # 3. Finish
        raw_file.status = "COMPLETED"
        raw_file.processed = True
        raw_file.notes = f"Processed {total_msgs} messages."
        raw_file.save()
        cache.set(f"progress:{raw_file_id}", 100, timeout=3600)
        
        active = cache.get("processing_files", [])
        if raw_file_id in active:
            active.remove(raw_file_id)
            cache.set("processing_files", active)

    except Exception as e:
        log.exception(f"Orchestrator Error: {e}")
        raw_file.status = "FAILED"
        raw_file.notes = str(e)
        raw_file.save()
        cache.set(f"progress:{raw_file_id}", 0)