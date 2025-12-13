import re
import hashlib
import logging
import os
from datetime import datetime
from django.utils import timezone
from django.db import connection
from django.db import transaction
from django.core.cache import cache
from pgvector.django import CosineDistance  # kept for potential ORM use / compatibility

# Models
from apps.ingestion.models import RawFile, RawMessageChunk
from apps.ingestion.dedupe.pre_llm_dedupe import PreLLMDedupe
from apps.ingestion.dedupe.dupe_tracker import DupeTracker
from apps.preprocessing.models import ListingChunk, EmbeddingRecord

# Extractor
from apps.preprocessing.extractor import extract_listings_from_batch

# Vectoriser (Handling the import safely)
try:
    from apps.embeddings.vectoriser import get_batch_embeddings
except ImportError:
    get_batch_embeddings = None  # safe fallback

log = logging.getLogger(__name__)

# --- Configurable constants ---
BATCH_SIZE = 7  # conservative
EMBEDDING_DB_FIELD = "embedding_vector"  # change here if your model uses a different column
VECTOR_DEDUPE_DISTANCE_THRESHOLD = 0.05

# --- 1. Strict WhatsApp Splitting Regex ---
MSG_START_RE = re.compile(
    r"^\[(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s+(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]\s+(?:~?\s?)(?P<sender>[^:]+):\s+",
    re.MULTILINE
)

# --- 2. Gatekeeper Regexes ---
KEYWORDS_RE = re.compile(r"(bhk|rent|sale|lease|price|cr\b|lacs?|sqft|carpet|furnished|office|shop|flat|buy|sell|want)", re.IGNORECASE)
JUNK_RE = re.compile(r"(security code changed|waiting for this message|message was deleted|encrypted|joined|left|added|null|media omitted)", re.IGNORECASE)


# --- Debug helper: rough token estimation ---
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text):
        return len(_enc.encode(text or ""))
except Exception:
    def count_tokens(text):
        return len((text or "").split())




def _generate_dedupe_hash(cleaned_text: str, sender: str) -> str:
    """
    Deterministic hash for deduplication.
    Note: date intentionally omitted to catch reposts across days.
    Include a few normalized fields to reduce accidental collisions.
    """
    norm_text = re.sub(r"\s+", "", (cleaned_text or "")).lower()
    norm_sender = re.sub(r"\s+", "", (sender or "")).lower()
    raw_key = f"{norm_text}|{norm_sender}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def chunk_list(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def parse_raw_chat_file(content: str):
    """
    Splits content by timestamp headers.
    """
    messages = []
    current_msg = None

    # Normalize newlines to avoid platform issues
    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    for line in lines:
        match = MSG_START_RE.match(line)
        if match:
            if current_msg:
                messages.append(current_msg)

            raw_date, raw_time = match.group("date"), match.group("time")
            sender = match.group("sender").strip()
            text_body = line[match.end():].strip()

            # Parse Timestamp
            dt = None
            try:
                # heuristics on year / seconds
                if len(raw_date.split("/")[-1]) == 2:
                    # two-digit year
                    fmt_base = "%d/%m/%y, %I:%M:%S %p"
                else:
                    fmt_base = "%d/%m/%Y, %I:%M:%S %p"

                # remove :%S if seconds aren't present
                if len(raw_time.split(":")) == 2:
                    fmt = fmt_base.replace(":%S", "")
                else:
                    fmt = fmt_base

                ts_str = f"{raw_date}, {raw_time}"
                dt = datetime.strptime(ts_str, fmt)
            except Exception:
                dt = None

            current_msg = {"timestamp": dt, "sender": sender, "text": text_body}
        else:
            if current_msg:
                current_msg["text"] += "\n" + line

    if current_msg:
        messages.append(current_msg)
    return messages


# --- Helper to produce pgvector literal used in raw SQL queries (keeps formatting consistent) ---
def _vector_to_pg_literal(vec):
    inner = ",".join(f"{float(x):.6f}" for x in vec)
    return f"'[{inner}]'::vector"


# --- Main Processing Logic ---
def process_file_in_background(raw_file_id: int):
    try:
        raw_file = RawFile.objects.get(pk=raw_file_id)
        tracker = DupeTracker()

        file_path = getattr(raw_file.file, "path", None)
        log.info("Processing RawFile id=%s path=%s", raw_file_id, file_path)

        if not file_path or not os.path.exists(file_path):
            log.error("File missing for RawFile id=%s path=%s", raw_file_id, file_path)
            raw_file.status = "FAILED"
            raw_file.notes = f"File not found at {file_path}"
            raw_file.save()
            return

        raw_file.status = "PROCESSING"
        raw_file.process_started_at = timezone.now()
        raw_file.save()

        cache.set(f"progress:{raw_file_id}", 5)

        # A. Ingestion (Read File)
        try:
            if hasattr(raw_file.file, "open"):
                raw_file.file.open("rb")
                content = raw_file.file.read().decode("utf-8", errors="replace")
                raw_file.file.close()
            else:
                with open(file_path, "rb") as f:
                    content = f.read().decode("utf-8", errors="replace")
        except Exception as e:
            raw_file.status = "FAILED"
            raw_file.notes = f"Read Error: {str(e)}"
            raw_file.save()
            return

        parsed_msgs = parse_raw_chat_file(content)
        deduper = PreLLMDedupe()
        total_processed = 0
        listings_created = 0

        # Process in batches
        for batch_idx, batch_msgs in enumerate(chunk_list(parsed_msgs, BATCH_SIZE)):

            log.info("\n==============================")
            log.info("Starting Batch %s", batch_idx + 1)
            log.info("==============================")

            est_tokens = sum(count_tokens(m.get("text", "")) for m in batch_msgs)
            log.info("Batch %s contains %s messages (~%s raw tokens)",
                    batch_idx + 1, len(batch_msgs), est_tokens)

            # Phase A: Local ingestion & filtering
            texts_to_extract = []  # list[str]
            map_index_to_raw_chunk = {}  # index -> RawMessageChunk

            for msg in batch_msgs:
                raw_text = msg.get("text") or ""
                if not raw_text:
                    continue

                if len(raw_text) < 20 or JUNK_RE.search(raw_text) or not KEYWORDS_RE.search(raw_text):
                    continue

                # parse / make aware timestamp if present
                ts = msg.get("timestamp")
                if ts:
                    try:
                        aware_dt = timezone.make_aware(ts)
                    except Exception:
                        aware_dt = None
                else:
                    aware_dt = None

                # Save RawMessageChunk (we keep behaviour same)
                raw_chunk = RawMessageChunk.objects.create(
                    rawfile=raw_file,
                    message_start=aware_dt,
                    sender=msg.get("sender"),
                    raw_text=raw_text,
                    cleaned_text=raw_text,
                    status="PROCESSED",
                    user=getattr(raw_file, "owner", None),
                )

                # pre-LLM dedupe check (cheap)
                try:
                    keep = deduper.should_keep(raw_text)
                except Exception as e:
                    log.warning("PreLLMDedupe failed unexpectedly: %s", e)
                    keep = True

                if keep:
                    map_index_to_raw_chunk[len(texts_to_extract)] = raw_chunk
                    texts_to_extract.append(raw_text)
                else:
                    log.info("[IN-CHAT DUPE] Raw message duplicate detected (sender=%s)", msg.get("sender"))
                    tracker.add_in_chat(raw_text)
                    continue

            if not texts_to_extract:
                continue

            # Phase B: Batch Extraction (LLM) - keep single call for the batch
            try:
                extraction_results = extract_listings_from_batch(texts_to_extract)
                log.info("Extractor completed for Batch %s — received %s results",
                batch_idx + 1, len(extraction_results))
            except Exception as e:
                log.error("Batch extraction failed on Batch %s: %s", batch_idx + 1, e)
                continue

            # Validate extractor output shape and mapping strategy
            # If extractor returns objects with .message_index fields we use them.
            # If not, but returns same-length list, we assume order-preserved mapping.
            # If neither, abort this batch to avoid mis-association.
            use_index_field = False
            if isinstance(extraction_results, (list, tuple)) and extraction_results:
                sample = extraction_results[0]
                if hasattr(sample, "message_index") or (isinstance(sample, dict) and "message_index" in sample):
                    use_index_field = True

            if use_index_field:
                # sanity check indices
                bad_idx = False
                for res in extraction_results:
                    idx = getattr(res, "message_index", res.get("message_index") if isinstance(res, dict) else None)
                    if idx is None or not isinstance(idx, int) or idx < 0 or idx >= len(texts_to_extract):
                        log.error("Extractor returned invalid message_index=%s for batch (size=%d). Aborting batch.", idx, len(texts_to_extract))
                        bad_idx = True
                        break
                if bad_idx:
                    continue  # skip batch to avoid corruption
            else:
                # if extractor produced same-length results but no indices, assume positional mapping
                if len(extraction_results) != len(texts_to_extract):
                    log.warning(
                        "Extractor returned %d results for %d inputs and no indices — skipping batch to avoid misalignment.",
                        len(extraction_results), len(texts_to_extract)
                    )
                    continue
                else:
                    log.debug("Extractor did not return indices but result length matches inputs; mapping by position.")

            # Phase C: Process extraction results -> prepare items for embedding + DB writes
            items_needing_vectors = []  # list of dicts {raw_chunk, item_data, hash, vector_text}
            for res_idx, res in enumerate(extraction_results):
                # resolve raw_chunk
                if use_index_field:
                    idx = getattr(res, "message_index", res.get("message_index") if isinstance(res, dict) else None)
                else:
                    idx = res_idx

                raw_chunk = map_index_to_raw_chunk.get(idx)
                if raw_chunk is None:
                    log.warning("No mapped RawMessageChunk for extractor result idx=%s; skipping result.", idx)
                    continue

                # Normalize result interface: allow dataclass/object or dict
                is_irrelevant = getattr(res, "is_irrelevant", res.get("is_irrelevant") if isinstance(res, dict) else False)
                listings = getattr(res, "listings", res.get("listings") if isinstance(res, dict) else [])

                # Update raw_chunk split count
                try:
                    raw_chunk.split_into = len(listings or [])
                    raw_chunk.save(update_fields=["split_into"])
                except Exception as e:
                    log.warning("Failed to update split_into for raw_chunk %s: %s", getattr(raw_chunk, "id", None), e)

                if is_irrelevant:
                    continue

                for item in listings or []:
                    # item may be pydantic model or dict
                    cleaned_text = getattr(item, "cleaned_text", item.get("cleaned_text") if isinstance(item, dict) else None)
                    listing_type = getattr(item, "listing_type", item.get("listing_type") if isinstance(item, dict) else None)
                    location = getattr(item, "location", item.get("location") if isinstance(item, dict) else None)

                    composite_hash = _generate_dedupe_hash(cleaned_text or "", raw_chunk.sender or "")

                    # Quick exact-hash dedupe (DB)
                    if ListingChunk.objects.filter(composite_hash=composite_hash).exists():
                        log.info("[DB DUPE - HASH] Duplicate listing found for hash=%s...", composite_hash[:10])
                        tracker.add_in_db(composite_hash)
                        ListingChunk.objects.filter(composite_hash=composite_hash).update(last_seen=timezone.now())
                        continue


                    # prepare vector text, avoiding "None" tokens
                    parts = list(filter(None, [cleaned_text, location, listing_type]))
                    vector_text = " | ".join(parts)

                    items_needing_vectors.append({
                        "raw_chunk": raw_chunk,
                        "item_data": item,
                        "hash": composite_hash,
                        "vector_text": vector_text
                    })

            if not items_needing_vectors:
                # update progress and continue
                total_processed += len(batch_msgs)
                if len(parsed_msgs) > 0:
                    progress = min(95, int(((batch_idx + 1) * BATCH_SIZE / max(1, len(parsed_msgs))) * 100))
                    cache.set(f"progress:{raw_file_id}", progress)
                continue

            # Phase D: Batch Embedding
            vectors = []
            if get_batch_embeddings:
                try:
                    vector_texts = [d["vector_text"] for d in items_needing_vectors]
                    vectors = get_batch_embeddings(vector_texts)
                    # safety: ensure we got a list of same length
                    if not isinstance(vectors, (list, tuple)) or len(vectors) != len(items_needing_vectors):
                        log.error("Embedding API returned unexpected result count: expected %d got %s. Aborting batch save to avoid mismatch.", len(items_needing_vectors), getattr(vectors, "__len__", lambda: "unknown")())
                        continue
                except Exception as e:
                    log.error("Batch embedding failed: %s", e)
                    vectors = []
            else:
                log.warning("Vectoriser not configured; embeddings will be skipped for this batch.")

            # Phase E: Vector dedupe (using raw SQL with pgvector operator) and saving
            for i, data in enumerate(items_needing_vectors):
                vector = vectors[i] if i < len(vectors) else None
                item = data["item_data"]
                raw_chunk = data["raw_chunk"]

                is_dup = False
                if vector:
                    try:
                        # Use raw SQL to leverage index operator <=> (distance)
                        vector_literal = _vector_to_pg_literal(vector)
                        sql = f"""
                            SELECT id, listing_chunk_id, {EMBEDDING_DB_FIELD} <=> {vector_literal} AS distance
                            FROM preprocessing_embeddingrecord
                            WHERE {EMBEDDING_DB_FIELD} IS NOT NULL
                            ORDER BY distance
                            LIMIT 1
                        """
                        with connection.cursor() as cur:
                            cur.execute(sql)
                            row = cur.fetchone()
                        if row:
                            nearest_id, nearest_listing_chunk_id, dist = row[0], row[1], row[2]
                            if dist is not None and float(dist) < VECTOR_DEDUPE_DISTANCE_THRESHOLD:
                                log.info(
                                        "[DB DUPE - VECTOR] Duplicate found via vector similarity (dist=%.4f) -> listing_id=%s", 
                                        float(dist), 
                                        nearest_listing_chunk_id
                                    )
                                tracker.add_in_db(f"vector:{nearest_listing_chunk_id}")
                                # mark duplicate found: update last_seen and skip creation
                                try:
                                    ListingChunk.objects.filter(pk=nearest_listing_chunk_id).update(last_seen=timezone.now())
                                    is_dup = True
                                except Exception:
                                    log.warning("Vector dedupe: failed to update last_seen for listing %s", nearest_listing_chunk_id)
                    except Exception as e:
                        # Don't fail the batch - but log it
                        log.warning("Vector dedupe SQL error: %s", e)

                if is_dup:
                    continue

                # Create ListingChunk
                try:
                    cleaned_text = getattr(item, "cleaned_text", item.get("cleaned_text") if isinstance(item, dict) else "")
                    listing_type = getattr(item, "listing_type", item.get("listing_type") if isinstance(item, dict) else "UNKNOWN")
                    property_type = getattr(item, "property_type", item.get("property_type") if isinstance(item, dict) else "UNKNOWN")

                    db_intent = "UNKNOWN"
                    if listing_type == "REQUIREMENT":
                        db_intent = "REQUIREMENT"
                    elif listing_type in ("RENT", "SALE"):
                        db_intent = "LISTING"

                    db_cat = "LAND" if property_type == "PLOT" else (property_type or "UNKNOWN")

                    listing = ListingChunk.objects.create(
                        raw_chunk=raw_chunk,
                        text=cleaned_text,
                        date_seen=raw_chunk.message_start,
                        metadata=(item.model_dump() if hasattr(item, "model_dump") else (item if isinstance(item, dict) else {})),
                        intent=db_intent,
                        category=db_cat,
                        transaction_type=(listing_type if listing_type != "REQUIREMENT" else "UNKNOWN"),
                        composite_hash=data["hash"],
                        status="ACTIVE"
                    )

                    log.info("Created ListingChunk id=%s (sender=%s, hash=%s...)",
                                listing.id,
                                raw_chunk.sender,
                                data["hash"][:8]
                    )

                except Exception as e:
                    log.error("Failed to create ListingChunk for raw_chunk %s: %s", getattr(raw_chunk, "id", None), e)
                    continue

                # Save EmbeddingRecord if vector exists
                if vector:
                    try:
                        EmbeddingRecord.objects.create(
                            listing_chunk=listing,
                            **{EMBEDDING_DB_FIELD: vector},
                            vector_db_id=str(listing.id),
                            vector_index_name="default",
                            embedded_at=timezone.now()
                        )
                        log.info("   ↳ Saved embedding for listing id=%s", listing.id)
                    except Exception as e:
                        log.error("Failed to save EmbeddingRecord for listing %s: %s", listing.id, e)

                listings_created += 1

                if not vector:
                    log.info("   ↳ Skipped embedding (no vector)")
                
                log.info("Finished Batch %s", batch_idx + 1)
                log.info("--------------------------------------------------")


            # Update progress
            total_processed += len(batch_msgs)
            if len(parsed_msgs) > 0:
                progress = min(95, int(((batch_idx + 1) * BATCH_SIZE / max(1, len(parsed_msgs))) * 100))
                cache.set(f"progress:{raw_file_id}", progress)

        # Finalize
        log.info(tracker.summary())
        raw_file.status = "COMPLETED"
        cache.set(f"progress:{raw_file_id}", 100)
        raw_file.process_finished_at = timezone.now()
        raw_file.notes = f"Processed {total_processed} messages. Created {listings_created} listings."
        raw_file.processed = True
        raw_file.save()

    except Exception as e:
        log.exception("Pipeline Critical Failure for file %s: %s", raw_file_id, e)
        if "raw_file" in locals():
            raw_file.status = "FAILED"
            raw_file.notes = f"Critical Error: {str(e)}"
            raw_file.save()
