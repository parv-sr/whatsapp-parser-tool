import re
import hashlib
import logging
import os
from rapidfuzz import fuzz
from datetime import datetime
from django.utils import timezone
from django.db import transaction
from django.core.cache import cache

# Models
from apps.ingestion.models import RawFile, RawMessageChunk
from apps.ingestion.dedupe.pre_llm_dedupe import PreLLMDedupe
from apps.preprocessing.models import ListingChunk, EmbeddingRecord

# Extractor
from apps.preprocessing.extractor import extract_listings_from_text

# Vectoriser (Handling the import safely)
try:
    from apps.embeddings.vectoriser import generate_embedding_and_push
except ImportError:
    generate_embedding_and_push = None

log = logging.getLogger(__name__)

# --- 1. Strict WhatsApp Splitting Regex ---
MSG_START_RE = re.compile(
    r"^\[(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s+(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]\s+(?:~?\s?)(?P<sender>[^:]+):\s+",
    re.MULTILINE
)

# --- 2. Gatekeeper Regexes ---
KEYWORDS_RE = re.compile(r"(bhk|rent|sale|lease|price|cr\b|lacs?|sqft|carpet|furnished|office|shop|flat|buy|sell|want)", re.IGNORECASE)
JUNK_RE = re.compile(r"(security code changed|waiting for this message|message was deleted|encrypted|joined|left|added|null|media omitted)", re.IGNORECASE)

def _generate_dedupe_hash(cleaned_text: str, sender: str, date_obj) -> str:
    """
    Deterministic hash for deduplication.
    """
    date_str = date_obj.strftime("%Y-%m-%d")
    # Normalize: remove spaces, lowercase
    norm_text = re.sub(r"\s+", "", cleaned_text).lower()
    norm_sender = re.sub(r"\s+", "", sender).lower()
    
    raw_key = f"{norm_text}|{norm_sender}|{date_str}"
    return hashlib.sha256(raw_key.encode('utf-8')).hexdigest()

def parse_raw_chat_file(content: str):
    """
    Splits content by timestamp headers.
    """
    messages = []
    current_msg = None
    
    # Normalize newlines to avoid platform issues
    lines = content.replace('\r\n', '\n').replace('\r', '\n').split('\n')

    for line in lines:
        match = MSG_START_RE.match(line)
        if match:
            if current_msg: messages.append(current_msg)
            
            # Parse Header
            raw_date, raw_time = match.group('date'), match.group('time')
            sender = match.group('sender').strip()
            text_body = line[match.end():].strip()
            
            # Parse Timestamp
            try:
                # Handle 2-digit vs 4-digit year
                fmt = "%d/%m/%y, %I:%M:%S %p" if len(raw_date.split('/')[-1]) == 2 else "%d/%m/%Y, %I:%M:%S %p"
                # Handle missing seconds in some exports
                if len(raw_time.split(':')) == 2: fmt = fmt.replace(":%S", "")
                ts_str = f"{raw_date}, {raw_time}"
                dt = datetime.strptime(ts_str, fmt)
            except ValueError:
                dt = datetime.now()

            current_msg = {'timestamp': dt, 'sender': sender, 'text': text_body}
        else:
            if current_msg:
                current_msg['text'] += "\n" + line

    if current_msg: messages.append(current_msg)
    return messages

# --- 3. Main Processing Logic ---

def process_file_in_background(raw_file_id: int):
    try:
        raw_file = RawFile.objects.get(pk=raw_file_id)

        file_path = raw_file.file.path
        log.info(f"Looking for file at: {file_path}")
        
        #RENDER DEBUG
        if not os.path.exists(file_path):
            log.error(f"CRITICAL: File missing! Directory listing of {os.path.dirname(file_path)}:")
            try:
                log.error(os.listdir(os.path.dirname(file_path)))
            except Exception as e:
                log.error(f"Could not list directory: {e}")
            raise FileNotFoundError(f"File not found at {file_path}")

        raw_file.status = "PROCESSING"
        raw_file.process_started_at = timezone.now()
        raw_file.save()

        cache.set(f"progress:{raw_file_id}", 5)


# your real code doing reading...


        # A. Ingestion (Read File)
        try:
            if hasattr(raw_file.file, 'open'):
                raw_file.file.open('rb')
                content = raw_file.file.read().decode('utf-8', errors='replace')
                raw_file.file.close()
            else:
                with open(raw_file.file.path, 'rb') as f:
                    content = f.read().decode('utf-8', errors='replace')
        except Exception as e:
            raw_file.status = "FAILED"
            raw_file.notes = f"Read Error: {str(e)}"
            raw_file.save()
            return

        parsed_msgs = parse_raw_chat_file(content)
        cache.set(f"progress:{raw_file_id}", 20)
        deduper = PreLLMDedupe()
        total_processed = 0
        listings_created = 0

        cache.set(f"progress:{raw_file_id}", 40)

        for msg in parsed_msgs:
            raw_text = msg['text']
            
            # B. Gatekeeper Check
            # Skip if too short or looks like system junk
            if len(raw_text) < 20 or JUNK_RE.search(raw_text):
                continue 
            
            # Skip if it doesn't contain ANY real estate keywords
            if not KEYWORDS_RE.search(raw_text):
                continue 

            # Create RawMessageChunk
            # FIX: Using exact field names from your models.py
            aware_dt = timezone.make_aware(msg['timestamp'])
            
            raw_chunk = RawMessageChunk.objects.create(
                rawfile=raw_file,           # model field: rawfile
                message_start=aware_dt,     # model field: message_start
                sender=msg['sender'],       # model field: sender
                raw_text=raw_text,          # model field: raw_text
                cleaned_text=raw_text,      # model field: cleaned_text (initially same)
                status="PROCESSED",          # model field: status
                user=raw_file.owner
            )

            # Pre-LLM dedupe
            if not deduper.should_keep(raw_text):
                print("Listing dropped")
                continue

            # C. LLM Extraction
            extracted_listings = extract_listings_from_text(raw_text)

            for item in extracted_listings:
                new_text = item.cleaned_text.strip().lower()
                candidates = ListingChunk.objects.filter(
                    status="ACTIVE",
                ).values("id", "text", "last_seen")

                duplicate_found = False

                for c in candidates:
                    score = fuzz.ratio(new_text, c["text"].lower())
                    if score >= 95:
                        ListingChunk.objects.filter(id=c["id"]).update(last_seen=timezone.now())
                        duplicate_found = True
                        break

                if duplicate_found:
                    print("Dupe found in main DB")
                    continue
            
            # Update split count
            raw_chunk.split_into = len(extracted_listings)
            raw_chunk.save()

            for item in extracted_listings:
                # D. Deduplication
                composite_hash = _generate_dedupe_hash(item.cleaned_text, msg['sender'], aware_dt)
                
                existing_listing = ListingChunk.objects.filter(composite_hash=composite_hash).first()
                
                if existing_listing:
                    existing_listing.last_seen = timezone.now()
                    existing_listing.save()
                    continue

                # E. Mapping LLM Output to DB Choices
                # LLM returns: RENT, SALE, REQUIREMENT, UNKNOWN
                # DB Intent: LISTING, REQUIREMENT, UNKNOWN
                # DB Txn: RENT, SALE, LEASE, UNKNOWN
                
                db_intent = "UNKNOWN"
                db_txn_type = "UNKNOWN"
                
                if item.listing_type == "REQUIREMENT":
                    db_intent = "REQUIREMENT"
                    db_txn_type = "UNKNOWN" # Requirements might not specify transaction type explicitly yet
                elif item.listing_type in ["RENT", "SALE"]:
                    db_intent = "LISTING"
                    db_txn_type = item.listing_type
                
                # Map Category (PLOT -> LAND)
                db_category = item.property_type
                if db_category == "PLOT":
                    db_category = "LAND"

                # Create ListingChunk
                listing = ListingChunk.objects.create(
                    raw_chunk=raw_chunk,
                    text=item.cleaned_text,
                    date_seen=aware_dt,
                    metadata=item.model_dump(), # Saves the full JSON structure
                    
                    intent=db_intent,
                    category=db_category,
                    transaction_type=db_txn_type,
                    
                    composite_hash=composite_hash,
                    # composite_key can be blank or populated if you wish
                    status="ACTIVE"
                )

                # F. Embeddings
                if generate_embedding_and_push:
                    try:
                        # Richer context for embedding
                        vector_text = f"{item.cleaned_text} | {item.location} | {item.listing_type}"
                        
                        emb_info = generate_embedding_and_push(
                            text=vector_text, 
                            metadata=item.model_dump(), 
                            listing_id=listing.id
                        )
                        
                        EmbeddingRecord.objects.create(
                            listing_chunk=listing,
                            embedding_vector=emb_info["vector_cache"],
                            vector_db_id=emb_info.get("vector_id"),
                            vector_index_name=emb_info.get("index_name"),
                            embedded_at=timezone.now()
                        )
                    except Exception as vec_err:
                        log.error(f"Vector embedding failed for listing {listing.id}: {vec_err}")

                listings_created += 1
            cache.set(f"progress:{raw_file_id}", 70)
            total_processed += 1

        # Finalize RawFile
        raw_file.status = "COMPLETED"
        cache.set(f"progress:{raw_file_id}", 100)
        raw_file.process_finished_at = timezone.now()
        raw_file.notes = f"Processed {total_processed} messages. Created {listings_created} listings."
        raw_file.processed = True
        raw_file.save()

    except Exception as e:
        log.exception(f"Pipeline Critical Failure for file {raw_file_id}: {e}")
        if 'raw_file' in locals():
            raw_file.status = "FAILED"
            raw_file.notes = f"Critical Error: {str(e)}"
            raw_file.save()