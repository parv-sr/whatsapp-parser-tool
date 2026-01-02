# Updated apps/preprocessing/extractor.py
import logging
import os
from typing import List, Optional, Literal
from django.conf import settings
import asyncio
import tiktoken
import multiprocessing
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# Initialize AsyncOpenAI Client
# Make sure OPENAI_API_KEY is set in your settings.py or .env
client = AsyncOpenAI(api_key=getattr(settings, "OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY")))

# Token encoder for GPT-4o-mini (cl100k_base)
ENCODING = tiktoken.get_encoding("cl100k_base")

MAX_INPUT_TOKENS = 100_000  # Conservative buffer for 128K context (prompt + output)
MAX_OUTPUT_TOKENS = 4000    # Per call; adjust based on expected listings

# Dynamic concurrency based on CPU cores (over-subscribe for I/O)
NUM_CORES = multiprocessing.cpu_count()
MAX_CONCURRENT_CALLS = NUM_CORES * 8  # e.g., 16 for 2 cores; tune if rate-limited

# --- Pydantic Schemas (The Contract) --- (Unchanged)
class PropertyListing(BaseModel):
    """
    Represents a single, cleaned real estate listing extracted from a message.
    """
    # 1. The "Cleaned Text" for Embeddings & Dedupe
    cleaned_text: str = Field(
        ..., 
        description="A single, concise sentence summarizing the listing. Remove emojis, agent names, and fluff. Example: '2 BHK fully furnished flat for rent in Bandra West, price 85k.'"
    )

    # 2. Core Classification
    listing_type: Literal['RENT', 'SALE', 'REQUIREMENT', 'UNKNOWN'] = Field(..., description="Is this for rent, sale, or a requirement?")
    property_type: Literal['RESIDENTIAL', 'COMMERCIAL', 'PLOT', 'UNKNOWN'] = Field(..., description="Type of property.")

    # 3. Metadata Fields (Normalized)
    location: str = Field(..., description="Specific locality (e.g., 'Pali Hill', 'BKC'). Do not include 'Mumbai'.")
    building_name: Optional[str] = Field(None, description="Name of the building, project, or society.")
    
    bhk: Optional[float] = Field(None, description="Number of bedrooms. Use 0.5 for RK/Studio.")
    sqft: Optional[int] = Field(None, description="Carpet area in square feet.")
    
    # Price should be an integer (e.g., 15000000 for 1.5 Cr)
    price: Optional[int] = Field(None, description="Total price or rent in INR. Normalize '1.5 Cr' to 15000000.")
    
    furnishing: Optional[Literal['FURNISHED', 'SEMI-FURNISHED', 'UNFURNISHED']] = Field(None)
    parking: Optional[int] = Field(None, description="Number of car parks.")
    
    features: List[str] = Field(default_factory=list, description="Key amenities (e.g., 'Sea View', 'Balcony', 'Terrace').")
    contact_numbers: List[str] = Field(default_factory=list, description="Extracted phone numbers.")

class ExtractionResult(BaseModel):
    listings: List[PropertyListing] = Field(default_factory=list, description="List of properties found in the message.")
    is_irrelevant: bool = Field(False, description="Set to True if the message is just conversation, admin alerts, or spam.")

class BatchItemResult(BaseModel):
    """
    Holds the extraction result for a single message within a batch.
    """
    message_index: int = Field(..., description="The index ID of the message provided in the prompt.")
    listings: List[PropertyListing] = Field(default_factory=list)
    is_irrelevant: bool = Field(False)

class BatchExtractionResult(BaseModel):
    results: List[BatchItemResult]

# --- Token Counting Helper ---
def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))

# --- Async Single Extraction ---
async def _async_extract_single(message_text: str, idx: int, semaphore: asyncio.Semaphore) -> BatchItemResult:
    async with semaphore:
        # Truncate if too large (rare for single WhatsApp msg, but handles multi-listing)
        tokens = count_tokens(message_text)
        if tokens > MAX_INPUT_TOKENS:
            message_text = ENCODING.decode(ENCODING.encode(message_text)[:MAX_INPUT_TOKENS - 1000])  # Buffer for prompt
            log.warning(f"Truncated message {idx} from {tokens} to ~{MAX_INPUT_TOKENS} tokens")

        try:
            completion = await client.beta.chat.completions.parse(
                model="gpt-4o-mini", 
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are an expert real estate data parser for Mumbai. "
                            "Your goal is to extract structured property listings from raw WhatsApp messages. "
                            "1. If a message contains multiple properties, split them into distinct objects. "
                            "2. Ignore 'forwarded' tags, timestamps, and emojis. "
                            "3. Normalize all prices to integers (1.5 Cr -> 15000000). "
                            "4. Generate a clean, factual summary sentence for 'cleaned_text' that captures all key details."
                        )
                    },
                    {"role": "user", "content": f"Message ID {idx}: {message_text}"},
                ],
                response_format=BatchItemResult.model_json_schema(),  # Enforce per-message schema
                temperature=0.0,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            result = completion.choices[0].message.parsed
            if result.is_irrelevant:
                return BatchItemResult(message_index=idx, listings=[], is_irrelevant=True)
            return result
        except Exception as e:
            log.error(f"Async extraction failed for message {idx}: {e}")
            return BatchItemResult(message_index=idx, listings=[], is_irrelevant=True)

# --- Main Async Batch Extraction ---
async def async_extract_listings_from_batch(messages: List[str]) -> List[BatchItemResult]:
    """
    Async batch processes a list of raw message strings in parallel.
    Returns a list of results corresponding to the input order.
    Handles token limits by truncating large messages.
    """
    if not messages:
        return []

    # Dynamic high concurrency: Semaphore based on CPU cores
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

    # Create parallel tasks for each message (parallelism within batch)
    tasks = [_async_extract_single(msg, i, semaphore) for i, msg in enumerate(messages)]
    
    # Gather results (non-blocking)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions in results
    processed_results = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            log.error(f"Task failed for message {i}: {res}")
            processed_results.append(BatchItemResult(message_index=i, listings=[], is_irrelevant=True))
        else:
            processed_results.append(res)
    
    return processed_results

# --- Sync Wrapper for Backward Compatibility ---
def extract_listings_from_batch(messages: List[str]) -> List[BatchItemResult]:
    """
    Sync wrapper: Runs the async batch via asyncio.run().
    Use this in threaded contexts.
    """
    if not messages:
        return []
    
    # Check total batch tokens (optional: split if >100K total, but per-msg truncation handles most)
    total_tokens = sum(count_tokens(msg) for msg in messages)
    if total_tokens > MAX_INPUT_TOKENS * len(messages):  # Rough check
        log.warning(f"Batch total tokens {total_tokens} exceeds safe limit; processing individually")
    
    try:
        return asyncio.run(async_extract_listings_from_batch(messages))
    except Exception as e:
        log.error(f"Sync wrapper failed: {e}")
        return [BatchItemResult(message_index=i, listings=[], is_irrelevant=True) for i in range(len(messages))]