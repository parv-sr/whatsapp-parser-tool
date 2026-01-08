# Updated apps/preprocessing/extractor.py (Fix exceptions & 400: Switch to JSON mode)
import logging
import os
from typing import List, Optional, Literal
from django.conf import settings
import asyncio
import tiktoken
import multiprocessing
import json  # For JSON parsing
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, after_log

log = logging.getLogger(__name__)

# Initialize AsyncOpenAI Client
client = AsyncOpenAI(api_key=getattr(settings, "OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY")))

# Token encoder for GPT-4o-mini (cl100k_base)
ENCODING = tiktoken.get_encoding("cl100k_base")

MAX_INPUT_TOKENS = 100_000
MAX_OUTPUT_TOKENS = 4000

# Conservative concurrency
NUM_CORES = multiprocessing.cpu_count()
MAX_CONCURRENT_CALLS = 2
BATCH_TOKEN_LIMIT = 50_000

# --- Pydantic Schemas (Unchanged) ---
class PropertyListing(BaseModel):
    cleaned_text: str = Field(..., description="A single, concise sentence summarizing the listing. Remove emojis, agent names, and fluff. Example: '2 BHK fully furnished flat for rent in Bandra West, price 85k.'")
    listing_type: Literal['RENT', 'SALE', 'REQUIREMENT', 'UNKNOWN'] = Field(..., description="Is this for rent, sale, or a requirement?")
    property_type: Literal['RESIDENTIAL', 'COMMERCIAL', 'PLOT', 'UNKNOWN'] = Field(..., description="Type of property.")
    location: str = Field(..., description="Specific locality (e.g., 'Pali Hill', 'BKC'). Do not include 'Mumbai'.")
    building_name: Optional[str] = Field(None, description="Name of the building, project, or society.")
    bhk: Optional[float] = Field(None, description="Number of bedrooms. Use 0.5 for RK/Studio.")
    sqft: Optional[int] = Field(None, description="Carpet area in square feet.")
    price: Optional[int] = Field(None, description="Total price or rent in INR. Normalize '1.5 Cr' to 15000000.")
    furnishing: Optional[Literal['FURNISHED', 'SEMI-FURNISHED', 'UNFURNISHED']] = Field(None)
    parking: Optional[int] = Field(None, description="Number of car parks.")
    features: List[str] = Field(default_factory=list, description="Key amenities (e.g., 'Sea View', 'Balcony', 'Terrace').")
    contact_numbers: List[str] = Field(default_factory=list, description="Extracted phone numbers.")

class BatchItemResult(BaseModel):
    message_index: int = Field(..., description="The index ID of the message provided in the prompt.")
    listings: List[PropertyListing] = Field(default_factory=list)
    is_irrelevant: bool = Field(False)

# --- Token Counting Helper ---
def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))

# --- Retriable Async Single Extraction (Use JSON mode to avoid 400 on parse) ---
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(openai.RateLimitError),  # Use RateLimitError for quota/rate
    after=after_log(log, logging.WARNING, "Retrying extraction after rate limit: attempt={context.attempt_number}")
)
async def _async_extract_single(message_text: str, idx: int, semaphore: asyncio.Semaphore) -> BatchItemResult:
    async with semaphore:
        # Truncate if too large
        tokens = count_tokens(message_text)
        if tokens > MAX_INPUT_TOKENS:
            message_text = ENCODING.decode(ENCODING.encode(message_text)[:MAX_INPUT_TOKENS - 1000])
            log.warning(f"Truncated message {idx} from {tokens} to ~{MAX_INPUT_TOKENS} tokens")

        system_prompt = (
            "You are an expert real estate data parser for Mumbai. "
            "Your goal is to extract structured property listings from raw WhatsApp messages. "
            "1. If a message contains multiple properties, split them into distinct objects. "
            "2. Ignore 'forwarded' tags, timestamps, and emojis. "
            "3. Normalize all prices to integers (1.5 Cr -> 15000000). "
            "4. Generate a clean, factual summary sentence for 'cleaned_text' that captures all key details. "
            "Output ONLY valid JSON matching this schema: " + json.dumps(BatchItemResult.model_json_schema())
        )

        try:
            completion = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Message ID {idx}: {message_text}"},
                ],
                response_format={"type": "json_object"},  # Fallback to JSON mode
                temperature=0.0,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            content = completion.choices[0].message.content
            parsed = json.loads(content)
            return BatchItemResult(**parsed)
        except openai.RateLimitError as e:
            log.error(f"Rate limit for message {idx}: {e}")
            raise
        except openai.BadRequestError as e:  # Handle 400 specifically
            log.error(f"Bad request for message {idx}: {e}. Prompt: {system_prompt[:200]}...")
            return BatchItemResult(message_index=idx, listings=[], is_irrelevant=True)
        except Exception as e:
            log.error(f"Extraction failed for message {idx}: {e}")
            return BatchItemResult(message_index=idx, listings=[], is_irrelevant=True)

# --- Main Async Batch Extraction (Unchanged) ---
async def async_extract_listings_from_batch(messages: List[str]) -> List[BatchItemResult]:
    if not messages:
        return []

    # Throttle: Check total batch tokens
    total_tokens = sum(count_tokens(msg) for msg in messages)
    if total_tokens > BATCH_TOKEN_LIMIT:
        log.warning(f"Batch {total_tokens} tokens > {BATCH_TOKEN_LIMIT}; splitting into smaller async sub-batches")
        sub_batches = []
        current_sub = []
        current_tokens = 0
        for msg in messages:
            msg_tokens = count_tokens(msg)
            if current_tokens + msg_tokens > BATCH_TOKEN_LIMIT:
                if current_sub:
                    sub_batches.append(current_sub)
                current_sub = [msg]
                current_tokens = msg_tokens
            else:
                current_sub.append(msg)
                current_tokens += msg_tokens
        if current_sub:
            sub_batches.append(current_sub)
        
        # Process sub-batches sequentially with pause
        all_results = []
        for sub_idx, sub_msgs in enumerate(sub_batches):
            sub_sem = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
            sub_tasks = [_async_extract_single(msg, i, sub_sem) for i, msg in enumerate(sub_msgs)]
            sub_results = await asyncio.gather(*sub_tasks)
            all_results.extend(sub_results)
            if sub_idx < len(sub_batches) - 1:
                await asyncio.sleep(5)  # Longer pause for TPM recovery
        return all_results

    # Normal
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
    tasks = [_async_extract_single(msg, i, semaphore) for i, msg in enumerate(messages)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            log.error(f"Task failed for message {i}: {res}")
            processed_results.append(BatchItemResult(message_index=i, listings=[], is_irrelevant=True))
        else:
            processed_results.append(res)
    
    return processed_results

# --- Sync Wrapper ---
def extract_listings_from_batch(messages: List[str]) -> List[BatchItemResult]:
    if not messages:
        return []
    try:
        return asyncio.run(async_extract_listings_from_batch(messages))
    except Exception as e:
        log.error(f"Sync wrapper failed: {e}")
        return [BatchItemResult(message_index=i, listings=[], is_irrelevant=True) for i in range(len(messages))]