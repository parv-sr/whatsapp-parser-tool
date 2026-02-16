# apps/preprocessing/extractor.py
import logging
import os
import json
import asyncio
import tiktoken
import multiprocessing
from typing import List, Optional, Literal
from django.conf import settings
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator  # <--- Added field_validator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, after_log

log = logging.getLogger(__name__)

# Initialize AsyncOpenAI Client
client = AsyncOpenAI(api_key=getattr(settings, "OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY")))

# Token encoder for GPT-4o-mini (cl100k_base)
ENCODING = tiktoken.get_encoding("cl100k_base")

# --- CONFIGURATION ---
# We pack ~15 messages per API call for maximum speed and efficiency.
MESSAGES_PER_PACKET = 15 
MAX_CONCURRENT_PACKETS = 3 # Process 3 packets (45 messages) concurrently per worker thread

# --- Pydantic Schemas ---
class PropertyListing(BaseModel):
    cleaned_text: str = Field(..., description="A single, concise sentence summarizing the listing. Remove emojis, agent names, and fluff. Example: '2 BHK fully furnished flat for rent in Bandra West, price 85k.'")
    listing_type: Literal['RENT', 'LEASE', 'SALE', 'OWNERSHIP', 'REQUIREMENT', 'UNKNOWN'] = Field(..., description="Is this for rent, sale, or a requirement?")
    property_type: Literal['RESIDENTIAL', 'COMMERCIAL', 'PLOT', 'LAND', 'UNKNOWN'] = Field(..., description="Type of property.")
    location: str = Field(..., description="Specific locality (e.g., 'Pali Hill', 'BKC'). Do not include 'Mumbai'.")
    building_name: Optional[str] = Field(None, description="Name of the building, project, or society.")
    bhk: Optional[float] = Field(None, description="Number of bedrooms. Use 0.5 for RK/Studio.")
    sqft: Optional[int] = Field(None, description="Carpet area in square feet.")
    price: Optional[int] = Field(None, description="Total price or rent in INR. Normalize '1.5 Cr' to 15000000.")
    
    # Allowed values
    furnishing: Optional[Literal['FULLY FURNISHED','FURNISHED', 'SEMI-FURNISHED', 'SEMI FURNISHED', 'UNFURNISHED', 'UNKNOWN']] = Field(None)
    
    parking: Optional[int] = Field(None, description="Number of car parks.")
    features: List[str] = Field(default_factory=list, description="Key amenities (e.g., 'Sea View', 'Balcony', 'Terrace').")
    contact_numbers: List[str] = Field(default_factory=list, description="Extracted phone numbers.")

    # --- VALIDATORS TO PREVENT DATA LOSS ---

    @field_validator('listing_type', mode='before')
    @classmethod
    def normalize_listing_type(cls, v):
        if not v:
            return "UNKNOWN"
        s = str(v).upper().strip()
        if s in {"OWNERSHIP", "OWN"}:
            return "OWNERSHIP"
        if "LEASE" in s or "LICENSE" in s:
            return "LEASE"
        if "RENT" in s:
            return "RENT"
        if "SALE" in s or "BUY" in s or "SELL" in s:
            return "SALE"
        if "REQUIRE" in s:
            return "REQUIREMENT"
        return "UNKNOWN"


    @field_validator('location', mode='before')
    @classmethod
    def ensure_location_string(cls, v):
        """Fixes cases where LLM returns null for location."""
        if v is None:
            return "Unknown"
        return str(v)

    @field_validator('furnishing', mode='before')
    @classmethod
    def normalize_furnishing(cls, v):
        """Maps fuzzy LLM outputs to strict allowed values."""
        if not v:
            return None
        
        # Normalize string
        s = str(v).upper().strip()
        
        # Map common variations seen in logs
        if "FULLY" in s or "FURNISHED" == s:
            return "FURNISHED"
        if "SEMI" in s:
            return "SEMI-FURNISHED"
        if "EMPTY" in s or "NOT" in s or "UNFURNISHED" in s:
            return "UNFURNISHED"
        
        # Fallback: if it matches one of the literals, return it, else UNKNOWN
        if s in ['FURNISHED', 'SEMI-FURNISHED', 'UNFURNISHED']:
            return s
            
        return "UNKNOWN"


class BatchItemResult(BaseModel):
    message_index: int = Field(..., description="The index ID of the message provided in the prompt.")
    listings: List[PropertyListing] = Field(default_factory=list)
    is_irrelevant: bool = Field(False)

# --- HELPER: PACK PROMPT ---
def construct_batch_prompt(messages: List[str]) -> str:
    """
    Creates a single prompt containing multiple numbered messages (0 to N).
    """
    prompt = "Extract listings from the following messages. Return a JSON object with a key 'results' which is a list of objects.\n\n"
    for i, msg in enumerate(messages):
        # We use a relative index (0, 1, 2...) in the prompt, mapped back later
        # Truncate extremely long messages to avoid token overflow
        safe_msg = msg[:3000] 
        prompt += f"--- Message {i} ---\n{safe_msg}\n\n"
    return prompt

# --- ASYNC WORKER: Process one "Packet" of 15 messages ---
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((json.JSONDecodeError, Exception))
)
async def _process_packet(messages: List[str], start_global_idx: int, semaphore: asyncio.Semaphore) -> List[BatchItemResult]:
    async with semaphore:
        prompt_user = construct_batch_prompt(messages)
        
        system_prompt = (
            "You are a high-performance real estate parser. "
            "Input: A list of numbered messages (Message 0, Message 1, etc.). "
            "Output: A JSON object `{\"results\": [...]}` where each item corresponds to a message.\n"
            "Rules:\n"
            "1. If a message has NO listing, set `is_irrelevant: true`.\n"
            "2. If a message has multiple listings, include them in the `listings` array.\n"
            "3. Normalize prices (1.5 Cr -> 15000000).\n"
            "3a. STRICTLY distinguish RENT  vs LEASE vs SALE/OWNERSHIP based on phrasing.\n"
            "4. `message_index` in output MUST match the 'Message X' number provided (0, 1, 2...).\n"
            "Schema: " + json.dumps(BatchItemResult.model_json_schema())
        )

        try:
            completion = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_user},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            
            # Parse Response
            content = completion.choices[0].message.content
            raw_json = json.loads(content)
            results_list = raw_json.get("results", [])
            
            # Map back to global indices
            final_results = []
            
            # Create a map for safety in case LLM skips an index or returns them out of order
            results_map = {r.get('message_index'): r for r in results_list if isinstance(r, dict)}
            
            # Iterate through the expected range to ensure every message gets a result object
            for local_i in range(len(messages)):
                global_i = start_global_idx + local_i
                
                if local_i in results_map:
                    # Valid LLM result found
                    item_data = results_map[local_i]
                    # Validate / Coerce into Pydantic model
                    # We override message_index to the global one to be safe
                    item_data['message_index'] = global_i 
                    try:
                        final_results.append(BatchItemResult(**item_data))
                    except Exception as val_err:
                        log.warning(f"Validation error for msg {global_i}: {val_err}")
                        # Fallback for validation error
                        final_results.append(BatchItemResult(message_index=global_i, listings=[], is_irrelevant=True))
                else:
                    # LLM missed this one, mark as irrelevant/empty fallback
                    final_results.append(BatchItemResult(message_index=global_i, listings=[], is_irrelevant=True))
            
            return final_results

        except Exception as e:
            log.error(f"Packet failed (Indices {start_global_idx}-{start_global_idx+len(messages)}): {e}")
            # Fallback: Return empty results for this packet so pipeline continues without crashing
            return [
                BatchItemResult(message_index=start_global_idx+i, listings=[], is_irrelevant=True) 
                for i in range(len(messages))
            ]

# --- MAIN ENTRY POINT ---
async def async_extract_listings_from_batch(messages: List[str]) -> List[BatchItemResult]:
    """
    Splits messages into packets, processes them in parallel.
    """
    if not messages:
        return []

    # 1. Chunk messages into packets
    packets = []
    for i in range(0, len(messages), MESSAGES_PER_PACKET):
        chunk = messages[i : i + MESSAGES_PER_PACKET]
        packets.append((chunk, i)) # (messages, start_global_index)

    # 2. Parallel Execution
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_PACKETS) 
    
    tasks = [
        _process_packet(pkt_msgs, start_idx, semaphore) 
        for pkt_msgs, start_idx in packets
    ]
    
    # Gather all list-of-lists
    nested_results = await asyncio.gather(*tasks)
    
    # Flatten results
    flat_results = []
    for sublist in nested_results:
        flat_results.extend(sublist)
        
    return flat_results

# --- Sync Wrapper ---
def extract_listings_from_batch(messages: List[str]) -> List[BatchItemResult]:
    if not messages:
        return []
    try:
        return asyncio.run(async_extract_listings_from_batch(messages))
    except Exception as e:
        log.error(f"Sync wrapper failed: {e}")
        # Final safety net
        return [BatchItemResult(message_index=i, listings=[], is_irrelevant=True) for i in range(len(messages))]