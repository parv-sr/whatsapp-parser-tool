import logging
import os
from typing import List, Optional, Literal
from django.conf import settings
from openai import OpenAI
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# Initialize OpenAI Client
# Make sure OPENAI_API_KEY is set in your settings.py or .env
client = OpenAI(api_key=getattr(settings, "OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY")))

# --- Pydantic Schemas (The Contract) ---

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

# --- Main Extraction Function ---

def extract_listings_from_text(message_text: str) -> List[PropertyListing]:
    """
    Uses GPT-4o-mini to parse a raw WhatsApp message into structured data.
    Handles multi-listing messages automatically.
    """
    # Safety check for empty/short text
    if not message_text or len(message_text) < 10:
        return []

    try:
        completion = client.beta.chat.completions.parse(
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
                {"role": "user", "content": message_text},
            ],
            response_format=ExtractionResult,
            temperature=0.0, # Deterministic output
        )

        result = completion.choices[0].message.parsed

        if result.is_irrelevant:
            return []
        
        return result.listings

    except Exception as e:
        log.error(f"LLM Extraction failed: {e}")
        return []