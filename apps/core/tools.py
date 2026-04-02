import json
from typing import Any, Dict, Optional

from langchain_core.tools import tool


@tool
async def search_property_listings(
    query: str,
    location: Optional[str] = None,
    transaction_type: Optional[str] = None,
    property_type: Optional[str] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    bhk: Optional[float] = None,
    top_k: int = 12,
) -> str:
    """Search WhatsApp-derived real-estate listings with structured filters and return matching contexts.

    Use this tool whenever the user asks for property recommendations, follow-up refinements,
    comparisons, or shortlist generation (for example: "same but in Bandra", "under 2 cr",
    "2 bhk for rent", "show best options").

    Inputs:
    - query: natural-language search intent. Include conversational context in this value.
    - location, transaction_type, property_type: optional hard filters.
    - min_price, max_price, bhk: optional numeric constraints.
    - top_k: number of retrieved candidates before final reasoning.

    Output:
    - JSON string of retrieved listing contexts. Each context includes listing id, normalized metadata,
      and retrieval scores. The caller should reason over these contexts and select the best 3-10 matches.
    """
    from apps.core.rag_graph import _extract_filters, _hybrid_retrieve_async, _load_contexts_async

    filters: Dict[str, Any] = _extract_filters(query)

    if location:
        filters["location"] = location
    if transaction_type:
        filters["transaction_type"] = transaction_type.upper()
    if property_type:
        filters["property_type"] = property_type.upper()
    if min_price is not None:
        filters["min_price"] = int(min_price)
    if max_price is not None:
        filters["max_price"] = int(max_price)
    if bhk is not None:
        filters["bhk"] = float(bhk)

    merged_query = query
    if location and location.lower() not in query.lower():
        merged_query = f"{query} in {location}"

    nearest = await _hybrid_retrieve_async(merged_query, max(3, min(30, int(top_k))), filters=filters)
    contexts = await _load_contexts_async(nearest)
    return json.dumps(contexts, ensure_ascii=False)
