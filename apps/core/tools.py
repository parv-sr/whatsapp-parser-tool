import json

from langchain_core.tools import tool



@tool
async def search_property_listings(
    location: str = None,
    transaction_type: str = None,
    property_type: str = None,
    bhk: float = None,
    must_have_features: list[str] = None,
) -> str:
    """Search the real-estate listings database and return grounded candidates for property queries.

    Call this tool whenever the user asks for properties (including follow-ups), so the answer is based on
    retrieved listing evidence instead of memory-only generation. Pass all known constraints from the current
    turn and conversation history (for example location, transaction type, property type, bhk, and required
    features). The tool applies structured filters, retrieves matching listings, and returns JSON text that
    the model can read, compare, and rank before responding.
    """
    filters: dict[str, object] = {}
    if location:
        filters["location"] = location
    if transaction_type:
        filters["transaction_type"] = transaction_type
    if property_type:
        filters["property_type"] = property_type
    if bhk is not None:
        filters["bhk"] = bhk
    if must_have_features:
        filters["must_have_features"] = must_have_features

    from apps.core.rag_graph import _hybrid_retrieve_async, _load_contexts_async

    nearest = await _hybrid_retrieve_async(query="", top_k=15, filters=filters)
    contexts = await _load_contexts_async(nearest)
    return json.dumps(contexts, ensure_ascii=False)
