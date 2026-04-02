from langchain_core.tools import tool
import json

@tool
async def search_property_listings(search_query: str) -> str:
    """
    Searches the database for real estate listings.
    For follow-up queries, COMBINE the user's previous constraints (e.g., 'pet friendly') 
    with their new constraints (e.g., 'in Bandra') into a single rich search phrase.
    Example: "2 BHK apartment in Bandra West, pet friendly, furnished"
    """
    # Inline import prevents circular dependency crashes on boot
    from apps.core.rag_graph import _hybrid_retrieve_async, _load_contexts_async
    
    # We pass the rich query straight to your brilliant hybrid retriever!
    nearest = await _hybrid_retrieve_async(query=search_query, top_k=15)
    contexts = await _load_contexts_async(nearest)
    return json.dumps(contexts)