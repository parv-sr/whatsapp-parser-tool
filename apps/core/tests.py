import asyncio
import os
from unittest.mock import AsyncMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from django.test import TestCase
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel
from apps.core import rag_graph


class RagGraphUnitTests(TestCase):
    def _has_pydantic_instance(self, value):
        if isinstance(value, BaseModel):
            return True
        if isinstance(value, dict):
            return any(self._has_pydantic_instance(v) for v in value.values())
        if isinstance(value, list):
            return any(self._has_pydantic_instance(v) for v in value)
        return False

    def test_extract_filters_for_buy_query(self):
        filters = rag_graph._extract_filters("Buy a 2bhk flat in Bandra")
        self.assertEqual(
            filters,
            {
                "transaction_type": "SALE",
                "property_type": "RESIDENTIAL",
            },
        )

    def test_non_domain_query_rejected(self):
        self.assertFalse(rag_graph._is_domain_query("What is the capital of France?"))

    def test_build_messages_wraps_snippets_as_json_blocks(self):
        contexts = [
            {
                "id": 10,
                "metadata": {"location": "Bandra", "transaction_type": "SALE", "property_type": "RESIDENTIAL"},
                "hybrid_score": 0.91,
            }
        ]
        messages = rag_graph._build_messages("Buy in Bandra", contexts)
        content = messages[-1]["content"]
        self.assertIn("[SNIPPET 1]", content)
        self.assertIn('"_id": 10', content)
        self.assertIn("[/SNIPPET 1]", content)

    def test_agent_routes_to_tool(self):
        ai_with_tool_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "search_property_listings",
                    "args": {"query": "I want a 2BHK in Bandra"},
                    "type": "tool_call",
                }
            ],
        )

        fake_llm = AsyncMock()
        fake_llm.bind_tools.return_value = fake_llm
        fake_llm.ainvoke = AsyncMock(return_value=ai_with_tool_call)

        with patch("apps.core.rag_graph._chat_llm", return_value=fake_llm):
            state = asyncio.run(rag_graph._agent_node({"messages": [HumanMessage(content="I want a 2BHK in Bandra")]}))

        messages = state["messages"]
        self.assertIsInstance(messages[-1], AIMessage)
        self.assertTrue(messages[-1].tool_calls)
        self.assertEqual(messages[-1].tool_calls[0]["name"], "search_property_listings")

    def test_agent_memory_followup(self):
        async def _fake_ainvoke(messages):
            joined = " ".join(getattr(msg, "content", "") for msg in messages if hasattr(msg, "content")).lower()
            args = {"query": "pet friendly options in Andheri West"} if "pets" in joined and "andheri west" in joined else {"query": "andheri west"}
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-2",
                        "name": "search_property_listings",
                        "args": args,
                        "type": "tool_call",
                    }
                ],
            )

        fake_llm = AsyncMock()
        fake_llm.bind_tools.return_value = fake_llm
        fake_llm.ainvoke = AsyncMock(side_effect=_fake_ainvoke)
        seeded_messages = [
            HumanMessage(content="I need a place that allows pets"),
            AIMessage(content="Got it, pet-friendly properties."),
            HumanMessage(content="What about in Andheri West?"),
        ]

        with patch("apps.core.rag_graph._chat_llm", return_value=fake_llm):
            state = asyncio.run(rag_graph._agent_node({"messages": seeded_messages}))

        tool_args = state["messages"][-1].tool_calls[0]["args"]
        serialized = str(tool_args).lower()
        self.assertIn("pet", serialized)
        self.assertIn("andheri west", serialized)

    def test_deterministic_rerank_top_k_stability_fixture(self):
        class FakeListing:
            def __init__(self, id, transaction_type, category, metadata):
                self.id = id
                self.transaction_type = transaction_type
                self.category = category
                self.metadata = metadata

        listing_map = {
            1: FakeListing(1, "RENT", "RESIDENTIAL", {"location": "Bandra West", "features": ["parking", "lift"]}),
            2: FakeListing(2, "RENT", "RESIDENTIAL", {"location": "Andheri East", "features": ["lift"]}),
            3: FakeListing(3, "SALE", "RESIDENTIAL", {"location": "Bandra", "features": ["parking"]}),
            4: FakeListing(4, "RENT", "COMMERCIAL", {"location": "Bandra Kurla Complex", "features": ["parking"]}),
        }
        fused_scores = {1: 0.22, 2: 0.35, 3: 0.55, 4: 0.30}
        rank_data = {lid: {"distance": 0.1 * lid} for lid in fused_scores}
        query = "rent residential bandra with parking"

        with patch("apps.core.rag_graph._load_listing_briefs", return_value=listing_map):
            top_a = rag_graph._deterministic_rerank(query=query, fused_scores=fused_scores, rank_data=rank_data, top_k=3)
            top_b = rag_graph._deterministic_rerank(
                query=query,
                fused_scores=dict(reversed(list(fused_scores.items()))),
                rank_data=rank_data,
                top_k=3,
            )

        ids_a = [row["listing_chunk_id"] for row in top_a]
        ids_b = [row["listing_chunk_id"] for row in top_b]
        self.assertEqual(ids_a, ids_b)
        self.assertEqual(len(ids_a), 3)

    def test_grade_documents_can_run_without_llm(self):
        state = {
            "query": "rent in bandra",
            "use_llm_grading": False,
            "contexts": [
                {
                    "id": 1,
                    "metadata": {"location": "Bandra", "transaction_type": "RENT"},
                    "hybrid_score": 0.2,
                    "deterministic_score": 0.81,
                }
            ],
        }
        result = asyncio.run(rag_graph._grade_documents_node(state))
        self.assertEqual(result["graded_contexts"][0]["relevance_reason"], "deterministic_only")
        self.assertGreaterEqual(result["graded_contexts"][0]["relevance_score"], 8)
