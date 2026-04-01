import asyncio
import os
from unittest.mock import AsyncMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from django.test import TestCase

from apps.core import rag_graph


class RagGraphUnitTests(TestCase):
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

    def test_langgraph_pipeline_falls_back_when_no_relevant_contexts(self):
        with patch("apps.core.rag_graph._hybrid_retrieve_async", new=AsyncMock(return_value=[])):
            state = asyncio.run(rag_graph.run_rag("3bhk apartment in bkc", top_k=5))

        self.assertIn("couldn't find an exact match", state["answer"].lower())

    def test_format_output_no_context_fallback(self):
        state = asyncio.run(
            rag_graph._format_output_node(
                {
                    "query": "need something",
                    "contexts": [],
                    "graded_contexts": [],
                    "final": {"answer": "No exact results yet.", "sources": []},
                }
            )
        )

        self.assertIn("<p>", state["answer"])
        self.assertEqual(state["sources"], [])

    def test_low_score_fallback_sets_final(self):
        fallback_input = {
            "query": "2bhk in andheri",
            "is_real_estate_query": True,
            "graded_contexts": [
                {"id": 101, "relevance_score": 3, "metadata": {"location": "Andheri", "bhk": "2 BHK", "price": "2.2 Cr"}}
            ],
        }
        self.assertEqual(rag_graph._route_after_grading(fallback_input), "fallback")
        fallback_state = asyncio.run(rag_graph._fallback_node(fallback_input))
        self.assertIn("final", fallback_state)
        self.assertEqual(fallback_state["final"]["sources"], [101])

    def test_successful_generate_path(self):
        payload = {
            "query": "Show sale flats",
            "model": "gpt-4o-mini",
            "graded_contexts": [
                {"id": 11, "relevance_score": 9, "metadata": {"location": "BKC", "transaction_type": "SALE", "property_type": "RESIDENTIAL"}}
            ],
            "final": {"answer": "Found one listing.", "sources": [11], "model": "gpt-4o-mini", "confidence": 0.88},
        }
        self.assertEqual(rag_graph._route_after_grading(payload), "generate")
        state = asyncio.run(rag_graph._format_output_node(payload))
        self.assertEqual(state["sources"][0]["id"], 11)
        self.assertEqual(state["model"], "gpt-4o-mini")
        self.assertAlmostEqual(state["confidence"], 0.88)

    def test_format_output_filters_malformed_source_ids(self):
        payload = {
            "query": "Show listings",
            "contexts": [
                {"id": 7, "metadata": {}},
                {"id": 9, "metadata": {"location": "Bandra"}},
            ],
            "final": {"answer": "Potential matches.", "sources": ["7", "abc", {"id": "9"}, {"id": None}, 42.1]},
        }
        state = asyncio.run(rag_graph._format_output_node(payload))
        self.assertEqual([item["id"] for item in state["sources"]], [7, 9])
        self.assertIn("metadata", state["sources"][0])
