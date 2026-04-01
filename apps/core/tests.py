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

    def test_normalize_shorthand_expands_locations_bhk_and_intent(self):
        normalized = rag_graph._normalize_real_estate_shorthand("Need 2BHK in BKC, Andheri W or Lokhandwala on lease")

        self.assertIn("2 bhk apartment", normalized.lower())
        self.assertIn("bandra kurla complex mumbai", normalized.lower())
        self.assertIn("andheri west mumbai", normalized.lower())
        self.assertIn("lokhandwala andheri west mumbai", normalized.lower())
        self.assertIn("for lease", normalized.lower())

    def test_ensure_constraints_in_rewrite_preserves_hard_constraints(self):
        rewritten = rag_graph._ensure_constraints_in_rewrite(
            "2 bhk in andheri w must-have balcony attached bath road view",
            "2 bhk apartment in andheri west mumbai for rent",
        )

        lowered = rewritten.lower()
        self.assertIn("must have balcony", lowered)
        self.assertIn("attached bath", lowered)
        self.assertIn("road view", lowered)
