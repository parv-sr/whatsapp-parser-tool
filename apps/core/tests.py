import asyncio
import os
from unittest.mock import AsyncMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from django.test import TestCase
from django.db import connections

from apps.core import rag_graph




class ConnectionCleanupTestCase(TestCase):
    def tearDown(self):
        connections.close_all()
        super().tearDown()

class RagGraphUnitTests(ConnectionCleanupTestCase):
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
        content = messages[-1].content
        self.assertIn("[SNIPPET 1]", content)
        self.assertIn('"_id": 10', content)
        self.assertIn("[/SNIPPET 1]", content)

    def test_langgraph_pipeline_routes_to_reject_when_gibberish(self):
        with patch("apps.core.rag_graph._hybrid_retrieve_async", new=AsyncMock(return_value=[])), patch(
            "apps.core.rag_graph._load_contexts_async", new=AsyncMock(return_value=[])
        ):
            state = asyncio.run(rag_graph.run_rag("asdfghjkl", top_k=5))

        self.assertEqual(state["answer"], rag_graph._reject_text())
