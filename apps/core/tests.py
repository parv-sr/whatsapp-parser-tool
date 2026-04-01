import asyncio
import os
from unittest.mock import AsyncMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.core.models import ChatMessage
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


class ChatStreamTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username="stream-user", password="testpass123")
        self.client.force_login(self.user)

    def test_chat_stream_works_in_sync_client_and_keeps_ndjson_framing(self):
        async def fake_stream_rag_events(**kwargs):
            yield {"type": "token", "delta": "Hello "}
            yield {"type": "token", "delta": "there"}
            yield {"type": "final", "sources": [{"id": 1}], "model": "gpt-4o-mini"}

        with (
            patch("apps.core.views.get_recent_messages_text", new=AsyncMock(return_value="")),
            patch("apps.core.views.stream_rag_events", side_effect=fake_stream_rag_events),
        ):
            response = self.client.post(
                reverse("core:chat_stream"),
                data='{"query":"Hi"}',
                content_type="application/json",
            )
            chunks = [c.decode("utf-8") for c in response.streaming_content]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/x-ndjson")
        self.assertEqual(chunks[0], '{"type": "token", "delta": "Hello "}\n')
        self.assertEqual(chunks[1], '{"type": "token", "delta": "there"}\n')
        self.assertEqual(chunks[2], '{"type": "done", "model": "gpt-4o-mini", "sources": [{"id": 1}]}\n')

        messages = list(ChatMessage.objects.filter(user=self.user).values_list("role", "content"))
        self.assertEqual(messages, [("user", "Hi"), ("assistant", "Hello there")])
