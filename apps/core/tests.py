import asyncio
import os
from unittest.mock import AsyncMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from django.test import TestCase
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

    def test_langgraph_pipeline_falls_back_when_no_relevant_contexts(self):
        with patch("apps.core.rag_graph._hybrid_retrieve_async", new=AsyncMock(return_value=[])):
            state = asyncio.run(rag_graph.run_rag("3bhk apartment in bkc", top_k=5))

        self.assertIn("couldn't find an exact match", state["answer"].lower())

    def test_to_plain_data_converts_nested_pydantic_values(self):
        payload = {
            "final": rag_graph.FinalAnswer(answer="ok", has_relevant_data=True, sources=[11]),
            "grade": rag_graph.DocumentGrade(score=8, reason="good"),
        }
        normalized = rag_graph._to_plain_data(payload)
        self.assertEqual(normalized["final"]["answer"], "ok")
        self.assertEqual(normalized["grade"]["score"], 8)
        self.assertFalse(self._has_pydantic_instance(normalized))

    def test_run_rag_removes_pydantic_instances_from_graph_state(self):
        class _FakeWorkflow:
            async def ainvoke(self, *_args, **_kwargs):
                return {
                    "final": rag_graph.FinalAnswer(answer="done", has_relevant_data=True, sources=[22]),
                    "graded_contexts": [{"grade": rag_graph.DocumentGrade(score=9, reason="fit")}],
                }

        with patch("apps.core.rag_graph.RAG_WORKFLOW", new=_FakeWorkflow()):
            state = asyncio.run(rag_graph.run_rag("sample query"))

        self.assertEqual(state["final"]["answer"], "done")
        self.assertEqual(state["graded_contexts"][0]["grade"]["score"], 9)
        self.assertFalse(self._has_pydantic_instance(state))
