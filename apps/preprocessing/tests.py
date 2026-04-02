import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase

from apps.ingestion.models import RawFile, RawMessageChunk
from apps.ingestion.pipeline import process_single_llm_batch
from apps.preprocessing.extractor import BatchItemResult, PropertyListing, _process_packet
from apps.preprocessing.models import ListingChunk


class PropertyListingSchemaTests(TestCase):
    def test_listing_intent_flips_to_request_from_cleaned_text(self):
        listing = PropertyListing(
            cleaned_text="I am looking for a flat in Bandra",
            listing_intent="OFFER",
            transaction_type="SALE",
            property_type="RESIDENTIAL",
            location="Bandra",
        )
        self.assertEqual(listing.listing_intent, "REQUEST")

    def test_schema_normalizers_for_transaction_location_and_furnishing(self):
        listing = PropertyListing(
            cleaned_text="Need office space on lease",
            listing_intent="REQUEST",
            transaction_type="",
            property_type="COMMERCIAL",
            location=None,
            furnishing="fully furnished",
        )
        self.assertEqual(listing.transaction_type, "RENT")
        self.assertEqual(listing.location, "Unknown")
        self.assertEqual(listing.furnishing, "FURNISHED")


class ExtractorAndPipelineIntegrationTests(TestCase):
    def test_extractor_sequential_fallback(self):
        sequential_payload = [
            {
                "message_index": 0,
                "is_irrelevant": False,
                "listings": [
                    {
                        "cleaned_text": "2 bhk for rent in Bandra West",
                        "listing_intent": "OFFER",
                        "transaction_type": "RENT",
                        "property_type": "RESIDENTIAL",
                        "location": "Bandra West",
                    }
                ],
            }
        ]

        side_effects = [
            Exception("Mock 400 Bad Request"),
            sequential_payload,
            sequential_payload,
            sequential_payload,
        ]

        with patch("apps.preprocessing.extractor._call_llm_api", new=AsyncMock(side_effect=side_effects)) as mock_call:
            results = asyncio.run(
                _process_packet(
                    ["msg one", "msg two", "msg three"],
                    start_global_idx=0,
                    semaphore=asyncio.Semaphore(1),
                )
            )

        self.assertEqual(mock_call.call_count, 4)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(isinstance(item, BatchItemResult) for item in results))
        self.assertEqual([item.message_index for item in results], [0, 1, 2])

    def test_process_packet_uses_mocked_openai_and_validates_schema(self):
        fake_json = {
            "results": [
                {
                    "message_index": 0,
                    "is_irrelevant": False,
                    "listings": [
                        {
                            "cleaned_text": "I am looking for a flat in Bandra",
                            "listing_intent": "OFFER",
                            "transaction_type": "buy",
                            "property_type": "RESIDENTIAL",
                            "location": "Bandra West",
                        }
                    ],
                }
            ]
        }

        mock_completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=__import__("json").dumps(fake_json)))]
        )

        with patch(
            "apps.preprocessing.extractor.client.chat.completions.create",
            new=AsyncMock(return_value=mock_completion),
        ):
            results = asyncio.run(_process_packet(["mock message"], 0, asyncio.Semaphore(1)))

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], BatchItemResult)
        self.assertEqual(results[0].listings[0].listing_intent, "REQUEST")
        self.assertEqual(results[0].listings[0].transaction_type, "SALE")

    def test_mocked_extract_listings_creates_listingchunk_rows(self):
        raw_file = RawFile.objects.create(
            file=SimpleUploadedFile("chunk.txt", b"x", content_type="text/plain"),
            file_name="chunk.txt",
            content="x",
        )
        c1 = RawMessageChunk.objects.create(rawfile=raw_file, sender="Sender A", raw_text="Available 2 bhk for sale in Bandra West")
        c2 = RawMessageChunk.objects.create(rawfile=raw_file, sender="Sender B", raw_text="Need office on lease in Lower Parel")

        fake_results = [
            BatchItemResult(
                message_index=0,
                is_irrelevant=False,
                listings=[
                    PropertyListing(
                        cleaned_text="2 bhk for sale in bandra",
                        listing_intent="OFFER",
                        transaction_type="SALE",
                        property_type="RESIDENTIAL",
                        location="Bandra",
                    )
                ],
            ),
            BatchItemResult(message_index=1, is_irrelevant=True, listings=[]),
        ]

        with patch("apps.ingestion.pipeline.extract_listings_from_batch", return_value=fake_results), \
        patch("apps.ingestion.pipeline.get_batch_embeddings", None), \
        patch("apps.ingestion.pipeline.connection.close"):
            process_single_llm_batch((0, [c1.id, c2.id], raw_file.id))

        created = ListingChunk.objects.filter(raw_chunk=c1)
        self.assertEqual(created.count(), 1)
        self.assertEqual(created.first().transaction_type, "SALE")
        self.assertEqual(created.first().category, "RESIDENTIAL")
        self.assertFalse(ListingChunk.objects.filter(raw_chunk=c2).exists())
