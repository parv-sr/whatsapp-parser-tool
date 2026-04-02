import io
import os
import zipfile
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase

from apps.ingestion.dedupe.pre_llm_dedupe import PreLLMDedupe
from apps.ingestion.models import RawFile, RawMessageChunk
from apps.ingestion.parsers.whatsapp_parser import (
    clean_block,
    parse_datetime,
    process_rawfile_from_uploaded_file,
)
from apps.ingestion.pipeline import _generate_dedupe_hash, process_single_llm_batch
from apps.preprocessing.extractor import BatchItemResult, PropertyListing
from apps.preprocessing.models import ListingChunk


class UploadNormalizationTests(TestCase):
    def test_txt_file_is_passthrough(self):
        txt = SimpleUploadedFile("chat.txt", b"hello world", content_type="text/plain")
        from apps.ingestion.views import _normalize_input_files

        normalized, errors = _normalize_input_files([txt])

        self.assertEqual(errors, [])
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0][0], "chat.txt")

    def test_zip_extracts_single_txt(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("inside/chat.txt", "hello from archive")

        archive = SimpleUploadedFile("bundle.zip", buf.getvalue(), content_type="application/zip")
        from apps.ingestion.views import _normalize_input_files

        normalized, errors = _normalize_input_files([archive])

        self.assertEqual(errors, [])
        self.assertEqual(len(normalized), 1)
        display_name, extracted_file = normalized[0]
        self.assertIn("bundle.zip -> chat.txt", display_name)
        self.assertEqual(extracted_file.read(), b"hello from archive")


class WhatsAppParserPipelineTests(TestCase):
    def _raw_file(self, content: str) -> RawFile:
        return RawFile.objects.create(
            file=SimpleUploadedFile("chat.txt", content.encode("utf-8"), content_type="text/plain"),
            file_name="chat.txt",
            content=content,
        )

    def test_clean_block_removes_header_and_system_and_keeps_message(self):
        block = "[29/12/25, 1:22:37 PM] ~ Nitin More:   *Available For outright sale*"
        cleaned = clean_block(block)
        self.assertEqual(cleaned, "*Available For outright sale*")

    def test_parse_datetime_parses_24h_and_12h(self):
        self.assertEqual(parse_datetime("29/12/25", "2:53 PM").year, 2025)
        self.assertEqual(parse_datetime("29/12/2025", "14:53").hour, 14)

    def test_process_rawfile_extracts_sender_datetime_ignores_system_and_dedupes(self):
        text = (
            "[29/12/25, 1:22:37 PM] ~ Nitin More: Available For outright sale in BKC\n"
            "[29/12/25, 1:22:39 PM] System: Messages and calls are end-to-end encrypted.\n"
            "[29/12/25, 1:24:00 PM] ~ Advait Makhija: Available For outright sale in BKC\n"
        )
        raw_file = self._raw_file(text)

        result = process_rawfile_from_uploaded_file(raw_file)
        chunks = list(RawMessageChunk.objects.filter(rawfile=raw_file).order_by("id"))

        self.assertEqual(result, {"created": 1, "ignored": 2})
        self.assertEqual(chunks[0].sender, "~ Nitin More")
        self.assertEqual(chunks[0].status, "NEW")
        self.assertIn("Available For outright sale", chunks[0].cleaned_text)
        self.assertEqual(chunks[1].status, "IGNORED")
        self.assertEqual(chunks[2].status, "DUPLICATE_LOCAL")


class DeduplicationTests(TestCase):
    def test_pre_llm_dedupe_flags_identical_messages_from_different_senders(self):
        deduper = PreLLMDedupe()
        message = "Available 2 bhk in Bandra West carpet 850 sqft price 3.2 cr"

        self.assertTrue(deduper.should_keep(message))
        self.assertFalse(deduper.should_keep(message))

    def test_dupe_tracker_threefold_in_chat_in_batch_in_db(self):
        raw_file = RawFile.objects.create(
            file=SimpleUploadedFile("pipeline.txt", b"x", content_type="text/plain"),
            file_name="pipeline.txt",
            content="x",
        )
        # First two chunks are exact same raw text => in-chat duplicate on second one.
        c1 = RawMessageChunk.objects.create(rawfile=raw_file, sender="Alice", raw_text="sale 2 bhk bandra")
        c2 = RawMessageChunk.objects.create(rawfile=raw_file, sender="Bob", raw_text="sale 2 bhk bandra")
        # Third chunk differs enough to pass pre-LLM dedupe but returns same listing => in-batch duplicate.
        c3 = RawMessageChunk.objects.create(rawfile=raw_file, sender="Alice", raw_text="sale 2 bhk bandra west with balcony")

        existing_hash = _generate_dedupe_hash("db dupe text", "Alice")
        ListingChunk.objects.create(
            raw_chunk=c1,
            text="existing",
            composite_key="k",
            composite_hash=existing_hash,
            metadata={},
            intent="LISTING",
            category="RESIDENTIAL",
            transaction_type="SALE",
        )

        mocked_results = [
            BatchItemResult(
                message_index=0,
                is_irrelevant=False,
                listings=[
                    PropertyListing(
                        cleaned_text="batch dupe text",
                        listing_intent="OFFER",
                        transaction_type="SALE",
                        property_type="RESIDENTIAL",
                        location="Bandra",
                    )
                ],
            ),
            BatchItemResult(
                message_index=1,
                is_irrelevant=False,
                listings=[
                    PropertyListing(
                        cleaned_text="batch dupe text",
                        listing_intent="OFFER",
                        transaction_type="SALE",
                        property_type="RESIDENTIAL",
                        location="Bandra",
                    ),
                    PropertyListing(
                        cleaned_text="db dupe text",
                        listing_intent="OFFER",
                        transaction_type="SALE",
                        property_type="RESIDENTIAL",
                        location="Bandra",
                    ),
                ],
            ),
        ]

        with patch("apps.ingestion.pipeline.extract_listings_from_batch", return_value=mocked_results), \
            patch("apps.ingestion.pipeline.get_batch_embeddings", None):
            response = process_single_llm_batch((0, [c1.id, c2.id, c3.id], raw_file.id))

        dupe = response["dupe"]
        self.assertEqual(dupe["in_chat"], 1)        
        self.assertEqual(dupe["in_batch"], 1)
        self.assertEqual(dupe["in_db"], 1)
        self.assertEqual(dupe["inserted"], 1)

    def test_dedupe_hash_defaults_remain_backward_compatible(self):
        base = _generate_dedupe_hash("db dupe text", "Alice")
        explicit_empty = _generate_dedupe_hash("db dupe text", "Alice", "", "", "")
        enriched = _generate_dedupe_hash("db dupe text", "Alice", "Bandra", "SALE", "OFFER")

        self.assertEqual(base, explicit_empty)
        self.assertNotEqual(base, enriched)
