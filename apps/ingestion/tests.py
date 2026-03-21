import io
import zipfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase

from apps.ingestion.views import _normalize_input_files

class UploadNormalizationTests(TestCase):
    def test_txt_file_is_passthrough(self):
        txt = SimpleUploadedFile("chat.txt", b"hello world", content_type="text/plain")
        normalized, errors = _normalize_input_files([txt])

        self.assertEqual(errors, [])
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0][0], "chat.txt")

    def test_zip_extracts_single_txt(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("inside/chat.txt", "hello from archive")

        archive = SimpleUploadedFile("bundle.zip", buf.getvalue(), content_type="application/zip")
        normalized, errors = _normalize_input_files([archive])

        self.assertEqual(errors, [])
        self.assertEqual(len(normalized), 1)
        display_name, extracted_file = normalized[0]
        self.assertIn("bundle.zip -> chat.txt", display_name)
        self.assertEqual(extracted_file.read(), b"hello from archive")
