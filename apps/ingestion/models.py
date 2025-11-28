from django.db import models
from django.contrib.postgres.fields import JSONField
from django.conf import settings

class RawFile(models.Model):
    """
    Represents an uploaded WhatsApp export .txt file.
    content stores full text (we also store the uploaded file path for reference).
    """
    file = models.FileField(upload_to="raw_chats/")  # keeps original .txt in media/raw_chats/
    file_name = models.CharField(max_length=255)
    content = models.TextField(blank=True)  # we will populate this after read
    source = models.CharField(max_length=255, blank=True, null=True)  # optional chat title
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    process_started_at = models.DateTimeField(null=True, blank=True)
    process_finished_at = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(blank=True, null=True)

    status = models.CharField(
        max_length=20,
        default="PENDING",
        choices=[
            ("PENDING", "Pending"),
            ("PROCESSING", "Processing"),
            ("COMPLETED", "Completed"),
            ("FAILED", "Failed"),
        ]
    )

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="uploaded_files"
    )

    def __str__(self):
        return f"{self.file_name} — uploaded {self.uploaded_at:%Y-%m-%d %H:%M}"


class RawMessageChunk(models.Model):
    """
    Each row represents a single WhatsApp message boundary (one message).
    We'll create these by splitting RawFile.content on the message header markers.
    """
    rawfile = models.ForeignKey(RawFile, on_delete=models.CASCADE, related_name="raw_chunks")
    message_start = models.DateTimeField(null=True, blank=True)
    sender = models.CharField(max_length=255, null=True, blank=True)
    raw_text = models.TextField()       # original message block including header if desired
    cleaned_text = models.TextField(blank=True, null=True)  # cleaned version
    split_into = models.IntegerField(default=0)  # number of subchunks (listings) created from this message
    status = models.CharField(max_length=20, default="NEW")  # NEW | PROCESSED | IGNORED | ERROR
    created_at = models.DateTimeField(auto_now_add=True)

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="raw_message_chunks",
    )

    def __str__(self):
        preview = (self.cleaned_text or self.raw_text)[:80].replace("\n", " ")
        return f"{self.sender or 'Unknown'} @ {self.message_start} — {preview}..."

class FileProcess(models.Model):
    file = models.FileField(upload_to="uploads/")
    status = models.CharField(max_length=255, default="Queued")
    progress = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Processing {self.file.name} ({self.progress}%)"