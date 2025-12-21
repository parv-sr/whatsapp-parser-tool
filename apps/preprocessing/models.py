# apps/preprocessing/models.py
from django.db import models
from django.utils import timezone
from pgvector.django import VectorField

# We will reference RawMessageChunk from ingestion
# Use settings.AUTH_USER_MODEL if you need user FK later


class ListingChunk(models.Model):
    """
    Final preprocessed chunk (one listing or requirement) ready for extraction/embedding.
    """
    # Link to raw message chunk (optional - sometimes a listing may be created independently)
    raw_chunk = models.ForeignKey(
        "ingestion.RawMessageChunk",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="listing_chunks",
    )

    # Clean textual representation (what will be embedded)
    text = models.TextField()

    # A datetime when this listing was seen/posted (from message_start or parsed)
    date_seen = models.DateTimeField(null=True, blank=True)

    # Composite key (human readable) and hashed key for efficient lookup
    composite_key = models.CharField(max_length=512, db_index=True, blank=True)
    composite_hash = models.CharField(max_length=128, db_index=True, blank=True)

    # Extracted structured metadata (location, price, bhk, contacts, etc.)
    metadata = models.JSONField(default=dict, blank=True)

    # Intent / category / txn type
    INTENT_CHOICES = [("LISTING", "Listing"), ("REQUIREMENT", "Requirement"), ("UNKNOWN", "Unknown")]
    CATEGORY_CHOICES = [("RESIDENTIAL", "Residential"), ("COMMERCIAL", "Commercial"), ("LAND", "Land"), ("UNKNOWN", "Unknown")]
    TXN_CHOICES = [("RENT", "Rent"), ("SALE", "Sale"), ("LEASE", "Lease"), ("UNKNOWN", "Unknown")]

    intent = models.CharField(max_length=20, choices=INTENT_CHOICES, default="UNKNOWN")
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES, default="UNKNOWN")
    transaction_type = models.CharField(max_length=20, choices=TXN_CHOICES, default="UNKNOWN")

    confidence = models.FloatField(default=1.0)

    STATUS_CHOICES = [("ACTIVE", "Active"), ("STALE", "Stale"), ("DELETED", "Deleted")]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="ACTIVE")

    last_seen = models.DateTimeField(default=timezone.now)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["composite_hash"], name="preproc_comphash_idx"),
            models.Index(fields=["last_seen"], name="preproc_lastseen_idx"),
        ]

    def __str__(self):
        return f"{(self.text[:80] + '...') if self.text else 'ListingChunk'} (status={self.status})"


class EmbeddingRecord(models.Model):
    """
    Small record to keep track of embedding state and pointer to vector DB id.
    """
    listing_chunk = models.OneToOneField(ListingChunk, on_delete=models.CASCADE, related_name="embedding")
    embedding_vector = VectorField(dimensions=1536, null=True, blank=True)
    vector_db_id = models.CharField(max_length=255, null=True, blank=True)
    vector_index_name = models.CharField(max_length=128, null=True, blank=True)
    embedded_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"EmbeddingRecord({self.listing_chunk_id}, vector_id={self.vector_db_id})"


class PreprocessAudit(models.Model):
    """
    Audit log for actions taken by the preprocess pipeline (flagging stale, deletes, merges).
    """
    entity_type = models.CharField(max_length=50)  # e.g., LISTINGCHUNK, RAWFILE
    entity_id = models.BigIntegerField(null=True, blank=True)
    action = models.CharField(max_length=100)  # e.g., 'CREATED', 'MARKED_STALE', 'DELETED', 'MERGED'
    performed_by = models.CharField(max_length=100, default="system")  # or user id
    timestamp = models.DateTimeField(auto_now_add=True)
    reason = models.TextField(blank=True, null=True)
    details = models.JSONField(blank=True, null=True)

    def __str__(self):
        return f"{self.action} on {self.entity_type}:{self.entity_id} by {self.performed_by} at {self.timestamp}"
