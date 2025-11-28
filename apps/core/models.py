from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.conf import settings


class ListingChunk(models.Model):
    """
    A single WhatsApp listing/message after preprocessing.
    Stores metadata + embedding + cleaned text.
    """
    # core text
    cleaned_text = models.TextField()

    # metadata fields
    bhk = models.IntegerField(null=True, blank=True)
    sqft = models.IntegerField(null=True, blank=True)
    price = models.FloatField(null=True, blank=True)
    parking = models.BooleanField(null=True, blank=True)

    features = ArrayField(
        base_field=models.CharField(max_length=255),
        null=True,
        blank=True
    )

    location = models.CharField(max_length=255, null=True, blank=True)
    furnishing = models.CharField(max_length=255, null=True, blank=True)

    listing_type = models.CharField(max_length=50, null=True, blank=True)
    building_name = models.CharField(max_length=255, null=True, blank=True)
    property_type = models.CharField(max_length=50, null=True, blank=True)

    contact_numbers = ArrayField(
        base_field=models.CharField(max_length=20),
        null=True,
        blank=True
    )

    # embedding (1536 dims)
    embedding = ArrayField(
        base_field=models.FloatField(),
        size=1536,
        null=True,
        blank=True
    )

    # For FAISS ID mapping
    listing_id = models.IntegerField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.location} — {self.bhk} BHK ({self.listing_id})"


class ChatMessage(models.Model):
    ROLE_CHOICES = (
        ("user", "USER"), 
        ("assistant", "ASSISTANT")
    )

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="chat_messages"
    )

    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["timestamp"]

    def __str__(self):
        return f"{self.user.username} - {self.role}: {self.content[:30]}"
    
    