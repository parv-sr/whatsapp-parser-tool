from celery import shared_task
from django.utils import timezone
from datetime import timedelta
from django.db import transaction

from apps.preprocessing.models import ListingChunk, PreprocessAudit


@shared_task
def mark_and_purge_old_listings():
    """
    Daily housekeeping:
    - Mark listings older than 30 days as STALE.
    - Delete listings older than 60 days.
    """

    now = timezone.now()
    stale_cutoff = now - timedelta(days=30)
    delete_cutoff = now - timedelta(days=60)

    # -------------------------------
    # 1) Mark stale
    # -------------------------------
    stale_qs = ListingChunk.objects.filter(
        status="ACTIVE",
        last_seen__lt=stale_cutoff
    ).only("id")  # minimal load

    stale_ids = list(stale_qs.values_list("id", flat=True))
    if stale_ids:
        ListingChunk.objects.filter(id__in=stale_ids).update(status="STALE")

        PreprocessAudit.objects.bulk_create([
            PreprocessAudit(
                entity_type="LISTINGCHUNK",
                entity_id=i,
                action="MARKED_STALE",
                performed_by="system",
                reason="last_seen > 30 days",
                details={}
            ) for i in stale_ids
        ], batch_size=200)

    # -------------------------------
    # 2) Delete entries older than 60 days
    # -------------------------------
    old_qs = ListingChunk.objects.filter(
        last_seen__lt=delete_cutoff
    ).only("id")

    old_ids = list(old_qs.values_list("id", flat=True))

    # Perform deletion in batches to avoid table locks
    BATCH_SIZE = 2000

    for i in range(0, len(old_ids), BATCH_SIZE):
        chunk = old_ids[i:i+BATCH_SIZE]

        PreprocessAudit.objects.bulk_create([
            PreprocessAudit(
                entity_type="LISTINGCHUNK",
                entity_id=j,
                action="DELETED",
                performed_by="system",
                reason="last_seen > 60 days",
                details={}
            ) for j in chunk
        ], batch_size=200)

        with transaction.atomic():
            ListingChunk.objects.filter(id__in=chunk).delete()
