from celery import shared_task
from django.db import close_old_connections
from .pipeline import process_file_in_background, process_raw_chunk_batch
import logging
from .models import RawFile

log = logging.getLogger(__name__)

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def process_file_task(self, file_id):
    """
    Orchestrator Task:
    Streams the file, breaks it into chunks, saves to DB, and dispatches worker tasks.
    """
    close_old_connections()
    try:
        # Quick check
        if not RawFile.objects.filter(pk=file_id).exists():
            log.warning(f"File {file_id} not found yet. Retrying...")
            raise Exception(f"RawFile {file_id} not found")

        # Call the streaming orchestrator in pipeline.py
        process_file_in_background(file_id)
        
    except Exception as e:
        log.exception(f"Celery orchestrator failed for file_id={file_id}: {e}")
        raise
    finally:
        close_old_connections()


@shared_task(bind=True, ignore_result=True)
def process_batch_task(self, chunk_ids):
    """
    Worker Task:
    Processes a specific list of RawMessageChunk IDs.
    - ignore_result=True saves Redis OPS (Critical for free tier)
    """
    close_old_connections()
    try:
        process_raw_chunk_batch(chunk_ids)
    except Exception as e:
        # Log but don't retry indefinitely to avoid queue clogging on free tier
        log.error(f"Batch processing failed for {len(chunk_ids)} chunks: {e}")
    finally:
        close_old_connections()