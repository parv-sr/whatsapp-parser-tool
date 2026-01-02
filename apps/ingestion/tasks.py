from celery import shared_task
from django.db import close_old_connections
import logging
from .models import RawFile
from .pipeline import process_file_in_background

log = logging.getLogger(__name__)

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def process_file_task(self, file_id):
    """
    Orchestrator Task:
    In the new Threaded Architecture, this single task handles the entire lifecycle:
    Streaming -> Batching -> Threaded Extraction -> Saving -> Completion.
    """
    close_old_connections()
    try:
        # Quick check
        if not RawFile.objects.filter(pk=file_id).exists():
            log.warning(f"File {file_id} not found yet. Retrying...")
            raise Exception(f"RawFile {file_id} not found")

        # Calls the threaded orchestrator
        process_file_in_background(file_id)
        
    except Exception as e:
        log.exception(f"Celery task failed for file_id={file_id}: {e}")
        # We re-raise to ensure Celery records the failure or retries
        raise
    finally:
        close_old_connections()