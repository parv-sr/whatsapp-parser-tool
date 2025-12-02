from celery import shared_task
from .pipeline import process_file_in_background
import logging
from .models import RawFile

log = logging.getLogger(__name__)

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def process_file_task(self, file_id):
    try:
        # Optional: Quick check to ensure DB is ready
        if not RawFile.objects.filter(pk=file_id).exists():
            log.warning(f"File {file_id} not found yet. Retrying...")
            # This forces a retry, waiting for the DB to catch up if there's lag
            raise Exception(f"RawFile {file_id} not found")

        process_file_in_background(file_id)
        
    except Exception as e:
        log.exception(f"Celery task failed for file_id={file_id}: {e}")
        raise