from celery import shared_task
from .pipeline import process_file_in_background
import logging

log = logging.getLogger(__name__)

@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def process_file_task(self, file_id):
    try:
        process_file_in_background(file_id)
    except Exception as e:
        log.exception(f"Celery task failed for file_id={file_id}: {e}")
        raise
