import os
import logging
from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_ready, worker_shutting_down, worker_shutdown
from django.conf import settings

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
log = logging.getLogger(__name__)

app = Celery("config")
app.config_from_object("django.conf:settings", namespace="CELERY")

# Force broker/backend from Django settings so stray env vars (e.g. BROKER_URL/CELERY_BROKER_URL)
# cannot accidentally switch transport at runtime.
app.conf.broker_url = settings.CELERY_BROKER_URL
app.conf.result_backend = settings.CELERY_RESULT_BACKEND

app.autodiscover_tasks()


app.conf.beat_schedule = {
    "daily-purge-and-stale-check": {
        "task": "apps.preprocessing.tasks.mark_and_purge_old_listings",
        "schedule": crontab(hour=2, minute=0), 
    }
}


@worker_ready.connect
def _on_worker_ready(sender=None, **kwargs):
    log.info("Celery worker ready hostname=%s pid=%s broker=%s", sender, os.getpid(), app.conf.broker_url)


@worker_shutting_down.connect
def _on_worker_shutting_down(sender=None, sig=None, how=None, exitcode=None, **kwargs):
    log.warning(
        "Celery worker shutting down hostname=%s pid=%s signal=%s how=%s exitcode=%s",
        sender,
        os.getpid(),
        sig,
        how,
        exitcode,
    )


@worker_shutdown.connect
def _on_worker_shutdown(sender=None, **kwargs):
    log.warning("Celery worker shutdown complete hostname=%s pid=%s", sender, os.getpid())
