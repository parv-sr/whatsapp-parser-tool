import os
from celery import Celery
from celery.schedules import crontab
from django.conf import settings

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

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
