from django.core.management.base import BaseCommand
from apps.preprocessing.tasks import mark_and_purge_old_listings

class Command(BaseCommand):
    help = "Marks stale listings and purges old ones."

    def handle(self, *args, **kwargs):
        mark_and_purge_old_listings()
        self.stdout.write(self.style.SUCCESS("Housekeeping completed."))
