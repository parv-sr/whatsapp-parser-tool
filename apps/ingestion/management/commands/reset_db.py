from django.core.management.base import BaseCommand
from django.db import connection

class Command(BaseCommand):
    help = (
        "Truncates all project-specific tables. This is IRREVERSIBLE."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--no-input',
            '--yes',
            action='store_true',
            help='Skips the confirmation prompt.',
        )

    def handle(self, *args, **options):
        
        tables_to_truncate = [
            "ingestion_rawfile",
            "ingestion_rawmessagechunk",
            "preprocessing_embeddingrecord",
            "preprocessing_listingchunk",
            "preprocessing_preprocessaudit",
            "langchain_pg_embedding"
        ]
        # ----------------------------------------------------

        self.stdout.write(
            self.style.WARNING(
                "\n"
                "THIS WILL DELETE ALL DATA FROM THE FOLLOWING TABLES IRREVERSIBLY:\n"
            )
        )
        
        for table in tables_to_truncate:
            self.stdout.write(f"- {table}")

        if not options['no_input']:
            confirmation = input("\nAre you sure you want to proceed? [yes/no]: ")
            if confirmation.lower() != 'yes':
                self.stdout.write(self.style.ERROR("Operation cancelled."))
                return

        # Build the TRUNCATE command
        # We add "RESTART IDENTITY" to reset all primary key sequences
        table_list = ", ".join(f'"{t}"' for t in tables_to_truncate)
        truncate_command = f"TRUNCATE TABLE {table_list} RESTART IDENTITY CASCADE;"

        self.stdout.write(self.style.NOTICE("\nExecuting TRUNCATE..."))
        
        try:
            with connection.cursor() as cursor:
                cursor.execute(truncate_command)
            
            self.stdout.write(
                self.style.SUCCESS("Successfully truncated all specified tables.")
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"An error occurred: {e}")
            )