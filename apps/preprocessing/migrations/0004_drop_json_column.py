# 3_drop_json_column.py
from django.db import migrations

class Migration(migrations.Migration):

    dependencies = [
        ("preprocessing", "0003_backfill_vectors"),
    ]

    operations = [
        migrations.RunSQL(
            """
            ALTER TABLE preprocessing_embeddingrecord
            DROP COLUMN IF EXISTS embedding_vector;
            """,
            reverse_sql="""
            ALTER TABLE preprocessing_embeddingrecord
            ADD COLUMN embedding_vector jsonb;
            """
        )
    ]
