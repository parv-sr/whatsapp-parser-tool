# 1_add_pgvector_column.py
from django.db import migrations
from django.conf import settings

class Migration(migrations.Migration):

    dependencies = [
        ("preprocessing", "0001_initial"), 
    ]

    operations = [
        migrations.RunSQL(
            """
            CREATE EXTENSION IF NOT EXISTS vector;

            ALTER TABLE preprocessing_embeddingrecord
            ADD COLUMN IF NOT EXISTS embedding_vector_v vector(1536);
            """,
            reverse_sql="""
            ALTER TABLE preprocessing_embeddingrecord
            DROP COLUMN IF EXISTS embedding_vector_v;
            """
        )
    ]
