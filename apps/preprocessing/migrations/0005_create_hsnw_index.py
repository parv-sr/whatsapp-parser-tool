# 4_create_hnsw_index.py
from django.db import migrations

class Migration(migrations.Migration):

    dependencies = [
        ("preprocessing", "0004_drop_json_column"),
    ]

    operations = [
        migrations.RunSQL(
            """
            CREATE INDEX IF NOT EXISTS idx_embeddingrecord_hnsw
            ON preprocessing_embeddingrecord
            USING hnsw (embedding_vector_v vector_cosine_ops)
            WITH (m = 16, ef_construction = 200);
            """,
            reverse_sql="DROP INDEX IF EXISTS idx_embeddingrecord_hnsw;"
        )
    ]
