# 2_backfill_vectors.py
import json
from django.db import migrations, connection

def backfill(apps, schema_editor):

    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT id, embedding_vector
            FROM preprocessing_embeddingrecord
            WHERE embedding_vector IS NOT NULL;
        """)

        records = cursor.fetchall()

        for rec_id, vec_json in records:
            try:
                vec = json.loads(vec_json)
                if not isinstance(vec, list):
                    continue

                # pgvector text format: '[1,2,3]'
                vector_literal = "[" + ",".join(str(float(x)) for x in vec) + "]"

                cursor.execute(
                    """
                    UPDATE preprocessing_embeddingrecord
                    SET embedding_vector_v = %s
                    WHERE id = %s;
                    """,
                    [vector_literal, rec_id]
                )

            except Exception:
                continue



class Migration(migrations.Migration):

    dependencies = [
        ("preprocessing", "0002_add_pgvector_column"),  # FIX NAME
    ]

    operations = [
        migrations.RunPython(backfill, reverse_code=migrations.RunPython.noop)
    ]
