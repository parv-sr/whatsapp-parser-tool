from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ingestion", "0005_alter_rawfile_status"),
    ]

    operations = [
        migrations.AddField(
            model_name="rawfile",
            name="dedupe_stats",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
