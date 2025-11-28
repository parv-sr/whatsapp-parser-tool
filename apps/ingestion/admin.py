from django.contrib import admin
from .models import RawFile, RawMessageChunk

@admin.register(RawFile)
class RawFileAdmin(admin.ModelAdmin):
    list_display = ("file_name", "uploaded_at", "processed")
    readonly_fields = ("content", "uploaded_at", "process_started_at", "process_finished_at")

@admin.register(RawMessageChunk)
class RawMessageChunkAdmin(admin.ModelAdmin):
    list_display = ("rawfile", "sender", "message_start", "status", "created_at")
    search_fields = ("sender", "raw_text", "cleaned_text")
