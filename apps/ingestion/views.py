# apps/ingestion/views.py
import os
import magic
import logging
import uuid

from django.shortcuts import render, redirect
from django.utils import timezone
from django.core.cache import cache
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.db import transaction

from .forms import MultiTxtUploadForm
from .models import RawFile
from .tasks import process_file_task

log = logging.getLogger(__name__)

@login_required
def upload_redirect(request):
    return redirect('upload_files')


@login_required
def upload_success(request):
    uploaded_files = request.session.get("uploaded_files", [])
    processing_files = request.session.get("processing_files", [])

    return render(request, "ingestion/upload_success.html", {
        "uploaded_files": uploaded_files,
        "processing_files": processing_files,  # now contains IDs, not names
    })


@login_required
def upload_files(request):
    """
    Fire-and-forget upload view.
    """
    if request.method == "POST":
        files = request.FILES.getlist("files")
        errors = []
        created = []

        uploaded_files = []
        processing_ids = []     # <-- FIX: use IDs, not names

        for file in request.FILES.getlist("files"):
            uploaded_files.append(file.name)

        request.session["uploaded_files"] = uploaded_files

        for f in files:
            name_lower = f.name.lower()
            ext_ok = name_lower.endswith(".txt")

            # Optional mime check
            mime_ok = True
            try:
                blob = f.read(2048)
                f.seek(0)
                mime = magic.from_buffer(blob, mime=True)
                if "text" not in mime and "plain" not in mime and not ext_ok:
                    mime_ok = False
            except Exception:
                mime = "unknown"

            if not ext_ok and not mime_ok:
                errors.append(f"{f.name}: not a .txt file (mime={mime})")
                continue

            # Save DB row
            rf = RawFile.objects.create(
                file=f,
                file_name=f.name,
                status="PENDING",
                uploaded_at=timezone.now(),
                owner = request.user,
            )
            created.append(rf)
            # track by ID
            processing_ids.append(rf.id)
            # Save progress=0 in cache for this file
            cache.set(f"progress:{rf.id}", 0)

            # Also track which files are being processed
            active = cache.get("processing_files", [])
            if rf.id not in active:
                active.append(rf.id)
                cache.set("processing_files", active)

            transaction.on_commit(lambda: process_file_task.delay((rf.id)))
            log.info("Scheduled Celery task for RawFile id=%s", rf.id)

        # Save to session
        request.session["uploaded_files"] = uploaded_files
        request.session["processing_files"] = processing_ids

        return redirect("ingestion:upload_success") if not errors else render(
            request,
            "ingestion/upload_result.html",
            {"errors": errors, "created": created}
        )

    # GET
    form = MultiTxtUploadForm()
    return render(request, "ingestion/upload_form.html", {"form": form})

@login_required
def progress_status(request):
    processing_files = cache.get("processing_files", [])
    
    if not processing_files:
        return JsonResponse({"files": []})

    # Dynamic DB query for all files (source of truth)
    files_qs = RawFile.objects.filter(pk__in=processing_files)
    
    result = []
    for f in files_qs:
        # Get cache progress or default based on status
        prog = cache.get(f"progress:{f.id}", 0)
        if f.status == "PENDING":
            prog = 0  # Explicit for pending
        elif f.status == "COMPLETED":
            prog = 100
        elif f.status == "FAILED":
            prog = 0
        
        result.append({
            "id": f.id,
            "progress": prog,
            "status": f.status,
            "error": f.notes if f.status == "FAILED" else None
        })

    return JsonResponse({"files": result})

@login_required
def uploads_list(request):
    from .models import RawFile
    files = RawFile.objects.order_by("-uploaded_at")
    return render(request, "ingestion/uploads_list.html", {"files": files})

