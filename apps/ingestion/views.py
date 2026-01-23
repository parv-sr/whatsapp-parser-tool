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
    Refactored to fix race conditions and loop variable capture bugs.
    """
    if request.method == "POST":
        files = request.FILES.getlist("files")
        errors = []
        created = []

        uploaded_files = []
        processing_ids = []

        # Pre-fill uploaded filenames for the session
        for file in files:
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

            # --- CRITICAL FIX START ---
            # Use an atomic block to ensure the DB row is committed before the task starts.
            try:
                with transaction.atomic():
                    # 1. Create the DB row
                    rf = RawFile.objects.create(
                        file=f,
                        file_name=f.name,
                        status="PENDING",
                        uploaded_at=timezone.now(),
                        owner=request.user,
                    )
                    created.append(rf)
                    processing_ids.append(rf.id)

                    # 2. Update Cache (Progress tracking)
                    cache.set(f"progress:{rf.id}", 0)

                    active = cache.get("processing_files", [])
                    if rf.id not in active:
                        active.append(rf.id)
                        cache.set("processing_files", active)

                    # 3. Schedule Task safely
                    # FIX: `file_id=rf.id` binds the current ID to the lambda immediately.
                    # Without this, all lambdas would run with the ID of the *last* file in the loop.
                    transaction.on_commit(lambda file_id=rf.id: process_file_task.delay(file_id))
                    
                    log.info("Scheduled Celery task for RawFile id=%s", rf.id)

            except Exception as e:
                log.exception(f"Failed to save or schedule file {f.name}")
                errors.append(f"{f.name}: Internal Error")
            # --- CRITICAL FIX END ---

        # Update session with final lists
        request.session["uploaded_files"] = uploaded_files
        request.session["processing_files"] = processing_ids

        return redirect("ingestion:upload_success") if not errors else render(
            request,
            "ingestion/upload_success.html",
            {"errors": errors, "created": created}
        )

    # GET request handling
    form = MultiTxtUploadForm()
    return render(request, "ingestion/upload_form.html", {"form": form})

    
@login_required
def progress_status(request):
    """
    Returns progress for specific file IDs provided in the GET param 'ids'.
    Example: /ingestion/progress/?ids=1,2,3
    """
    ids_param = request.GET.get('ids', '')
    
    if not ids_param:
        return JsonResponse({"files": []})

    try:
        # Parse "1,2,3" into [1, 2, 3]
        target_ids = [int(x) for x in ids_param.split(',') if x.isdigit()]
    except ValueError:
        return JsonResponse({"files": []}, status=400)

    # Fetch DB objects (Source of Truth)
    files_qs = RawFile.objects.filter(pk__in=target_ids)
    
    result = []
    for f in files_qs:
        # 1. Check DB Status first (Ultimate truth)
        if f.status == "COMPLETED":
            prog = 100
        elif f.status == "FAILED":
            prog = 0
        else:
            # 2. If running, check Cache for granular 0-99%
            # We use the DB cache backend we set up earlier
            prog = cache.get(f"progress:{f.id}", 0)
            
            # Sanity check: If status is PROCESSING but cache is empty/0, 
            # show at least 5% so user knows it's alive.
            if f.status == "PROCESSING" and prog == 0:
                prog = 5

        result.append({
            "id": f.id,
            "progress": prog,
            "status": f.status,
            "error": f.notes
        })

    return JsonResponse({"files": result})


@login_required
def uploads_list(request):
    from .models import RawFile
    files = RawFile.objects.order_by("-uploaded_at")
    return render(request, "ingestion/uploads_list.html", {"files": files})

