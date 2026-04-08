# apps/ingestion/views.py
import logging
from celery import current_app as celery_current_app

import magic
import zipfile
from pathlib import Path

from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.core.files.base import ContentFile
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.cache import never_cache

from config.celery import app as project_celery_app
from .forms import MultiTxtUploadForm
from .models import RawFile

try:
    import rarfile
except Exception:
    rarfile = None

log = logging.getLogger(__name__)


def _progress_payload_for_file(raw_file):
    if raw_file.status == "COMPLETED":
        progress = 100
    elif raw_file.status in {"FAILED", "CANCELLED"}:
        progress = 0
    else:
        progress = int(cache.get(f"progress:{raw_file.id}", 0) or 0)
        if raw_file.status == "PROCESSING" and progress == 0:
            progress = 5

    if raw_file.status == "PENDING":
        stage = 1
        stage_label = "Queued"
    elif raw_file.status == "PROCESSING":
        if progress < 25:
            stage, stage_label = 2, "Parsing"
        elif progress < 85:
            stage, stage_label = 3, "Extracting"
        else:
            stage, stage_label = 4, "Finalizing"
    elif raw_file.status == "COMPLETED":
        stage, stage_label = 5, "Completed"
    elif raw_file.status == "CANCELLED":
        stage, stage_label = 0, "Cancelled"
    else:
        stage, stage_label = 0, "Failed"

    dedupe_stats = cache.get(f"dedupe_stats:{raw_file.id}", None)
    if dedupe_stats is None:
        dedupe_stats = raw_file.dedupe_stats or {}

    payload = {
        "id": raw_file.id,
        "progress": progress,
        "status": raw_file.status,
        "stage": stage,
        "stage_label": stage_label,
        "error": raw_file.notes,
        "runtime_logs": cache.get(f"runtime_logs:{raw_file.id}", []) or [],
        "dedupe_stats": dedupe_stats,
    }
    log.debug("progress payload generated: %s", payload)
    return payload


def _extract_archive_text(uploaded_file, ext):
    uploaded_file.seek(0)
    archive_name = Path(uploaded_file.name).name

    if ext == ".zip":
        with zipfile.ZipFile(uploaded_file) as zf:
            text_members = [
                info for info in zf.infolist()
                if not info.is_dir() and info.filename.lower().endswith(".txt")
            ]
            log.info("zip scan file=%s txt_members=%s", archive_name, [x.filename for x in text_members])
            if len(text_members) != 1:
                raise ValueError("ZIP must contain exactly one .txt file")
            member = text_members[0]
            raw_bytes = zf.read(member)
            extracted_name = Path(member.filename).name
    else:
        if rarfile is None:
            raise ValueError("RAR support is unavailable on this server")

        with rarfile.RarFile(uploaded_file) as rf:
            text_members = [
                info for info in rf.infolist()
                if not info.is_dir() and info.filename.lower().endswith(".txt")
            ]
            log.info("rar scan file=%s txt_members=%s", archive_name, [x.filename for x in text_members])
            if len(text_members) != 1:
                raise ValueError("RAR must contain exactly one .txt file")
            member = text_members[0]
            raw_bytes = rf.read(member)
            extracted_name = Path(member.filename).name

    if not raw_bytes:
        raise ValueError("Archive text file is empty")

    content = ContentFile(raw_bytes)
    content.name = extracted_name
    display_name = f"{archive_name} -> {extracted_name}"
    log.info("archive extracted source=%s extracted=%s size=%s", archive_name, extracted_name, len(raw_bytes))
    return display_name, content


def _normalize_input_files(raw_files):
    normalized = []
    errors = []

    for uploaded_file in raw_files:
        original_name = uploaded_file.name
        ext = Path(original_name).suffix.lower()
        log.info("normalizing upload source=%s ext=%s size=%s", original_name, ext, getattr(uploaded_file, "size", "unknown"))

        if ext == ".txt":
            normalized.append((original_name, uploaded_file))
            continue

        if ext in {".zip", ".rar"}:
            try:
                display_name, extracted_file = _extract_archive_text(uploaded_file, ext)
                normalized.append((display_name, extracted_file))
            except (ValueError, zipfile.BadZipFile, RuntimeError) as exc:
                msg = f"{original_name}: {exc}"
                log.warning("archive normalize failed source=%s err=%s", original_name, exc)
                errors.append(msg)
            except Exception as exc:
                msg = f"{original_name}: archive extraction error - {exc}"
                log.exception("archive normalize unexpected failure source=%s", original_name)
                errors.append(msg)
            continue

        errors.append(f"{original_name}: unsupported file type")
        log.warning("unsupported upload source=%s", original_name)

    return normalized, errors



@login_required
def upload_redirect(request):
    log.info("upload_redirect user_id=%s", request.user.id)
    return redirect("upload_files")


@login_required
def upload_success(request):
    uploaded_files = request.session.get("uploaded_files", [])
    processing_files = request.session.get("processing_files", [])
    log.info(
        "upload_success view user_id=%s uploaded=%s processing_ids=%s",
        request.user.id,
        len(uploaded_files),
        processing_files,
    )

    return render(
        request,
        "ingestion/upload_success.html",
        {
            "uploaded_files": uploaded_files,
            "processing_files": processing_files,
        },
    )


@login_required
def upload_files(request):
    if request.method == "POST":
        source_files = request.FILES.getlist("files")
        log.info("upload_files POST user_id=%s source_file_count=%s", request.user.id, len(source_files))

        errors = []
        created = []
        processing_ids = []

        normalized_files, normalize_errors = _normalize_input_files(source_files)
        errors.extend(normalize_errors)
        uploaded_files = [name for name, _ in normalized_files]
        request.session["uploaded_files"] = uploaded_files

        for display_name, file_obj in normalized_files:
            mime = "unknown"
            try:
                blob = file_obj.read(2048)
                file_obj.seek(0)
                mime = magic.from_buffer(blob, mime=True)
                if "text" not in mime and "plain" not in mime:
                    err = f"{display_name}: invalid text mime ({mime})"
                    errors.append(err)
                    log.warning("mime validation failed display=%s mime=%s", display_name, mime)
                    continue
            except Exception:
                log.exception("mime detection failed display=%s", display_name)

            try:
                with transaction.atomic():
                    rf = RawFile.objects.create(
                        file=file_obj,
                        file_name=display_name,
                        status="PENDING",
                        uploaded_at=timezone.now(),
                        owner=request.user,
                    )
                    created.append(rf)
                    processing_ids.append(rf.id)
                    cache.set(f"progress:{rf.id}", 0, timeout=3600)
                    cache.set(f"cancel:{rf.id}", False, timeout=3600)

                    active = cache.get("processing_files", [])
                    if rf.id not in active:
                        active.append(rf.id)
                        cache.set("processing_files", active, timeout=3600)

                    log.info("raw file created id=%s name=%s mime=%s active_ids=%s", rf.id, rf.file_name, mime, active)

                def schedule_task(file_id):
                    current_broker = getattr(celery_current_app.conf, "broker_url", "unknown")
                    project_broker = getattr(project_celery_app.conf, "broker_url", "unknown")
                    log.info(
                        "queueing celery task file_id=%s current_broker=%s project_broker=%s",
                        file_id,
                        current_broker,
                        project_broker,
                    )
                    try:
                        result = project_celery_app.send_task(
                            "apps.ingestion.tasks.process_file_task",
                            args=[file_id],
                            queue="celery",
                        )
                    except Exception as queue_err:
                        RawFile.objects.filter(pk=file_id).update(
                            status="FAILED",
                            notes="Task enqueue failed. Verify Celery broker/worker connectivity.",
                        )
                        cache.set(f"progress:{file_id}", 0, timeout=3600)
                        log.exception(
                            "task enqueue failed file_id=%s current_broker=%s project_broker=%s err=%s",
                            file_id,
                            current_broker,
                            project_broker,
                            queue_err,
                        )
                        return None

                    log.info("task queued file_id=%s task_id=%s", file_id, getattr(result, "id", None))
                    return result

                transaction.on_commit(lambda fid=rf.id: schedule_task(fid), robust=True)
                log.info("on_commit callback registered file_id=%s", rf.id)

            except Exception as exc:
                log.exception("failed to save/schedule file=%s", display_name)
                errors.append(f"{display_name}: Internal Error - {exc}")

        request.session["uploaded_files"] = uploaded_files
        request.session["processing_files"] = processing_ids
        log.info("session updated uploaded=%s processing_ids=%s errors=%s", uploaded_files, processing_ids, len(errors))

        return redirect("ingestion:upload_success") if not errors else render(
            request,
            "ingestion/upload_success.html",
            {
                "errors": errors,
                "created": created,
                "uploaded_files": uploaded_files,
                "processing_files": processing_ids,
            },
        )

    log.info("upload_files GET user_id=%s", request.user.id)
    form = MultiTxtUploadForm()
    return render(request, "ingestion/upload_form.html", {"form": form})

@login_required
@never_cache
def progress_status(request):
    ids_param = request.GET.get("ids", "")
    log.info("progress_status user_id=%s ids_param=%s", request.user.id, ids_param)

    if not ids_param:
        response = JsonResponse({"files": []})
        response["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response["Pragma"] = "no-cache"
        response["Expires"] = "0"
        return response

    try:
        target_ids = [int(x) for x in ids_param.split(",") if x.isdigit()]
    except ValueError:
        log.warning("progress_status invalid ids_param=%s", ids_param)
        return JsonResponse({"files": []}, status=400)

    files_qs = RawFile.objects.filter(pk__in=target_ids)
    files_by_id = {raw_file.id: raw_file for raw_file in files_qs}

    result = []
    for file_id in target_ids:
        raw_file = files_by_id.get(file_id)
        if not raw_file:
            log.warning("progress_status requested missing file_id=%s", file_id)
            continue
        result.append(_progress_payload_for_file(raw_file))

    response = JsonResponse({"files": result})
    response["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response["Pragma"] = "no-cache"
    response["Expires"] = "0"
    log.info("progress_status response_count=%s ids=%s", len(result), target_ids)
    return response


@login_required
def cancel_upload(request, file_id):
    if request.method != "POST":
        log.warning("cancel_upload non-post user_id=%s file_id=%s", request.user.id, file_id)
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        raw_file = RawFile.objects.get(pk=file_id)
    except RawFile.DoesNotExist:
        log.warning("cancel_upload missing file user_id=%s file_id=%s", request.user.id, file_id)
        return JsonResponse({"error": "File not found"}, status=404)

    if raw_file.owner_id and raw_file.owner_id != request.user.id:
        log.warning("cancel_upload forbidden user_id=%s file_id=%s owner_id=%s", request.user.id, file_id, raw_file.owner_id)
        return JsonResponse({"error": "Forbidden"}, status=403)

    if raw_file.status in {"COMPLETED", "FAILED", "CANCELLED"}:
        payload = _progress_payload_for_file(raw_file)
        log.info("cancel_upload ignored terminal status=%s file_id=%s", raw_file.status, raw_file.id)
        return JsonResponse({"ok": True, "file": payload})

    raw_file.status = "CANCELLED"
    raw_file.process_finished_at = timezone.now()
    raw_file.notes = "Cancelled by user"
    raw_file.save(update_fields=["status", "process_finished_at", "notes"])

    cache.set(f"cancel:{raw_file.id}", True, timeout=3600)
    cache.set(f"progress:{raw_file.id}", 0, timeout=3600)

    active = cache.get("processing_files", [])
    if raw_file.id in active:
        active.remove(raw_file.id)
        cache.set("processing_files", active, timeout=3600)

    payload = _progress_payload_for_file(raw_file)
    log.info("cancel_upload success user_id=%s file_id=%s", request.user.id, raw_file.id)
    return JsonResponse({"ok": True, "file": payload})


@login_required
def uploads_list(request):
    files = RawFile.objects.order_by("-uploaded_at")
    for file in files:
        payload = _progress_payload_for_file(file)
        file.initial_progress = payload["progress"]
        file.initial_stage = payload["stage"]
        file.initial_stage_label = payload["stage_label"]
        file.runtime_logs = payload.get("runtime_logs", [])
        file.dedupe_stats = payload.get("dedupe_stats", {})

    log.info("uploads_list user_id=%s total_files=%s", request.user.id, len(files))
    return render(request, "ingestion/uploads_list.html", {"files": files})
