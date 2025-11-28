# apps/ingestion/parsers/whatsapp_parser.py

import re
import unicodedata
from datetime import datetime
from django.utils import timezone
from ..models import RawFile, RawMessageChunk
from apps.ingestion.dedupe.pre_llm_dedupe import PreLLMDedupe

import regex as _regex

# ---------------------------------------------------------------------
# Normalization Helpers
# ---------------------------------------------------------------------

def _normalize_whitespace(s: str) -> str:
    if s is None:
        return s
    s = unicodedata.normalize("NFKC", s)

    replacements = {
        "\u00A0": " ",   # NBSP
        "\u202F": " ",   # narrow NBSP
        "\u200B": "",    # zero-width space
        "\u200E": "",    # LTR
        "\u200F": "",    # RTL
    }
    for k, v in replacements.items():
        s = s.replace(k, v)

    s = re.sub(r"[ \t]+", " ", s)
    return s


try:
    EMOJI_PATTERN = _regex.compile(r"\p{So}|\p{Sk}|\p{Cf}", flags=_regex.UNICODE)
except:
    EMOJI_PATTERN = re.compile(
        "[\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\u2600-\u26FF"
        "\u2700-\u27BF]+",
        re.UNICODE,
    )

def _strip_emojis(s: str) -> str:
    if not s:
        return s
    return EMOJI_PATTERN.sub(" ", s)


# ---------------------------------------------------------------------
# WhatsApp Header Detection
# ---------------------------------------------------------------------

HEADER_REGEX = re.compile(
    r"""
    ^[\u200E\u200F\u202A\u202B\u202C\u202D\u202E\s]*
    (?:\[)?                                          
    (?P<date>\d{1,2}/\d{1,2}/\d{2,4})                 
    [,\s]+
    (?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:AM|PM|am|pm))?) 
    (?:\])?
    [\s\-–—:]*                                      
    (?P<sender>[^:\n\r]{1,200}?)\s*:\s*               
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)


SYSTEM_REGEX = re.compile(
    r"(end-to-end encrypted|message deleted|joined using|left the group|created group|changed the subject|<media omitted>)",
    re.IGNORECASE,
)


def parse_datetime(date_str, time_str):
    if not date_str:
        return timezone.now()

    date_str = _normalize_whitespace(date_str)
    time_str = _normalize_whitespace(time_str) if time_str else ""

    combined = f"{date_str} {time_str}".strip()
    fmts = [
        "%d/%m/%y %I:%M %p",
        "%d/%m/%y %H:%M",
        "%d/%m/%Y %I:%M %p",
        "%d/%m/%Y %H:%M",
    ]
    for f in fmts:
        try:
            return datetime.strptime(combined, f)
        except:
            pass

    return timezone.now()


# ---------------------------------------------------------------------
# Split messages by header
# ---------------------------------------------------------------------
def _split_messages_whatsapp(text: str):
    text = _normalize_whitespace(text)
    matches = list(HEADER_REGEX.finditer(text))

    if not matches:
        yield None, text
        return

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        yield m, block


# ---------------------------------------------------------------------
# Clean each message block
# ---------------------------------------------------------------------
def clean_block(block: str) -> str:
    if not block:
        return ""

    block = _strip_emojis(block)
    block = _normalize_whitespace(block)

    # remove header part
    m = HEADER_REGEX.match(block)
    if m:
        block = block[m.end():].strip()

    # remove system lines
    lines = []
    for line in block.splitlines():
        if not SYSTEM_REGEX.search(line.strip()):
            lines.append(line)

    cleaned = "\n".join(lines).strip()
    return cleaned


# ---------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------
def process_rawfile_from_uploaded_file(rawfile: RawFile):
    rawfile.process_started_at = timezone.now()
    rawfile.save(update_fields=["process_started_at"])

    created = 0
    ignored = 0

    text = rawfile.content or ""

    deduper = PreLLMDedupe()

    for header, block in _split_messages_whatsapp(text):
        if header:
            date_raw = header.group("date")
            time_raw = header.group("time")
            sender = header.group("sender").strip()
            message_start = parse_datetime(date_raw, time_raw)
        else:
            sender = None
            message_start = None

        cleaned = clean_block(block)

        # If block is completely system text → ignore
        if not cleaned:
            RawMessageChunk.objects.create(
                rawfile=rawfile,
                message_start=message_start,
                sender=sender,
                raw_text=block[:10000],
                cleaned_text="",
                status="IGNORED",
            )
            ignored += 1
            continue

        # Pre-LLM dedupe check (drop duplicates inside same chat file)
        if not deduper.should_keep(cleaned):
            RawMessageChunk.objects.create(
                rawfile=rawfile,
                message_start=message_start,
                sender=sender,
                raw_text=block[:10000],
                cleaned_text=cleaned[:10000],
                status="DUPLICATE_LOCAL",
            )
            ignored += 1
            continue


        # Create NEW chunk (LLM takes over next stage)
        RawMessageChunk.objects.create(
            rawfile=rawfile,
            message_start=message_start,
            sender=sender,
            raw_text=block[:10000],
            cleaned_text=cleaned[:10000],
            status="NEW",
        )
        created += 1

    rawfile.process_finished_at = timezone.now()
    rawfile.processed = True
    rawfile.save(update_fields=["process_finished_at", "processed"])

    return {"created": created, "ignored": ignored}
