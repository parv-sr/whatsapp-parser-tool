import bleach

ALLOWED_TAGS = [
    "table", "thead", "tbody", "tr", "th", "td",
    "p", "br",
    "strong", "b", "em", "i",
    "ul", "ol", "li",
]

ALLOWED_ATTRIBUTES = {
    "*": ["class"],
    "table": ["border", "cellpadding", "cellspacing"],
    "th": ["colspan", "rowspan"],
    "td": ["colspan", "rowspan"],
}

def clean_html(dirty: str) -> str:
    """
    Sanitize LLM output so no unsafe or unsupported HTML is rendered in Django.
    Removes scripts, events, SVG, iframe, style, click handlers, etc.
    """
    if not dirty:
        return ""

    return bleach.clean(
        dirty,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        strip=True,
    )
