from pathlib import Path

_MAX_CHARS = 4000  # keep system prompt manageable


def load_resume(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""

    if p.suffix.lower() == ".pdf":
        text = _read_pdf(p)
        if text:
            return text[:_MAX_CHARS]

    # Plain text fallback
    return p.read_text(encoding="utf-8", errors="ignore")[:_MAX_CHARS]


def _read_pdf(p: Path) -> str:
    # Try PyMuPDF first (fast)
    try:
        import fitz
        doc = fitz.open(str(p))
        return "\n".join(page.get_text() for page in doc)
    except ImportError:
        pass

    # Try pdfminer
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(p))
    except ImportError:
        pass

    return ""


def build_system_prompt(resume_text: str, extra: str = "") -> str:
    lines = [
        "You are a real-time interview assistant. The user will share interview questions "
        "or coding problems. Provide clear, confident, concise answers. "
        "Format code in markdown code blocks. Be direct — no filler preamble.",
    ]
    if resume_text:
        lines.append(
            f"\n## Candidate background (resume excerpt)\n{resume_text}"
        )
    if extra:
        lines.append(f"\n## Additional context\n{extra}")
    return "\n".join(lines)
