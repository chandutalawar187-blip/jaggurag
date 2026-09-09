from .client import parse_json_safely
import re
from .schema import *

class DocumentExtractor:
    def __init__(self, client, loader):
        self.client, self.loader = client, loader
    def extract(self, loaded):
        pages = []
        for item in loaded["pages"]:
            n, text = item["page_number"], item["text"]
            visual_required = (
                item.get("image_count", 0) > 0
                or item.get("drawing_count", 0) > 0
                or len(text.strip()) < 40
                or "\ufffd" in text
            )
            analysis = {
                "page_number": n,
                "text": text,
                "visual_required": visual_required,
                "alphabet_items": _alphabet_items_from_text(text, n),
            }
            if visual_required:
                raw = self.client.analyze(self.loader.render_page(loaded, n),
                    _analysis_prompt(n, text))
                visual = parse_json_safely(raw)
                if isinstance(visual, dict):
                    allowed = getattr(PageAnalysis, "model_fields", None) or getattr(PageAnalysis, "__fields__", {})
                    for key, value in visual.items():
                        if key in allowed and value is not None:
                            analysis[key] = value
                analysis["page_number"] = n
                analysis["text"] = text
                analysis["visual_required"] = visual_required
                analysis["alphabet_items"] = _merge_alphabet_items(
                    analysis.get("alphabet_items", []), visual.get("alphabet_items", [])
                    if isinstance(visual, dict) else [], n
                )
            try:
                pages.append(PageAnalysis(**analysis))
            except Exception:
                # Preserve usable text if a provider returns malformed fields.
                pages.append(PageAnalysis(
                    page_number=n,
                    text=text,
                    visual_required=visual_required,
                    alphabet_items=analysis.get("alphabet_items", []),
                ))
        return Document(source=loaded["path"], pages=pages)


def _analysis_prompt(page: int, text: str) -> str:
    return f"""Analyze PDF page {page}. Return ONLY valid JSON matching this shape:
{{"headings":[],"objects":[{{"label":"","description":""}}],
"image_labels":[],"tables":[],"lists":[],"relationships":[],
"alphabet_items":[{{"letter":"","word":"","page":{page}}}]}}
Use null/empty arrays when unknown. Never invent details.
Extract visible labels and relationships between labels and pictured objects."""


def _alphabet_items_from_text(text: str, page: int) -> list[dict]:
    if not re.search(r"\b(?:vocabulary|alphabet|words?\s+pack)\b", text, re.I):
        return []
    words = []
    for line in text.splitlines():
        compact = re.sub(r"\s+", "", line).lower()
        if len(compact) >= 2 and compact.isalpha() and compact != "aa" and compact not in {
            "vocabulary", "words", "pack", "worksheetspack", "wordspack",
        }:
            words.append(compact)
    letter_match = re.match(r"\s*([A-Za-z])\1?\b", text, re.IGNORECASE)
    letter = letter_match.group(1).upper() if letter_match else None
    if not letter:
        return []
    return [
        {"letter": letter, "word": word, "page": page}
        for word in dict.fromkeys(words)
        if word != letter.lower() * 2
    ]


def _merge_alphabet_items(existing, visual, page):
    merged = []
    for item in list(existing or []) + list(visual or []):
        if not isinstance(item, dict):
            continue
        item = dict(item)
        item.setdefault("page", page)
        if item.get("letter") and item.get("word"):
            key = (str(item["letter"]).upper(), str(item["word"]).lower())
            if key not in {
                (str(x.get("letter", "")).upper(), str(x.get("word", "")).lower())
                for x in merged
            }:
                merged.append(item)
    return merged
