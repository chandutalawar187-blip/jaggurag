import re
class Calculator:
    @staticmethod
    def word_count(text): return len(re.findall(r"\b[\w'-]+\b", text))

class AnswerGenerator:
    def __init__(self, client=None): self.client = client
    def answer(self, question, route, document, retriever):
        if route == "alphabet_count":
            letter = re.search(r'\b([a-z])\b', question.lower())
            items = [
                item for page in document.pages for item in page.alphabet_items
                if not letter or item.letter.lower() == letter.group(1)
            ]
            return str(len(items))
        if route == "word_count":
            m = re.search(r'page\s+(\d+)', question.lower())
            if m:
                p = retriever.page(int(m.group(1)))
                return str(Calculator.word_count(p.text)) if p else "I couldn't find that page."
            return str(sum(Calculator.word_count(p.text) for p in document.pages))
        if route == "alphabet_list":
            letter = None if _is_broad_alphabet_request(question) else _question_letter(question)
            items = [
                item for page in document.pages for item in page.alphabet_items
                if not letter or item.letter.lower() == letter.group(1)
            ]
            if not items:
                return "I couldn't find alphabet items in the document."
            words = ", ".join(dict.fromkeys(item.word for item in items))
            if letter:
                return f"{letter.group(1).upper()} is for {words}."
            grouped = {}
            for item in items:
                grouped.setdefault(item.letter.upper(), [])
                if item.word not in grouped[item.letter.upper()]:
                    grouped[item.letter.upper()].append(item.word)
            return "\n".join(
                f"{key}: {', '.join(words)}" for key, words in grouped.items()
            )
        if route == "page_objects":
            m = re.search(r'page\s+(\d+)', question.lower())
            p = retriever.page(int(m.group(1))) if m else None
            if not p or not p.objects: return "I couldn't find objects for that page."
            return ", ".join(x.label for x in p.objects)
        if route == "reasoning" and self.client:
            context = "\n".join(
                f"Page {x['page']}:\n{x['text']}" for x in retriever.search(question)
            )
            if not context:
                return "I couldn't find enough information in the document to answer that."
            return self.client.analyze_text(
                f"Answer only from this document context. If unsupported, say so.\n"
                f"{context}\nQuestion: {question}"
            )
        return "I couldn't find enough information in the document to answer that."


def _question_letter(question):
    text = question.lower()
    match = re.search(
        r'\b([a-z])\s+(?:for|stands?\s+for|alphabet|letter)\b', text
    )
    if match:
        return match
    return re.search(r'\bfor\s+([a-z])\b', text)


def _is_broad_alphabet_request(question):
    text = question.lower()
    return bool(
        re.search(r'\ba\s*(?:to|-)\s*z\b', text)
        or re.search(r'\ball\s+(?:the\s+)?letters?\b', text)
        or re.search(r'\bletters?\s+(?:of|in)\s+the\s+alphabet\b', text)
    )
