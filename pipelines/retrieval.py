class Retriever:
    def __init__(self, document): self.document = document
    def search(self, term, limit=10):
        q = term.lower()
        out = []
        for p in self.document.pages:
            hay = " ".join([
                p.text,
                " ".join(x.text for x in p.headings),
                " ".join(x.label for x in p.objects),
                " ".join(x.word for x in p.alphabet_items),
            ]).lower()
            score = sum(1 for token in q.split() if token in hay)
            if score:
                out.append({"page": p.page_number, "text": p.text, "page_data": p, "score": score})
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:limit]
    def page(self, n):
        return next((p for p in self.document.pages if p.page_number == n), None)
