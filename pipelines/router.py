import re
class QuestionRouter:
    def route(self, q):
        s = q.lower()
        if re.search(r'\b(count|how many)\b', s) and re.search(r'alphabet|letter|word', s): return "alphabet_count"
        if re.search(r'\b(word|words)\b', s) and re.search(r'\b(count|how many)\b', s): return "word_count"
        if re.search(r'\b(?:for\s+)?[a-z]\s+(?:for|stands?\s+for|alphabet|letter)\b', s): return "alphabet_list"
        if re.search(r'\b(?:a\s*(?:to|-)\s*z|all\s+(?:the\s+)?letters?|letters?\s+(?:of|in)\s+the\s+alphabet)\b', s) and re.search(r'\b(word|words|list|alphabet)\b', s):
            return "alphabet_list"
        if re.search(r'\b(list|which|what)\b', s) and re.search(r'alphabet|letter|words?', s): return "alphabet_list"
        if re.search(r'\b(page|on page)\b', s) and re.search(r'object|shown|see', s): return "page_objects"
        if re.search(r'\b(why|how|explain|teach|reason)\b', s): return "reasoning"
        return "generic"
