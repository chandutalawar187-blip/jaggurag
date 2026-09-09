import json
import re
class QwenVisionClient:
    """Provider-neutral client; inject a callable for tests or use OpenAI-compatible SDK."""
    def __init__(self, api_key="", model="qwen-vl-plus", base_url=None, responder=None):
        self.api_key, self.model, self.base_url, self.responder = api_key, model, base_url, responder
    def analyze(self, image_bytes, prompt):
        if self.responder: return self.responder(image_bytes, prompt)
        if not self.api_key: raise RuntimeError("QWEN_API_KEY is not configured and no mock responder was supplied")
        try:
            from openai import OpenAI
        except ImportError:
            try:
                import httpx, base64
                r = httpx.post((self.base_url or "").rstrip("/") + "/chat/completions",
                    headers={"Authorization": "Bearer " + self.api_key},
                    json={"model": self.model, "temperature": 0, "messages": [{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image_bytes).decode()}}
                    ]}]}, timeout=90)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except ImportError:
                raise RuntimeError("Install openai or httpx to use Qwen, or inject a responder")
            except Exception as e:
                raise RuntimeError("Qwen vision request failed: " + str(e)) from e
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            r = client.chat.completions.create(model=self.model, messages=[{"role":"user","content":[
                {"type":"text","text":prompt}, {"type":"image_url","image_url":{"url":"data:image/png;base64," + __import__("base64").b64encode(image_bytes).decode()}}
            ]}], temperature=0)
            return r.choices[0].message.content
        except Exception as e:
            raise RuntimeError("Qwen vision request failed: " + str(e)) from e

    def analyze_text(self, prompt):
        """Run a grounded text-only request through the same provider adapter."""
        return self._request(prompt, None)

    def _request(self, prompt, image_bytes):
        if self.responder:
            return self.responder(image_bytes or b"", prompt)
        if not self.api_key:
            raise RuntimeError("QWEN_API_KEY is not configured and no mock responder was supplied")
        import base64
        content = [{"type": "text", "text": prompt}]
        if image_bytes:
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(image_bytes).decode()
            }})
        try:
            from openai import OpenAI
            response = OpenAI(api_key=self.api_key, base_url=self.base_url).chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": content}], temperature=0
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise RuntimeError("Qwen text request failed: " + str(exc)) from exc

def parse_json_safely(raw):
    if isinstance(raw, dict): return raw
    text = str(raw).strip().replace("```json", "").replace("```", "").strip()
    try: return json.loads(text)
    except Exception: pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        try: return json.loads(candidate)
        except Exception: pass
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))
        except Exception: pass
    return {}
