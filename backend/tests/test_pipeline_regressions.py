from types import SimpleNamespace

from app.models.schemas import ChatRequest, ChatResponse
from app.rag import pipeline


def test_process_query_uses_processed_query_for_retrieval(monkeypatch):
    seen = {}

    def fake_retrieve_candidates(query, processed_query):
        seen["query"] = query
        seen["processed_query"] = processed_query
        return [], 0, 0

    def fake_general_response(query, generate_fn, model_name, conversation_id, start_time, retrieval=None):
        return ChatResponse(
            answer="fallback",
            answer_type="general_knowledge",
            confidence=0.0,
            model=model_name,
            response_time_ms=0,
            conversation_id=conversation_id,
        )

    monkeypatch.setattr(pipeline, "synchronize_dataset", lambda: None)
    monkeypatch.setattr(pipeline, "_is_database_summary_request", lambda q: False)
    monkeypatch.setattr(pipeline, "_preprocess_query", lambda q, conversation_id: "previous question current question")
    monkeypatch.setattr(pipeline, "chroma_count", lambda: 1)
    monkeypatch.setattr(pipeline, "_is_document_contents_request", lambda q: False)
    monkeypatch.setattr(pipeline, "find_documents_by_name", lambda q: [])
    monkeypatch.setattr(pipeline, "_is_alphabet_query", lambda q: False)
    monkeypatch.setattr(pipeline, "_retrieve_candidates", fake_retrieve_candidates)
    monkeypatch.setattr(pipeline, "rerank", lambda q, chunks, top_k=None: [])
    monkeypatch.setattr(pipeline, "calculate_confidence", lambda chunks, q: 0.0)
    monkeypatch.setattr(pipeline, "_has_meaningful_evidence", lambda q, chunks: False)
    monkeypatch.setattr(pipeline, "_get_llm_functions", lambda model: (lambda *args, **kwargs: "", lambda *args, **kwargs: "", "local_qwen"))
    monkeypatch.setattr(pipeline, "_general_knowledge_response", fake_general_response)

    request = ChatRequest(message="What is the answer?", model="local_qwen", conversation_id="abc")
    pipeline.process_query(request)

    assert seen["query"] == "What is the answer?"
    assert seen["processed_query"] == "previous question current question"
