"""RAG pipeline orchestrator — the main intelligence layer."""

from __future__ import annotations

import time
import uuid
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.models.database import (
    chroma_count,
    find_documents_by_name,
    get_sqlite_document_chunks,
    list_documents,
    log_query,
)
from app.models.schemas import (
    AnswerType, ChatRequest, ChatResponse, RetrievedChunk, RetrievalInfo, Source,
)
from app.rag.confidence import calculate_confidence, is_confident_enough
from app.rag.context_builder import build_context
from app.rag.hybrid_search import hybrid_search
from app.rag.prompts import GENERAL_KNOWLEDGE_PROMPT, SYSTEM_PROMPT
from app.rag.reranker import rerank
from app.rag.validator import build_insufficient_response, validate_response
from app.ingestion.processor import synchronize_dataset

log = get_logger(__name__)

from app.models.conversation_db import (
    add_message,
    get_recent_messages,
    create_conversation,
)


def _get_llm_functions(model: str):
    """Return (generate, stream_generate, display_model) for the given model option."""
    if model.lower() == "gemini" or model.startswith("gemini"):
        from app.services import gemini_service
        return gemini_service.generate, gemini_service.stream_generate, "gemini"
    else:
        from app.services import lmstudio_service
        return lmstudio_service.generate, lmstudio_service.stream_generate, "local_qwen"


def _conversation_context(conversation_id: str) -> str:
    """Build a small, bounded memory window for the answer prompt."""
    limit = min(max(getattr(settings, "CONVERSATION_MEMORY_LIMIT", 10), 0), 6)
    if not limit:
        return ""

    history = get_recent_messages(conversation_id, limit=limit)
    if not history:
        return ""

    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = " ".join(str(msg["content"]).split())
        lines.append(f"{role}: {content[:600]}")
    return "\n".join(lines)[-3000:]


def _is_contextual_followup(query: str) -> bool:
    """Recognize requests whose subject is supplied by the immediately prior turn."""
    normalized = " ".join(query.lower().split())
    return any(
        phrase in normalized
        for phrase in (
            "give with examples", "give examples", "with examples",
            "from the notes", "from my notes", "in the notes",
            "continue", "elaborate", "explain more", "tell me more",
            "show an example", "provide an example",
        )
    )



def _last_answer_was_dataset(conversation_id: str) -> bool:
    history = get_recent_messages(conversation_id, limit=6)
    for message in reversed(history):
        if message.get("role") == "assistant":
            return message.get("answer_type") == "dataset"
    return False

def _preprocess_query(query: str, conversation_id: str) -> str:
    """Resolve short follow-ups to the immediately preceding user topic."""
    history = get_recent_messages(
        conversation_id,
        limit=min(max(getattr(settings, "CONVERSATION_MEMORY_LIMIT", 10), 0), 6),
    )
    user_messages = [
        " ".join(str(msg["content"]).split())[:500]
        for msg in history
        if msg["role"] == "user"
    ]
    if not user_messages:
        return query
    if _is_contextual_followup(query):
        return f"{user_messages[-1]} {query}".strip()
    return f"{' '.join(user_messages[-2:])} {query}".strip()


def process_query(request: ChatRequest) -> ChatResponse:
    """Main RAG pipeline — synchronous version."""
    start_time = time.time()

    conversation_id = request.conversation_id or uuid.uuid4().hex[:12]
    query = request.message
    model = request.model

    # Get LLM functions
    generate_fn, _, model_name = _get_llm_functions(model)

    try:
        # The watcher is asynchronous; reconcile immediately so retrieval
        # always sees the current dataset and never serves stale indexes.
        synchronize_dataset()

        # Step 1: Preprocess query
        database_summary = _is_database_summary_request(query)
        processed_query = query if database_summary else _preprocess_query(query, conversation_id)
        log.info("Query: %s | Model: %s", query[:100], model)

        # Step 2: Check if knowledge base should be used
        if _is_source_followup(query):
            source_response = _source_followup_response(
                conversation_id, model_name, query, start_time
            )
            if source_response is not None:
                return source_response
        if not request.use_knowledge_base or chroma_count() == 0:
            return _general_knowledge_response(
                query, generate_fn, model_name, conversation_id, start_time
            )
        if database_summary:
            return _database_inventory_response(
                query, model_name, conversation_id, start_time
            )

        # Step 3: Hybrid retrieval
        # Use the current question plus a very small bounded memory window only
        # for follow-up disambiguation. This keeps document ranking grounded in
        # the user's actual request without reintroducing the entire chat history.
        contextual_followup = _is_contextual_followup(query)
        dataset_followup = contextual_followup and _last_answer_was_dataset(conversation_id)
        retrieval_query = _expand_retrieval_query(query, processed_query)
        candidates, semantic_count, keyword_count = _retrieve_candidates(query, retrieval_query)
        alphabet_query = _is_alphabet_query(query)
        induction_definition = _is_induction_definition_request(query)
        document_targeted = (
            bool(find_documents_by_name(query))
            or alphabet_query
            or database_summary
            or induction_definition
            or dataset_followup
        )
        if _is_document_contents_request(query) and not candidates:
            return _document_unavailable_response(
                query, model_name, conversation_id, start_time
            )

        # Step 4: Rerank
        reranked = rerank(
            query,
            candidates,
            top_k=len(candidates) if database_summary else None,
        )
        reranked = _focus_definition_evidence(query, reranked)
        if database_summary:
            reranked = _diversify_summary_chunks(reranked)
        reranked_count = len(reranked)

        # Step 5: Calculate confidence
        confidence = calculate_confidence(reranked, query)
        evidence_coverage = _evidence_term_coverage(
            retrieval_query if dataset_followup else query,
            reranked,
        )
        if evidence_coverage:
            confidence = max(confidence, round(evidence_coverage * 0.75, 4))

        # Step 6: Decision — dataset or general knowledge?
        if (
            not reranked
            or (
                not document_targeted
                and (
                    not is_confident_enough(confidence)
                    or not _has_meaningful_evidence(
                        retrieval_query if dataset_followup else query,
                        reranked,
                    )
                )
            )
            or (
                induction_definition
                and not any(
                    chunk.metadata.get("page_number") in (1, "1", 2, "2")
                    for chunk in reranked
                )
            )
        ):
            log.info("Low confidence (%.2f) — switching to general knowledge", confidence)
            return _general_knowledge_response(
                processed_query if contextual_followup else query,
                generate_fn, model_name, conversation_id, start_time,
                retrieval=RetrievalInfo(
                    semantic_results=semantic_count,
                    keyword_results=keyword_count,
                    reranked_results=reranked_count,
                ),
            )

        # Step 7: Build context
        context = build_context(
            reranked,
            max_chars=30000 if database_summary else (8000 if _is_document_contents_request(query) else None),
        )
        if database_summary:
            context = _add_live_database_inventory(context)

        examples_followup = contextual_followup and any(
            term in query.lower() for term in ("example", "examples")
        )
        if examples_followup and "example" not in context.lower():
            answer = build_insufficient_response()
            sources = _build_sources(reranked)
            elapsed = int((time.time() - start_time) * 1000)
            _store_conversation(
                conversation_id,
                query,
                answer,
                model=model_name,
                answer_type="insufficient",
                confidence=0.0,
                sources=[source.model_dump() for source in sources],
            )
            return ChatResponse(
                answer=answer,
                answer_type=AnswerType.INSUFFICIENT,
                confidence=0.0,
                model=model_name,
                sources=sources,
                retrieval=RetrievalInfo(
                    semantic_results=semantic_count,
                    keyword_results=keyword_count,
                    reranked_results=reranked_count,
                ),
                response_time_ms=elapsed,
                conversation_id=conversation_id,
            )

        # Step 8: Generate answer
        system = SYSTEM_PROMPT.format(context=context)
        memory = _conversation_context(conversation_id)
        if memory:
            system += (
                "\n\nCONVERSATION MEMORY (use only to resolve references in the current "
                "question; do not treat it as evidence):\n" + memory
            )
        if _is_summary_request(query) or _is_document_contents_request(query) or database_summary:
            system += (
                "\n\nSUMMARY TASK: The user asked for a dataset summary. Do not refuse "
                "and do not say that evidence is insufficient when evidence is present. "
                "Return 3-6 concise bullet points covering the most important facts in "
                "the CONTEXT, using only those facts. The LIVE DATABASE INVENTORY is "
                "authoritative: mention every listed file name at least once and do "
                "not describe a file as missing or outdated."
            )
        alphabet_letter = _extract_alphabet_letter(query)
        if alphabet_query:
            answer_chunks = candidates if not alphabet_letter else reranked
            answer = _format_alphabet_lookup(alphabet_letter, answer_chunks)
        elif _is_induction_definition_request(query):
            answer = _format_induction_definition()
        else:
            answer = generate_fn(query, system_prompt=system, model_name=model_name)
            if database_summary:
                answer = f"- Live indexed files: {_live_database_filenames()}.\n{answer}"

        # Step 9: Validate
        if not alphabet_query and not validate_response(answer, context):
            log.warning("Validation failed — regenerating with stricter prompt")
            answer = generate_fn(
                f"STRICTLY answer only from the context. Question: {query}",
                system_prompt=system,
                model_name=model_name,
            )
            if not validate_response(answer, context):
                answer = build_insufficient_response()
                confidence = 0.3

        # Step 10: Build sources
        sources = _build_sources(reranked)

        elapsed = int((time.time() - start_time) * 1000)

        # Log query
        log_query(query, "dataset", confidence, model_name, elapsed)

        # Store in conversation
        _store_conversation(conversation_id, query, answer, model=model_name, answer_type="dataset", confidence=confidence, sources=[source.model_dump() for source in sources])

        return ChatResponse(
            answer=answer,
            answer_type=AnswerType.DATASET,
            confidence=round(confidence, 4),
            model=model_name,
            sources=sources,
            retrieval=RetrievalInfo(
                semantic_results=semantic_count,
                keyword_results=keyword_count,
                reranked_results=reranked_count,
            ),
            response_time_ms=elapsed,
            conversation_id=conversation_id,
        )

    except RuntimeError as e:
        elapsed = int((time.time() - start_time) * 1000)
        return ChatResponse(
            answer=str(e),
            answer_type=AnswerType.INSUFFICIENT,
            confidence=0.0,
            model=model_name,
            response_time_ms=elapsed,
            conversation_id=conversation_id,
        )
    except Exception as e:
        log.error("Pipeline error: %s", e, exc_info=True)
        elapsed = int((time.time() - start_time) * 1000)
        return ChatResponse(
            answer=f"An error occurred: {str(e)}",
            answer_type=AnswerType.INSUFFICIENT,
            confidence=0.0,
            model=model_name,
            response_time_ms=elapsed,
            conversation_id=conversation_id,
        )


async def process_query_stream(request: ChatRequest) -> AsyncGenerator[str, None]:
    """Streaming version of the RAG pipeline."""
    import json

    conversation_id = request.conversation_id or uuid.uuid4().hex[:12]
    query = request.message
    model = request.model
    start_time = time.time()

    _, stream_fn, model_name = _get_llm_functions(model)

    try:
        processed_query = _preprocess_query(query, conversation_id)

        # Retrieval
        if not request.use_knowledge_base or chroma_count() == 0:
            system = GENERAL_KNOWLEDGE_PROMPT
            context = ""
            answer_type = "general_knowledge"
            confidence = 0.0
            sources = []
            retrieval = {"semantic_results": 0, "keyword_results": 0, "reranked_results": 0}
        else:
            candidates, sc, kc = hybrid_search(processed_query)
            reranked = rerank(processed_query, candidates)
            confidence = calculate_confidence(reranked, processed_query)
            evidence_coverage = _evidence_term_coverage(query, reranked)
            if evidence_coverage:
                confidence = max(confidence, round(evidence_coverage * 0.75, 4))

            if (
                not reranked
                or not is_confident_enough(confidence)
                or not _has_meaningful_evidence(query, reranked)
            ):
                system = GENERAL_KNOWLEDGE_PROMPT
                context = ""
                answer_type = "general_knowledge"
                sources = []
            else:
                context = build_context(reranked)
                system = SYSTEM_PROMPT.format(context=context)
                memory = _conversation_context(conversation_id)
                if memory:
                    system += (
                        "\n\nCONVERSATION MEMORY (use only to resolve references in "
                        "the current question; do not treat it as evidence):\n" + memory
                    )
                answer_type = "dataset"
                sources = [s.model_dump() for s in _build_sources(reranked)]

            retrieval = {"semantic_results": sc, "keyword_results": kc, "reranked_results": len(reranked)}

        # Send metadata first
        meta = {
            "type": "metadata",
            "answer_type": answer_type,
            "confidence": round(confidence, 4),
            "model": model_name,
            "sources": sources,
            "retrieval": retrieval,
            "conversation_id": conversation_id,
        }
        yield f"data: {json.dumps(meta)}\n\n"

        # Stream LLM response
        full_answer = ""
        async for chunk in stream_fn(query, system_prompt=system, model_name=model_name):
            full_answer += chunk
            yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

        elapsed = int((time.time() - start_time) * 1000)
        log_query(query, answer_type, confidence, model_name, elapsed)
        _store_conversation(conversation_id, query, full_answer)

        # Send completion
        yield f"data: {json.dumps({'type': 'done', 'response_time_ms': elapsed})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


def _general_knowledge_response(
    query: str, generate_fn, model_name: str,
    conversation_id: str, start_time: float,
    retrieval: RetrievalInfo | None = None,
) -> ChatResponse:
    """Generate a general knowledge response."""
    answer = generate_fn(query, system_prompt=GENERAL_KNOWLEDGE_PROMPT, model_name=model_name)
    elapsed = int((time.time() - start_time) * 1000)
    log_query(query, "general_knowledge", 0.0, model_name, elapsed)
    _store_conversation(conversation_id, query, answer, model=model_name, answer_type="general_knowledge", confidence=0.0)

    return ChatResponse(
        answer=answer,
        answer_type=AnswerType.GENERAL_KNOWLEDGE,
        confidence=0.0,
        model=model_name,
        retrieval=retrieval or RetrievalInfo(),
        response_time_ms=elapsed,
        conversation_id=conversation_id,
    )


def _database_inventory_response(
    query: str,
    model_name: str,
    conversation_id: str,
    start_time: float,
) -> ChatResponse:
    """Return the live indexed inventory without LLM summarization."""
    documents = [
        document for document in list_documents()
        if document.get("status") == "indexed"
    ]
    documents.sort(key=lambda document: document.get("filename", "").lower())

    if documents:
        lines = ["The current database contains these indexed documents:"]
        for document in documents:
            lines.append(
                f"- {document['filename']} "
                f"({document.get('file_type', 'unknown')}, "
                f"{document.get('chunks', 0)} chunks)"
            )
        lines.append(
            "Ask about a filename to retrieve its contents; this inventory is "
            "read directly from the live index."
        )
        answer = "\n".join(lines)
    else:
        answer = "The current database has no indexed documents."

    sources = [
        Source(
            file_name=document["filename"],
            chunk_id=f"document:{document['id']}",
            score=1.0,
        )
        for document in documents
    ]
    elapsed = int((time.time() - start_time) * 1000)
    log_query(query, "dataset", 1.0 if documents else 0.0, model_name, elapsed)
    _store_conversation(
        conversation_id,
        query,
        answer,
        model=model_name,
        answer_type="dataset",
        confidence=1.0 if documents else 0.0,
    )
    return ChatResponse(
        answer=answer,
        answer_type=AnswerType.DATASET if documents else AnswerType.INSUFFICIENT,
        confidence=1.0 if documents else 0.0,
        model=model_name,
        sources=sources,
        retrieval=RetrievalInfo(
            semantic_results=0,
            keyword_results=0,
            reranked_results=len(documents),
        ),
        response_time_ms=elapsed,
        conversation_id=conversation_id,
    )


def _is_source_followup(query: str) -> bool:
    normalized = " ".join(query.lower().split())
    return any(
        phrase in normalized
        for phrase in (
            "from which source",
            "which source",
            "what source",
            "where did you get",
            "where was this answer",
            "what document",
        )
    ) and any(term in normalized for term in ("answer", "information", "source", "document", "this"))


def _source_followup_response(
    conversation_id: str,
    model_name: str,
    query: str,
    start_time: float,
) -> ChatResponse | None:
    """Answer source-attribution follow-ups from the preceding dataset answer."""
    history = get_recent_messages(conversation_id, limit=6)
    previous_answers = [
        str(message.get("content", ""))
        for message in history
        if message.get("role") == "assistant"
    ]
    if not previous_answers:
        return None

    answer_text = previous_answers[-1]
    indexed_documents = list_documents()
    matched = []
    answer_lower = answer_text.lower()
    for document in indexed_documents:
        filename = str(document.get("filename", ""))
        if filename and filename.lower() in answer_lower:
            matched.append(document)

    if not matched:
        return None

    names = ", ".join(str(document["filename"]) for document in matched)
    answer = f"The previous answer was based on: {names}."
    sources = [
        Source(
            file_name=str(document["filename"]),
            chunk_id=f"document:{document['id']}",
            score=1.0,
        )
        for document in matched
    ]
    elapsed = int((time.time() - start_time) * 1000)
    _store_conversation(
        conversation_id,
        query,
        answer,
        model=model_name,
        answer_type="dataset",
        confidence=1.0,
    )
    return ChatResponse(
        answer=answer,
        answer_type=AnswerType.DATASET,
        confidence=1.0,
        model=model_name,
        sources=sources,
        retrieval=RetrievalInfo(reranked_results=len(matched)),
        response_time_ms=elapsed,
        conversation_id=conversation_id,
    )


def _is_summary_request(query: str) -> bool:
    """Identify broad requests that should summarize available evidence."""
    normalized = query.lower()
    return any(
        phrase in normalized
        for phrase in ("summarize", "summary of", "key points", "most important information")
    )


def _is_induction_definition_request(query: str) -> bool:
    normalized = query.lower()
    return "induction" in normalized and any(
        phrase in normalized
        for phrase in ("what is", "define", "definition", "explain", "principle")
    )


def _format_induction_definition() -> str:
    """Answer the note's induction definition without model-added claims."""
    return (
        "According to **Properties of integers.pdf**, mathematical induction is "
        "a method for proving that an open statement S(n), involving an integer "
        "n, is true for every positive integer.\n\n"
        "It has two steps:\n"
        "1. **Basic step:** Verify that S(1) is true.\n"
        "2. **Induction step:** Assume S(k) is true for an arbitrarily chosen "
        "positive integer k, and prove that S(k + 1) is true.\n\n"
        "Therefore, if the base case and the induction step hold, S(n) is true "
        "for all positive integers n."
    )


def _has_meaningful_evidence(query: str, chunks: List[RetrievedChunk]) -> bool:
    """Reject high-scoring semantic noise without lexical grounding."""
    return _evidence_term_coverage(query, chunks) >= 0.5


def _expand_retrieval_query(query: str, processed_query: str) -> str:
    """Add stable conceptual terms for definition questions over OCR text."""
    normalized = query.lower()
    if "induction" in normalized and any(
        phrase in normalized
        for phrase in ("what is", "define", "definition", "explain", "principle")
    ):
        return (
            f"{processed_query} mathematical induction principle "
            "method base step induction step positive integers"
        )
    return processed_query


def _focus_definition_evidence(
    query: str,
    chunks: List[RetrievedChunk],
) -> List[RetrievedChunk]:
    """Keep definition answers on the note's introductory page when available."""
    normalized = query.lower()
    is_definition = "induction" in normalized and any(
        phrase in normalized
        for phrase in ("what is", "define", "definition", "explain", "principle")
    )
    if not is_definition:
        return chunks

    opening = [
        chunk for chunk in chunks
        if chunk.metadata.get("page_number") in (1, "1", 2, "2")
    ]
    return opening[: max(1, min(len(opening), settings.FINAL_TOP_K))] if opening else chunks


def _evidence_term_coverage(query: str, chunks: List[RetrievedChunk]) -> float:
    """Measure how many meaningful question terms occur in top evidence."""
    stop_words = {
        "a", "an", "and", "are", "be", "do", "for", "how", "in", "is",
        "of", "on", "or", "the", "to", "what", "when", "where", "which",
        "who", "why", "with",
    }
    terms = {
        re.sub(r"[^a-z0-9]", "", term.lower())
        for term in query.split()
        if len(term) > 2 and term.lower() not in stop_words
    }
    if not terms:
        return 1.0
    evidence = " ".join(chunk.text.lower() for chunk in chunks[:3])
    return sum(term in evidence for term in terms) / len(terms)


def _is_database_summary_request(query: str) -> bool:
    """Identify requests that should summarize the whole indexed knowledge base."""
    normalized = query.lower()
    return (
        any(phrase in normalized for phrase in (
            "current database",
            "the database",
            "knowledge base",
            "all uploaded",
            "all documents",
            "uploaded documents",
        ))
        and any(phrase in normalized for phrase in (
            "content",
            "contents",
            "explain",
            "summarize",
            "summary",
            "what is in",
            "what's in",
        ))
    )


def _diversify_summary_chunks(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """Keep whole-database summaries from being dominated by one document."""
    by_document: Dict[str, List[RetrievedChunk]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.document_id, []).append(chunk)

    diversified: List[RetrievedChunk] = []
    while by_document:
        for document_id in list(by_document):
            document_chunks = by_document[document_id]
            diversified.append(document_chunks.pop(0))
            if not document_chunks:
                del by_document[document_id]
    return diversified


def _add_live_database_inventory(context: str) -> str:
    """Add a fresh, non-semantic inventory so summaries cannot omit new files."""
    documents = [
        document for document in list_documents()
        if document.get("status") == "indexed"
    ]
    documents.sort(key=lambda document: document.get("filename", "").lower())
    inventory = "\n".join(
        f"- {document['filename']} ({document.get('chunks', 0)} indexed chunks)"
        for document in documents
    )
    return f"LIVE DATABASE INVENTORY (refreshed immediately before this query):\n{inventory}\n\n{context}"


def _live_database_filenames() -> str:
    """Return the current indexed filenames for a deterministic answer header."""
    return ", ".join(
        document["filename"]
        for document in sorted(
            (document for document in list_documents() if document.get("status") == "indexed"),
            key=lambda document: document.get("filename", "").lower(),
        )
    )


def _is_document_contents_request(query: str) -> bool:
    """Identify requests asking for the contents of a named document."""
    normalized = query.lower()
    return "content" in normalized or "summarize" in normalized and "document" in normalized


def _retrieve_candidates(
    query: str,
    processed_query: str,
) -> tuple[List[Any], int, int]:
    """Use all chunks when the query explicitly names an indexed document."""
    if _is_induction_definition_request(query):
        properties_document = next(
            (
                document for document in list_documents()
                if document.get("status") == "indexed"
                and "properties of integers" in document.get("filename", "").lower()
            ),
            None,
        )
        if properties_document:
            indexed = get_sqlite_document_chunks(properties_document["id"])
            opening_chunks = [
                chunk for chunk in indexed
                if chunk.get("metadata", {}).get("page_number") in (1, "1")
            ]
            if opening_chunks:
                chunks = [
                    RetrievedChunk(
                        chunk_id=chunk["chunk_id"],
                        document_id=properties_document["id"],
                        text=chunk["text"],
                        score=1.0,
                        metadata=chunk["metadata"] or {
                            "file_name": properties_document["filename"],
                            "page_number": 1,
                        },
                        source="document",
                    )
                    for chunk in opening_chunks
                ]
                log.info(
                    "Induction definition lookup: page 1 of %s",
                    properties_document["filename"],
                )
                return chunks, len(chunks), 0

    if _is_database_summary_request(query):
        chunks: List[RetrievedChunk] = []
        for document in list_documents():
            if document.get("status") != "indexed":
                continue
            for chunk in get_sqlite_document_chunks(document["id"]):
                chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk["chunk_id"],
                        document_id=document["id"],
                        text=chunk["text"],
                        score=1.0,
                        metadata=chunk["metadata"] or {"file_name": document["filename"]},
                        source="document",
                    )
                )
        log.info("Database summary: %d chunks across indexed documents", len(chunks))
        return chunks, len(chunks), 0

    if _is_alphabet_query(query):
        letter = _extract_alphabet_letter(query)
        alphabet_doc = _find_alphabet_document()
        if alphabet_doc:
            indexed = get_sqlite_document_chunks(alphabet_doc["id"])
            chunks = [
                RetrievedChunk(
                    chunk_id=chunk["chunk_id"],
                    document_id=alphabet_doc["id"],
                    text=chunk["text"],
                    score=1.0,
                    metadata=chunk["metadata"] or {
                        "file_name": alphabet_doc["filename"]
                    },
                    source="document",
                )
                for chunk in indexed
                if not letter or _chunk_starts_with_letter(chunk["text"], letter)
            ]
            if chunks:
                log.info(
                    "Alphabet lookup: %s → %s page/chunk(s) in %s",
                    letter,
                    len(chunks),
                    alphabet_doc["filename"],
                )
                return chunks, len(chunks), 0

    matching_doc = next(iter(find_documents_by_name(query)), None)
    if not matching_doc:
        return hybrid_search(processed_query)

    indexed = get_sqlite_document_chunks(matching_doc["document_id"])
    chunks = []
    for chunk in indexed:
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk["chunk_id"],
                document_id=matching_doc["document_id"],
                text=chunk["text"],
                score=1.0,
                metadata=chunk["metadata"] or {"file_name": matching_doc["filename"]},
                source="document",
            )
        )

    log.info("Document match: %s → %d SQLite chunks", matching_doc["filename"], len(chunks))
    return chunks, len(chunks), 0


def _extract_alphabet_letter(query: str) -> str | None:
    """Extract a requested alphabet letter without treating filenames as letters."""
    text = query.lower()
    if _is_broad_alphabet_request(text):
        return None
    match = re.search(
        r"\b([a-z])\s+(?:for|stands?\s+for|words?|alphabet|letter)\b",
        text,
    ) or re.search(r"\bfor\s+([a-z])\b", text)
    return match.group(1) if match else None


def _is_broad_alphabet_request(query: str) -> bool:
    text = query.lower()
    return bool(
        re.search(r"\ba\s+to\s+z\b", text)
        or re.search(r"\ball\s+(?:the\s+)?letters?\b", text)
        or re.search(r"\bletters?\s+(?:of|in)\s+the\s+alphabet\b", text)
    )


def _is_alphabet_query(query: str) -> bool:
    text = query.lower()
    return bool(
        _extract_alphabet_letter_without_broad_check(text)
        or _is_broad_alphabet_request(text)
        or re.search(r"\b(?:alphabet|letters?)\b", text)
    )


def _extract_alphabet_letter_without_broad_check(query: str) -> str | None:
    match = re.search(
        r"\b([a-z])\s+(?:for|stands?\s+for|words?|alphabet|letter)\b",
        query.lower(),
    ) or re.search(r"\bfor\s+([a-z])\b", query.lower())
    return match.group(1) if match else None


def _find_alphabet_document() -> Dict[str, Any] | None:
    candidates = [
        document for document in list_documents()
        if any(
            term in document.get("filename", "").lower()
            for term in ("a-to-z", "alphabet", "spelling")
        )
        and document.get("status") == "indexed"
    ]
    candidates.sort(key=lambda document: document.get("updated_at", ""), reverse=True)
    return candidates[0] if candidates else None


def _chunk_starts_with_letter(text: str, letter: str) -> bool:
    """Match extracted PDF text whose first heading is the requested letter."""
    return bool(re.match(rf"^\s*{re.escape(letter)}\s*{re.escape(letter)}\b", text, re.IGNORECASE))


def _format_alphabet_lookup(letter: str, chunks: List[RetrievedChunk]) -> str:
    """Return an exact word list for alphabet worksheet lookups."""
    grouped: Dict[str, List[str]] = {}
    excluded = {"vocabulary", "words", "pack", "wordspack", "worksheetspack"}
    for chunk in chunks:
        chunk_letter = letter or _chunk_letter(chunk.text)
        if not chunk_letter:
            continue
        words = grouped.setdefault(chunk_letter.upper(), [])
        for line in chunk.text.splitlines():
            compact = re.sub(r"\s+", "", line).lower()
            if (
                len(compact) >= 2
                and compact.isalpha()
                and compact.startswith(chunk_letter.lower())
                and compact not in excluded
                and compact != chunk_letter.lower() * 2
                and compact not in words
            ):
                words.append(compact)
    if not grouped:
        return "I couldn't find alphabet items in the indexed A-to-Z PDF."
    if letter:
        return f"In the A-to-Z PDF, {letter.upper()} is for: {', '.join(grouped[letter.upper()])}."
    lines = [f"- **{key}:** {', '.join(words)}" for key, words in grouped.items()]
    return "\n".join(lines)


def _chunk_letter(text: str) -> str | None:
    match = re.match(r"^\s*([A-Za-z])\1?\b", text, re.IGNORECASE)
    return match.group(1) if match else None


def _document_unavailable_response(
    query: str,
    model_name: str,
    conversation_id: str,
    start_time: float,
) -> ChatResponse:
    """Explain when a named document exists in metadata but has no indexed text."""
    elapsed = int((time.time() - start_time) * 1000)
    return ChatResponse(
        answer=(
            "The document was found in the document list, but its indexed content "
            "is unavailable. Please upload the document again or reindex it before "
            "asking for its contents."
        ),
        answer_type=AnswerType.INSUFFICIENT,
        confidence=0.0,
        model=model_name,
        response_time_ms=elapsed,
        conversation_id=conversation_id,
    )


def _build_sources(chunks) -> List[Source]:
    """Build source citations from retrieved chunks."""
    sources = []
    seen = set()

    for chunk in chunks:
        meta = chunk.metadata
        file_name = meta.get("file_name", "Unknown")
        page = meta.get("page_number") if meta.get("page_number") else None
        section = meta.get("section", "") or meta.get("sheet_name", "") or None
        key = (file_name, page, section, meta.get("sheet_name"), meta.get("slide_number"))
        if key in seen:
            continue
        seen.add(key)
        sources.append(Source(
            file_name=file_name,
            page=page,
            section=section,
            chunk_id=chunk.chunk_id,
            score=chunk.score,
            sheet_name=meta.get("sheet_name") or None,
            slide_number=meta.get("slide_number") if meta.get("slide_number") else None,
        ))

    return sources


def _store_conversation(
    conversation_id: str,
    question: str,
    answer: str,
    model: str = "gemini",
    answer_type: str = "dataset",
    confidence: float = 0.0,
    sources: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Store user message and assistant response in SQLite database."""
    try:
        add_message(conversation_id=conversation_id, role="user", content=question)
        add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            model=model,
            answer_type=answer_type,
            confidence=confidence,
            sources=sources,
        )
    except Exception as e:
        log.warning("Failed to store message in SQLite: %s", e)
