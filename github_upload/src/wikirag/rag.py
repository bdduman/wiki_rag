from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from .classifier import classify_query, mentioned_entities


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


class Generator(Protocol):
    def generate(self, prompt: str, model: str | None = None) -> str:
        ...


class SearchIndex(Protocol):
    def query(
        self,
        embedding: list[float],
        top_k: int,
        entity_type: str | None = None,
        title: str | None = None,
    ) -> list[dict]:
        ...


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    query_type: str
    contexts: list[dict]
    citations: list[str]
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    cached: bool = False


_RESPONSE_CACHE: dict[tuple[str, str, int, float], RagAnswer] = {}


def _cache_key(query: str, model: str | None, top_k: int, max_distance: float) -> tuple[str, str, int, float]:
    return (query.strip().lower(), model or "", top_k, max_distance)


def clear_response_cache() -> None:
    _RESPONSE_CACHE.clear()


def get_cached_response(
    query: str,
    model: str | None,
    top_k: int,
    max_distance: float,
) -> RagAnswer | None:
    cached = _RESPONSE_CACHE.get(_cache_key(query, model, top_k, max_distance))
    if cached is None:
        return None
    return RagAnswer(
        answer=cached.answer,
        query_type=cached.query_type,
        contexts=cached.contexts,
        citations=cached.citations,
        retrieval_ms=cached.retrieval_ms,
        generation_ms=0.0,
        cached=True,
    )


def store_response(
    query: str,
    model: str | None,
    top_k: int,
    max_distance: float,
    answer: RagAnswer,
) -> None:
    _RESPONSE_CACHE[_cache_key(query, model, top_k, max_distance)] = answer


def build_prompt(query: str, contexts: list[dict]) -> str:
    context_lines = []
    for index, context in enumerate(contexts, start=1):
        metadata = context.get("metadata", {})
        title = metadata.get("title", "Unknown source")
        source_url = metadata.get("source_url", "")
        text = context.get("text", "")
        context_lines.append(f"[{index}] {title} ({source_url})\n{text}")

    joined_context = "\n\n".join(context_lines)
    return f"""You are a local Wikipedia RAG assistant.
Answer the question using only the retrieved context below.
If the context does not contain enough facts for a normal question, say "I don't know."
Keep the answer concise. Cite relevant context with bracket numbers like [1].
For comparison questions, if there is context for each named entity, synthesize a comparison from those separate chunks; do not require a source chunk that already compares them.

Retrieved context:
{joined_context}

Question: {query}
Answer:"""


def _deduplicate_contexts(contexts: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for context in sorted(contexts, key=lambda item: item.get("distance", 999.0)):
        chunk_id = context.get("id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        unique.append(context)
    return unique


def _keyword_score(query: str, text: str) -> int:
    query_terms = {
        term.strip(".,?!:;()[]{}").lower()
        for term in query.split()
        if len(term.strip(".,?!:;()[]{}")) > 3
    }
    text_lower = text.lower()
    return sum(1 for term in query_terms if term in text_lower)


def _rerank_contexts(query: str, contexts: list[dict]) -> list[dict]:
    def score(context: dict) -> float:
        distance = float(context.get("distance", 999.0))
        text = context.get("text", "")
        metadata = context.get("metadata", {})
        title_bonus = 0.25 if metadata.get("title", "").lower() in query.lower() else 0.0
        intro_bonus = max(0.0, 0.18 - float(metadata.get("chunk_index", 99)) * 0.03)
        keyword_bonus = min(0.18, _keyword_score(query, text) * 0.03)
        return distance - title_bonus - intro_bonus - keyword_bonus

    return sorted(contexts, key=score)


def build_citations(contexts: list[dict]) -> list[str]:
    citations = []
    seen = set()
    for index, context in enumerate(contexts, start=1):
        metadata = context.get("metadata", {})
        title = metadata.get("title", "Unknown source")
        source_url = metadata.get("source_url", "")
        label = f"[{index}] {title} - {source_url}".strip()
        if label not in seen:
            citations.append(label)
            seen.add(label)
    return citations


def retrieve_contexts(
    query: str,
    embedder: Embedder,
    index: SearchIndex,
    top_k: int,
) -> tuple[str, list[dict]]:
    query_type = classify_query(query)
    query_embedding = embedder.embed(query)
    entities = mentioned_entities(query)

    if entities:
        per_entity_k = max(8, top_k)
        grouped_contexts = []
        for title, entity_type in entities:
            intro_contexts = []
            if hasattr(index, "get_by_title"):
                intro_contexts = index.get_by_title(title, entity_type=entity_type, limit=2)
            entity_contexts = _rerank_contexts(
                query,
                intro_contexts
                + index.query(
                    query_embedding,
                    top_k=per_entity_k,
                    entity_type=entity_type,
                    title=title,
                ),
            )
            grouped_contexts.append(entity_contexts)

        per_entity_take = max(1, top_k // len(grouped_contexts))
        contexts = []
        for entity_contexts in grouped_contexts:
            contexts.extend(entity_contexts[:per_entity_take])
        remaining = []
        for entity_contexts in grouped_contexts:
            remaining.extend(entity_contexts[per_entity_take:])
        contexts.extend(_rerank_contexts(query, remaining))
        return query_type, _deduplicate_contexts(contexts)[:top_k]

    if query_type == "person":
        contexts = index.query(query_embedding, top_k=top_k, entity_type="person")
    elif query_type == "place":
        contexts = index.query(query_embedding, top_k=top_k, entity_type="place")
    elif query_type == "both":
        contexts = index.query(query_embedding, top_k=top_k, entity_type="person")
        contexts += index.query(query_embedding, top_k=top_k, entity_type="place")
    else:
        contexts = index.query(query_embedding, top_k=top_k, entity_type=None)

    return query_type, _rerank_contexts(query, _deduplicate_contexts(contexts))[:top_k]


def answer_query(
    query: str,
    embedder: Embedder,
    generator: Generator,
    index: SearchIndex,
    top_k: int = 5,
    max_distance: float = 0.65,
    model: str | None = None,
    use_cache: bool = True,
) -> RagAnswer:
    if use_cache:
        cached = get_cached_response(query, model, top_k, max_distance)
        if cached is not None:
            return cached

    retrieval_started = perf_counter()
    query_type, contexts = retrieve_contexts(query, embedder, index, top_k)
    retrieval_ms = (perf_counter() - retrieval_started) * 1000
    useful_contexts = [
        context for context in contexts if float(context.get("distance", 999.0)) <= max_distance
    ]

    if not useful_contexts:
        result = RagAnswer(
            answer="I don't know.",
            query_type=query_type,
            contexts=contexts,
            citations=build_citations(contexts),
            retrieval_ms=retrieval_ms,
        )
        if use_cache:
            store_response(query, model, top_k, max_distance, result)
        return result

    prompt = build_prompt(query, useful_contexts)
    generation_started = perf_counter()
    answer = generator.generate(prompt, model=model)
    generation_ms = (perf_counter() - generation_started) * 1000
    result = RagAnswer(
        answer=answer,
        query_type=query_type,
        contexts=useful_contexts,
        citations=build_citations(useful_contexts),
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
    )
    if use_cache:
        store_response(query, model, top_k, max_distance, result)
    return result
