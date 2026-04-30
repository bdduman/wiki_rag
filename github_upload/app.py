from __future__ import annotations

import sys
from time import perf_counter
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wikirag.config import load_settings
from wikirag.db import connect, count_pages
from wikirag.ollama_client import OllamaClient, OllamaError
from wikirag.rag import (
    RagAnswer,
    answer_query,
    build_citations,
    build_prompt,
    clear_response_cache,
    get_cached_response,
    retrieve_contexts,
    store_response,
)
from wikirag.vector_store import VectorStore


st.set_page_config(page_title="Local Wikipedia RAG", layout="wide")

settings = load_settings()


@st.cache_resource
def get_services() -> tuple[OllamaClient, VectorStore]:
    return (
        OllamaClient(settings.ollama_base_url, settings.llm_model, settings.embed_model),
        VectorStore(settings.chroma_path, settings.collection_name),
    )


def initialize_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []


initialize_state()
ollama, vector_store = get_services()

with st.sidebar:
    st.title("Local Wikipedia RAG")
    st.caption("Runs with local Wikipedia data, Chroma, SQLite, and Ollama.")
    show_context = st.toggle("Show retrieved context", value=True)
    enable_streaming = st.toggle("Streaming responses", value=True)
    enable_cache = st.toggle("Cache responses", value=True)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()
    if st.button("Clear response cache", use_container_width=True):
        clear_response_cache()
        st.toast("Response cache cleared.")

    st.divider()
    st.write("Default model")
    st.code(settings.llm_model)
    st.write("Embedding")
    st.code(settings.embed_model)

    try:
        with connect(settings.sqlite_path) as connection:
            page_count = count_pages(connection)
        st.metric("Indexed pages", page_count)
    except Exception:
        st.warning("No local index found. Run `python scripts/ingest.py` first.")

st.title("Local Wikipedia RAG Assistant")
st.write("Ask about the indexed people and places. Answers are grounded in retrieved Wikipedia chunks.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if show_context and message.get("contexts"):
            with st.expander("Retrieved context"):
                for context in message["contexts"]:
                    metadata = context.get("metadata", {})
                    title = metadata.get("title", "Unknown source")
                    distance = context.get("distance", 0.0)
                    st.caption(f"{title} | distance: {distance:.3f}")
                    st.write(context.get("text", ""))
        if message.get("citations"):
            with st.expander("Citations"):
                for citation in message["citations"]:
                    st.write(citation)
        if message.get("metrics"):
            st.caption(message["metrics"])

query = st.chat_input("Ask a question about a famous person or place")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating an answer..."):
            try:
                if enable_streaming:
                    cached_result = (
                        get_cached_response(
                            query,
                            settings.llm_model,
                            settings.top_k,
                            settings.max_distance,
                        )
                        if enable_cache
                        else None
                    )
                    if cached_result is not None:
                        answer = cached_result.answer
                        query_type = cached_result.query_type
                        result_contexts = cached_result.contexts
                        citations = cached_result.citations
                        retrieval_ms = cached_result.retrieval_ms
                        generation_ms = cached_result.generation_ms
                        cached = True
                        st.markdown(answer)
                    else:
                        retrieval_started = perf_counter()
                        query_type, contexts = retrieve_contexts(
                            query=query,
                            embedder=ollama,
                            index=vector_store,
                            top_k=settings.top_k,
                        )
                        retrieval_ms = (perf_counter() - retrieval_started) * 1000
                        useful_contexts = [
                            context
                            for context in contexts
                            if float(context.get("distance", 999.0)) <= settings.max_distance
                        ]
                        citations = build_citations(useful_contexts or contexts)
                        if not useful_contexts:
                            answer = "I don't know."
                            generation_ms = 0.0
                        else:
                            prompt = build_prompt(query, useful_contexts)
                            generation_started = perf_counter()
                            chunks = []
                            placeholder = st.empty()
                            for token in ollama.generate_stream(prompt, model=settings.llm_model):
                                chunks.append(token)
                                placeholder.markdown("".join(chunks))
                            answer = "".join(chunks).strip()
                            generation_ms = (perf_counter() - generation_started) * 1000
                        result_contexts = useful_contexts or contexts
                        cached = False
                        if enable_cache:
                            store_response(
                                query,
                                settings.llm_model,
                                settings.top_k,
                                settings.max_distance,
                                RagAnswer(
                                    answer=answer,
                                    query_type=query_type,
                                    contexts=result_contexts,
                                    citations=citations,
                                    retrieval_ms=retrieval_ms,
                                    generation_ms=generation_ms,
                                ),
                            )
                else:
                    result = answer_query(
                        query=query,
                        embedder=ollama,
                        generator=ollama,
                        index=vector_store,
                        top_k=settings.top_k,
                        max_distance=settings.max_distance,
                        model=settings.llm_model,
                        use_cache=enable_cache,
                    )
                    answer = result.answer
                    query_type = result.query_type
                    result_contexts = result.contexts
                    citations = result.citations
                    retrieval_ms = result.retrieval_ms
                    generation_ms = result.generation_ms
                    cached = result.cached
                    st.markdown(answer)

                metrics = (
                    f"type: {query_type} | retrieval: {retrieval_ms:.0f} ms | "
                    f"generation: {generation_ms:.0f} ms | cache: {'hit' if cached else 'miss'}"
                )
                st.caption(metrics)

                if show_context:
                    with st.expander(f"Retrieved context ({query_type})"):
                        for index, context in enumerate(result_contexts, start=1):
                            metadata = context.get("metadata", {})
                            title = metadata.get("title", "Unknown source")
                            source_url = metadata.get("source_url", "")
                            distance = context.get("distance", 0.0)
                            st.caption(f"[{index}] {title} | distance: {distance:.3f} | {source_url}")
                            st.write(context.get("text", ""))
                with st.expander("Citations"):
                    for citation in citations:
                        st.write(citation)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "contexts": result_contexts,
                        "citations": citations,
                        "metrics": metrics,
                    }
                )
                st.session_state.history.append({"question": query, "answer": answer})
            except OllamaError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
