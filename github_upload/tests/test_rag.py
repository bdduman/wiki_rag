from wikirag.rag import answer_query, retrieve_contexts


class FakeOllama:
    def embed(self, text):
        return [0.1, 0.2, 0.3]

    def generate(self, prompt, model=None):
        assert "Retrieved context" in prompt
        return "Grounded answer."


class FakeIndex:
    def __init__(self):
        self.calls = []

    def query(self, embedding, top_k, entity_type=None, title=None):
        self.calls.append((entity_type, title))
        return [
            {
                "id": f"{entity_type or 'all'}-{title or 'any'}-1",
                "text": "Albert Einstein was a theoretical physicist.",
                "metadata": {"title": title or "Albert Einstein", "type": entity_type or "person"},
                "distance": 0.2,
            }
        ]


def test_retrieval_filters_people_queries():
    index = FakeIndex()
    query_type, contexts = retrieve_contexts(
        "Who was Albert Einstein?",
        embedder=FakeOllama(),
        index=index,
        top_k=3,
    )

    assert query_type == "person"
    assert index.calls == [("person", "Albert Einstein")]
    assert contexts


def test_low_score_returns_i_do_not_know():
    class LowScoreIndex(FakeIndex):
        def query(self, embedding, top_k, entity_type=None, title=None):
            return [
                {
                    "id": "bad",
                    "text": "Unrelated context.",
                    "metadata": {"title": "Unknown"},
                    "distance": 0.99,
                }
            ]

    result = answer_query(
        "Who is the president of Mars?",
        embedder=FakeOllama(),
        generator=FakeOllama(),
        index=LowScoreIndex(),
        max_distance=0.65,
    )

    assert result.answer == "I don't know."


def test_answer_includes_citations_and_latency_fields():
    result = answer_query(
        "Who was Albert Einstein?",
        embedder=FakeOllama(),
        generator=FakeOllama(),
        index=FakeIndex(),
        use_cache=False,
    )

    assert result.answer == "Grounded answer."
    assert result.citations
    assert result.retrieval_ms >= 0
    assert result.generation_ms >= 0


def test_cached_answer_sets_cache_hit():
    index = FakeIndex()
    first = answer_query(
        "Who was Albert Einstein?",
        embedder=FakeOllama(),
        generator=FakeOllama(),
        index=index,
        use_cache=True,
    )
    second = answer_query(
        "Who was Albert Einstein?",
        embedder=FakeOllama(),
        generator=FakeOllama(),
        index=index,
        use_cache=True,
    )

    assert first.cached is False
    assert second.cached is True
