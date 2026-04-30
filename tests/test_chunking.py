from wikirag.chunking import chunk_text


def test_chunk_text_uses_overlap_for_large_documents():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, chunk_size=120, overlap=30)

    assert len(chunks) > 1
    assert chunks[0][-30:].strip() in chunks[1]
    assert all(len(chunk) <= 120 for chunk in chunks)


def test_chunk_text_rejects_invalid_overlap():
    try:
        chunk_text("hello", chunk_size=100, overlap=100)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

