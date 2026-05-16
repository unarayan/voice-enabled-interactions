from components.chunker_component import ChunkRecord, SemanticChunker


class _FakeEmbeddings:
    def embed_documents(self, texts):
        # Vectors that move steadily so cosine similarity is high within
        # short windows but drops over longer distance.
        return [[1.0, float(i + 1)] for i, _ in enumerate(texts)]

    def embed_query(self, text):
        return [1.0, 0.0]


def _set(monkeypatch, **overrides):
    for key, value in overrides.items():
        monkeypatch.setattr(f"components.chunker_component.config.chunking.{key}", value)


def test_fixed_strategy_splits_long_text(monkeypatch):
    _set(monkeypatch, strategy="fixed", max_chunk_chars=60, min_chunk_chars=10, overlap_chars=0)
    chunker = SemanticChunker(_FakeEmbeddings())
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."

    chunks = chunker.chunk_text(text)

    assert chunks, "expected at least one chunk"
    assert all(isinstance(c, ChunkRecord) for c in chunks)
    assert all(len(c.text) <= 80 for c in chunks)  # max + a little headroom for overlap


def test_semantic_strategy_returns_chunks(monkeypatch):
    _set(
        monkeypatch,
        strategy="semantic",
        max_chunk_chars=120,
        min_chunk_chars=10,
        overlap_chars=0,
        semantic_similarity_threshold=0.99,  # high so we force splits
    )
    chunker = SemanticChunker(_FakeEmbeddings())
    text = (
        "Apples are sweet. Bananas are yellow. Carrots are crunchy. "
        "Dragonfruit is exotic. Eggplants are purple."
    )

    chunks = chunker.chunk_text(text)

    assert len(chunks) >= 1
    assert all(c.text for c in chunks)


def test_markdown_headers_become_section_boundaries(monkeypatch):
    _set(monkeypatch, strategy="semantic", max_chunk_chars=2000, min_chunk_chars=10, overlap_chars=0)
    chunker = SemanticChunker(_FakeEmbeddings())
    text = "# Section A\nApples are sweet.\n\n# Section B\nBananas are yellow."

    chunks = chunker.chunk_text(text)

    assert len(chunks) == 2
    assert "Section A" in chunks[0].text
    assert "Section B" in chunks[1].text


def test_llm_strategy_uses_line_markers(monkeypatch):
    """LLM strategy calls the llm_generate_fn and uses its marker response."""
    _set(monkeypatch, strategy="llm", max_chunk_chars=500, min_chunk_chars=10, overlap_chars=0)
    calls: list[str] = []

    def mock_llm(prompt: str, max_tokens: int) -> str:
        calls.append(prompt)
        # Signal: start new chunk at line 3.
        return "[1, 3]"

    chunker = SemanticChunker(_FakeEmbeddings(), llm_generate_fn=mock_llm)
    assert chunker.strategy == "llm"

    text = "Line one.\nLine two.\nLine three.\nLine four.\nLine five."
    chunks = chunker.chunk_text(text)

    assert len(calls) >= 1, "LLM generate function was never called"
    assert len(chunks) == 2
    assert all(isinstance(c, ChunkRecord) for c in chunks)
    assert "Line one" in chunks[0].text
    assert "Line three" in chunks[1].text


def test_llm_strategy_falls_back_when_no_llm_fn(monkeypatch):
    """SemanticChunker must demote strategy to 'semantic' when llm_generate_fn is None."""
    _set(monkeypatch, strategy="llm", max_chunk_chars=200, min_chunk_chars=10, overlap_chars=0)
    chunker = SemanticChunker(_FakeEmbeddings(), llm_generate_fn=None)

    assert chunker.strategy == "semantic"


def test_llm_strategy_falls_back_on_bad_response(monkeypatch):
    """When the LLM returns garbage the chunker falls back to semantic silently."""
    _set(
        monkeypatch,
        strategy="llm",
        max_chunk_chars=500,
        min_chunk_chars=10,
        overlap_chars=0,
        semantic_similarity_threshold=0.99,
    )

    def mock_llm(prompt: str, max_tokens: int) -> str:
        return "not valid json at all !!! @@"

    chunker = SemanticChunker(_FakeEmbeddings(), llm_generate_fn=mock_llm)
    text = "Apples are tasty. Oranges are citrusy. Grapes are sweet."
    chunks = chunker.chunk_text(text)

    # Should have fallen back to semantic and produced at least one chunk.
    assert len(chunks) >= 1
    assert all(c.text for c in chunks)


def test_empty_text_returns_no_chunks(monkeypatch):
    _set(monkeypatch, strategy="semantic", max_chunk_chars=200, min_chunk_chars=10, overlap_chars=0)
    chunker = SemanticChunker(_FakeEmbeddings())

    assert chunker.chunk_text("") == []
    assert chunker.chunk_text("   \n\n   ") == []


def test_chunk_indices_are_sequential(monkeypatch):
    _set(monkeypatch, strategy="fixed", max_chunk_chars=40, min_chunk_chars=5, overlap_chars=0)
    chunker = SemanticChunker(_FakeEmbeddings())
    text = "Word " * 40  # long enough to produce multiple chunks

    chunks = chunker.chunk_text(text)

    assert len(chunks) >= 2
    assert [c.index for c in chunks] == list(range(len(chunks)))

