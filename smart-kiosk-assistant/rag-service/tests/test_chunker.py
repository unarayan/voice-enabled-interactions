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
