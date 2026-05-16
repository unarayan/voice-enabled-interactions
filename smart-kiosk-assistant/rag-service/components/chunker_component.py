"""Document chunking strategies for the RAG service.

Two strategies are supported, configurable via ``chunking.strategy``:

* ``semantic`` (default) — embedding-based. Splits text into sentences,
  embeds each one with the configured embedding model, and starts a new
  chunk whenever consecutive sentences fall below a similarity threshold
  or the running chunk hits ``max_chunk_chars``. Markdown headers are
  honored as hard boundaries when present, so structured documents stay
  organized without any markdown-specific tuning.

* ``fixed`` — character-based recursive split. Splits on paragraph,
  then line, then sentence, then word boundaries until each piece fits
  inside ``max_chunk_chars``. Deterministic, fast, and content-agnostic.

Both strategies share the same post-processing (whitespace cleanup,
overlap, tiny-chunk merging) so they're interchangeable from the
pipeline's point of view.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import numpy as np

from utils.config_loader import config


logger = logging.getLogger(__name__)

_HEADER_RE = re.compile(r"^#{1,6}\s+\S")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")


@dataclass(slots=True)
class ChunkRecord:
    text: str
    index: int


class SemanticChunker:
    """Strategy-dispatching chunker.

    The class name is kept for backwards compatibility with the pipeline,
    but it now selects between semantic (embedding) and fixed-size
    (recursive) splitting based on ``config.chunking.strategy``.
    """

    def __init__(self, embedding_component) -> None:
        cfg = config.chunking
        self.strategy = str(getattr(cfg, "strategy", "semantic")).lower()
        self.max_chunk_chars = int(getattr(cfg, "max_chunk_chars", 1200))
        self.min_chunk_chars = int(getattr(cfg, "min_chunk_chars", 200))
        self.overlap_chars = int(getattr(cfg, "overlap_chars", 150))
        self.similarity_threshold = float(getattr(cfg, "semantic_similarity_threshold", 0.72))
        self.embedding_component = embedding_component

        if self.strategy not in {"semantic", "fixed"}:
            logger.warning("Unknown chunking strategy %r; falling back to 'semantic'", self.strategy)
            self.strategy = "semantic"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def chunk_text(self, text: str) -> list[ChunkRecord]:
        normalized = self._normalize(text)
        if not normalized:
            return []

        t0 = time.monotonic()
        logger.info(
            "[chunker] strategy=%s input_chars=%d max=%d min=%d overlap=%d",
            self.strategy, len(normalized), self.max_chunk_chars,
            self.min_chunk_chars, self.overlap_chars,
        )

        if self.strategy == "fixed":
            chunks = self._fixed_size(normalized)
        else:
            chunks = self._semantic(normalized)

        chunks = self._merge_tiny(chunks)
        chunks = self._apply_overlap(chunks)

        elapsed = time.monotonic() - t0
        logger.info("[chunker] done chunks=%d elapsed=%.1fs", len(chunks), elapsed)
        return [ChunkRecord(text=c, index=i) for i, c in enumerate(chunks)]

    # ------------------------------------------------------------------
    # Strategy 1: semantic (embedding-based)
    # ------------------------------------------------------------------
    def _semantic(self, text: str) -> list[str]:
        """Split on markdown headers, then on embedding-similarity drops.

        Headers (lines starting with ``#``) act as hard boundaries so a
        new section always starts a new chunk. Within each section we
        embed sentences and cut whenever cosine similarity between
        neighbours falls below ``similarity_threshold`` or the running
        chunk exceeds ``max_chunk_chars``.
        """
        sections = self._split_by_headers(text)
        chunks: list[str] = []

        for section in sections:
            if len(section) <= self.max_chunk_chars:
                chunks.append(section)
                continue

            sentences = self._split_sentences(section)
            if len(sentences) <= 1:
                # No usable sentence boundaries — fall back to fixed split.
                chunks.extend(self._fixed_size(section))
                continue

            vectors = np.asarray(
                self.embedding_component.embed_documents(sentences),
                dtype=np.float32,
            )
            heading = section.split("\n", 1)[0] if section.lstrip().startswith("#") else ""

            current: list[str] = [sentences[0]]
            for i in range(1, len(sentences)):
                similarity = self._cosine(vectors[i - 1], vectors[i])
                projected = " ".join(current + [sentences[i]])
                if len(projected) > self.max_chunk_chars or similarity < self.similarity_threshold:
                    chunks.append(self._attach_heading(heading, " ".join(current)))
                    current = [sentences[i]]
                else:
                    current.append(sentences[i])
            if current:
                chunks.append(self._attach_heading(heading, " ".join(current)))

        return [c.strip() for c in chunks if c.strip()]

    # ------------------------------------------------------------------
    # Strategy 2: fixed-size recursive split
    # ------------------------------------------------------------------
    _SEPARATORS = ("\n\n", "\n", ". ", " ", "")

    def _fixed_size(self, text: str) -> list[str]:
        """Recursive char-based split.

        Tries paragraph → line → sentence → word → character boundaries
        in order until each chunk fits in ``max_chunk_chars``.
        """
        return self._recursive_split(text, self._SEPARATORS, self.max_chunk_chars)

    def _recursive_split(self, text: str, separators: tuple[str, ...], limit: int) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= limit:
            return [text]

        sep, *rest = separators
        rest_tuple = tuple(rest) if rest else ("",)

        if sep == "":
            # Last resort: hard slice on character boundary.
            return [text[i : i + limit] for i in range(0, len(text), limit)]

        parts = text.split(sep)
        chunks: list[str] = []
        buf = ""
        for part in parts:
            piece = part if not buf else f"{buf}{sep}{part}"
            if len(piece) <= limit:
                buf = piece
                continue
            if buf:
                chunks.append(buf)
            if len(part) <= limit:
                buf = part
            else:
                chunks.extend(self._recursive_split(part, rest_tuple, limit))
                buf = ""
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _split_by_headers(text: str) -> list[str]:
        sections: list[list[str]] = [[]]
        for line in text.split("\n"):
            if _HEADER_RE.match(line) and any(l.strip() for l in sections[-1]):
                sections.append([line])
            else:
                sections[-1].append(line)
        return [s for s in ("\n".join(lines).strip() for lines in sections) if s]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        flat = text.replace("\n", " ")
        return [p.strip() for p in _SENTENCE_SPLIT_RE.split(flat) if p.strip()]

    @staticmethod
    def _attach_heading(heading: str, body: str) -> str:
        body = body.strip()
        if heading and not body.startswith(heading):
            return f"{heading}\n{body}"
        return body

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        an, bn = np.linalg.norm(a), np.linalg.norm(b)
        if an == 0.0 or bn == 0.0:
            return 0.0
        return float(np.dot(a, b) / (an * bn))

    @staticmethod
    def _normalize(text: str) -> str:
        lines = [_WHITESPACE_RE.sub(" ", line).rstrip() for line in text.split("\n")]
        return _BLANKLINES_RE.sub("\n\n", "\n".join(lines)).strip()

    def _merge_tiny(self, chunks: list[str]) -> list[str]:
        """Merge sub-min_chunk_chars chunks into the previous one,
        unless the tiny chunk starts a new heading (in which case we keep
        the structural boundary)."""
        merged: list[str] = []
        for chunk in chunks:
            if (
                merged
                and len(chunk) < self.min_chunk_chars
                and not chunk.lstrip().startswith("#")
            ):
                merged[-1] = f"{merged[-1]}\n{chunk}".strip()
            else:
                merged.append(chunk)
        return merged

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if self.overlap_chars <= 0 or len(chunks) < 2:
            return chunks
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-self.overlap_chars :]
            # Snap to a clean line boundary so overlap doesn't start mid-word.
            nl = tail.find("\n")
            if nl != -1:
                tail = tail[nl + 1 :]
            tail = tail.strip()
            out.append(f"{tail}\n{chunks[i]}".strip() if tail else chunks[i])
        return out
