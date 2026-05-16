"""Document chunking strategies for the RAG service.

Three strategies are supported, configurable via ``chunking.strategy``:

* ``semantic`` (default) — embedding-based. Splits text into sentences,
  embeds each one with the configured embedding model, and starts a new
  chunk whenever consecutive sentences fall below a similarity threshold
  or the running chunk hits ``max_chunk_chars``. Markdown headers are
  honored as hard boundaries when present.

* ``fixed`` — character-based recursive split. Splits on paragraph,
  then line, then sentence, then word boundaries until each piece fits
  inside ``max_chunk_chars``. Deterministic, fast, and content-agnostic.

* ``llm`` — LLM-assisted. Passes sliding passages through the LLM to
  identify semantic break-points in free-form, unstructured text.
  Significantly slower than the other two strategies; requires a working
  LLM pipeline and a GPU driver that handles the token budget set in
  ``chunking.llm_max_tokens`` without stalling (driver 26.18+ recommended).
  Only use this when ``semantic`` genuinely produces poor chunks.

All three strategies share the same post-processing (whitespace cleanup,
overlap, tiny-chunk merging).
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

    def __init__(self, embedding_component, llm_generate_fn=None) -> None:
        """
        Parameters
        ----------
        embedding_component:
            Object with ``embed_documents(texts) -> list[list[float]]``.
        llm_generate_fn:
            Optional callable ``(prompt: str, max_tokens: int) -> str``.
            Required only when ``chunking.strategy = llm``; ignored otherwise.
        """
        cfg = config.chunking
        self.strategy = str(getattr(cfg, "strategy", "llm")).lower()
        self.max_chunk_chars = int(getattr(cfg, "max_chunk_chars", 1200))
        self.min_chunk_chars = int(getattr(cfg, "min_chunk_chars", 200))
        self.overlap_chars = int(getattr(cfg, "overlap_chars", 150))
        self.similarity_threshold = float(getattr(cfg, "semantic_similarity_threshold", 0.72))
        self.embedding_component = embedding_component
        self._llm_generate = llm_generate_fn

        # LLM strategy config
        llm_cfg = getattr(cfg, "llm", None)
        self._llm_passage_chars = int(getattr(llm_cfg, "passage_chars", 4000) if llm_cfg else 4000)
        self._llm_max_tokens = int(getattr(llm_cfg, "max_tokens", 512) if llm_cfg else 512)

        if self.strategy not in {"semantic", "fixed", "llm"}:
            logger.warning("Unknown chunking strategy %r; falling back to 'semantic'", self.strategy)
            self.strategy = "semantic"

        if self.strategy == "llm" and self._llm_generate is None:
            logger.warning(
                "chunking.strategy=llm but no LLM function provided; falling back to 'semantic'"
            )
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
        elif self.strategy == "llm":
            chunks = self._llm(normalized)
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
    # Strategy 3: LLM-assisted split (line-marker approach)
    # ------------------------------------------------------------------
    _LLM_MARKER_PROMPT = (
        "You are a document segmenter. "
        "The text below has been split into numbered lines. "
        "Output ONLY a compact JSON array of 1-based integer line numbers where a new semantic chunk "
        "should START. Always include line 1. Example output: [1, 5, 12]. "
        "Do not include any commentary, keys, or markdown fences.\n\n"
        "Numbered lines:\n{numbered}\n\n"
        "JSON array of chunk start line numbers:"
    )

    def _llm(self, text: str) -> list[str]:
        """Send overlapping passages to the LLM and ask it to identify chunk
        boundaries as 1-based line-number markers. Falls back to ``_semantic``
        on any error or malformed response.

        Requesting line-number markers instead of regenerated text avoids
        hallucination, keeps the output token budget tiny (a short integer
        array regardless of passage length), and guarantees the original text
        is preserved verbatim. The passage size is controlled by
        ``chunking.llm.passage_chars`` (default 4000) and the generation cap
        by ``chunking.llm.max_tokens`` (default 512).
        """
        passages = self._sliding_passages(text, self._llm_passage_chars)
        chunks: list[str] = []
        seen: set[str] = set()

        for passage in passages:
            lines, numbered = self._number_lines(passage)
            prompt = self._LLM_MARKER_PROMPT.format(numbered=numbered)
            try:
                raw = self._llm_generate(prompt, self._llm_max_tokens)
                markers = self._parse_line_markers(raw, len(lines))
                if not markers:
                    raise ValueError("no valid line markers in LLM response")
                for c in self._split_by_line_markers(lines, markers):
                    if c not in seen:
                        seen.add(c)
                        chunks.append(c)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[chunker] LLM split failed for passage, falling back to semantic: %s", exc)
                for c in self._semantic(passage):
                    if c not in seen:
                        seen.add(c)
                        chunks.append(c)

        return chunks or self._semantic(text)

    @staticmethod
    def _number_lines(text: str) -> tuple[list[str], str]:
        """Return ``(lines, numbered_text)`` with 1-based line-number prefixes."""
        lines = text.split("\n")
        numbered = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))
        return lines, numbered

    @staticmethod
    def _parse_line_markers(raw: str, n_lines: int) -> list[int]:
        """Parse a JSON array of 1-based start-line indices from LLM output.

        Returns an empty list if parsing fails or the array contains no valid
        indices — callers should treat that as a signal to fall back.
        """
        import json

        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        try:
            items = json.loads(raw[start:end])
            if not isinstance(items, list):
                return []
            return sorted(
                {int(x) for x in items if isinstance(x, (int, float)) and 1 <= int(x) <= n_lines}
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return []

    @staticmethod
    def _split_by_line_markers(lines: list[str], markers: list[int]) -> list[str]:
        """Split *lines* into chunks that start at each 1-based *marker* index.

        Line 1 is always treated as a start even if absent from *markers*.
        """
        starts = sorted({1} | set(markers))
        chunks: list[str] = []
        for i, start_ln in enumerate(starts):
            end_ln = starts[i + 1] if i + 1 < len(starts) else len(lines) + 1
            chunk = "\n".join(lines[start_ln - 1 : end_ln - 1]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    @staticmethod
    def _sliding_passages(text: str, passage_chars: int) -> list[str]:
        """Split text into overlapping passages that fit in ``passage_chars``."""
        if len(text) <= passage_chars:
            return [text]
        overlap = passage_chars // 8
        step = passage_chars - overlap
        passages: list[str] = []
        i = 0
        while i < len(text):
            passages.append(text[i : i + passage_chars])
            i += step
        return passages

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
