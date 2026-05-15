from __future__ import annotations

import datetime
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Callable

import numpy as np

from utils.config_loader import config


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChunkRecord:
    text: str
    index: int


class SemanticChunker:
    def __init__(
        self,
        embedding_component,
        llm_text_generator: Callable[[str, int | None, float | None], str],
        llm_tokenizer=None,
    ) -> None:
        chunk_cfg = config.chunking
        self.max_chunk_chars = int(getattr(chunk_cfg, "max_chunk_chars", 1200))
        self.min_chunk_chars = int(getattr(chunk_cfg, "min_chunk_chars", 180))
        self.overlap_chars = int(getattr(chunk_cfg, "overlap_chars", 120))
        self.semantic_similarity_threshold = float(getattr(chunk_cfg, "semantic_similarity_threshold", 0.72))
        self.llm_passage_chars = int(getattr(chunk_cfg, "llm_passage_chars", 6000))
        self.llm_passage_tokens = int(getattr(chunk_cfg, "llm_passage_tokens", 0))
        self.llm_passage_overlap_tokens = int(getattr(chunk_cfg, "llm_passage_overlap_tokens", 0))
        self.llm_text_generator = llm_text_generator
        self.embedding_component = embedding_component
        self.llm_tokenizer = llm_tokenizer
        _debug_dir = getattr(chunk_cfg, "save_chunks_debug", None)
        self.save_chunks_debug: str | None = str(_debug_dir) if _debug_dir else None

    def chunk_text(self, text: str) -> list[ChunkRecord]:
        normalized = self._normalize_text(text)
        if not normalized:
            return []

        t0 = time.monotonic()
        logger.info(
            "[CHUNKER] Starting chunking | strategy=semantic_llm | input_chars=%d",
            len(normalized),
        )

        chunks = self._semantic_llm_chunks(normalized)
        chunks = self._apply_overlap(chunks)

        if self.save_chunks_debug and chunks:
            self._save_debug_chunks(chunks)

        records = [ChunkRecord(text=chunk, index=index) for index, chunk in enumerate(chunks)]
        elapsed = time.monotonic() - t0
        logger.info("[CHUNKER] Done | total_chunks=%d | elapsed=%.1fs", len(records), elapsed)
        return records

    # Marker output is always a tiny JSON array — never OOM risk.
    _MARKER_MAX_TOKENS = 256

    def _semantic_llm_chunks(self, text: str) -> list[str]:
        # ── Phase 0: detect document domain & structure once per ingest ──
        profile = self._detect_document_profile(text)

        # ── Phase 1: coarse splitting — always on clean boundaries ──
        if self.llm_tokenizer is not None and self.llm_passage_tokens > 0:
            coarse_passages = self._split_by_tokens(
                text, self.llm_passage_tokens, self.llm_passage_overlap_tokens,
            )
        else:
            coarse_passages = self._split_by_size(text, self.llm_passage_chars)

        total = len(coarse_passages)
        logger.info("[CHUNKER] semantic_llm | passages=%d | %s", total, profile["description"])

        results: list[str] = []

        for p_idx, passage in enumerate(coarse_passages, start=1):
            passage_chars = len(passage)

            if passage_chars <= self.min_chunk_chars:
                logger.info(
                    "[CHUNKER] Passage %d/%d | chars=%d | too small, keeping as-is",
                    p_idx, total, passage_chars,
                )
                results.append(passage)
                continue

            lines, numbered = self._number_lines(passage)
            logger.info(
                "[CHUNKER] Passage %d/%d | chars=%d | lines=%d | requesting split markers",
                p_idx, total, passage_chars, len(lines),
            )
            logger.info("[CHUNKER] Passage %d/%d | preview: %r", p_idx, total, passage[:200])

            prompt = self._build_marker_prompt(numbered, profile, len(lines))
            t0 = time.monotonic()

            try:
                raw = self.llm_text_generator(prompt, self._MARKER_MAX_TOKENS, 0.0)
                elapsed = time.monotonic() - t0
                logger.info(
                    "[CHUNKER] Passage %d/%d | LLM markers (%.1fs, %d chars): %r",
                    p_idx, total, elapsed, len(raw), raw[:400],
                )

                markers = self._parse_line_markers(raw, len(lines))
                if markers:
                    chunks = self._split_by_line_markers(lines, markers)
                    logger.info(
                        "[CHUNKER] Passage %d/%d | marker split → %d chunks | starts=%s",
                        p_idx, total, len(chunks), markers,
                    )
                    for ci, c in enumerate(chunks, start=1):
                        logger.info(
                            "[CHUNKER] Passage %d/%d | chunk %d/%d | chars=%d | preview: %r",
                            p_idx, total, ci, len(chunks), len(c), c[:120],
                        )
                    results.extend(chunks)
                    continue

                logger.warning(
                    "[CHUNKER] Passage %d/%d | no valid markers parsed (%.1fs) → embedding fallback",
                    p_idx, total, elapsed,
                )
            except Exception as exc:  # noqa: BLE001
                elapsed = time.monotonic() - t0
                logger.warning(
                    "[CHUNKER] Passage %d/%d | LLM error after %.1fs (%s) → embedding fallback",
                    p_idx, total, elapsed, exc,
                )

            fb = self._semantic_embedding_chunks(passage)
            logger.info(
                "[CHUNKER] Passage %d/%d | embedding fallback → %d chunks",
                p_idx, total, len(fb),
            )
            results.extend(fb)

        return self._cleanup_chunks(results)

    def _semantic_embedding_chunks(self, text: str) -> list[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [sentences[0]]

        sentence_vectors = np.array(self.embedding_component.embed_documents(sentences), dtype=np.float32)
        chunks: list[str] = []
        current_sentences = [sentences[0]]

        for index in range(1, len(sentences)):
            prev_vector = sentence_vectors[index - 1]
            current_vector = sentence_vectors[index]
            similarity = self._cosine_similarity(prev_vector, current_vector)
            projected = " ".join(current_sentences + [sentences[index]])

            if len(projected) > self.max_chunk_chars or similarity < self.semantic_similarity_threshold:
                candidate = " ".join(current_sentences).strip()
                if candidate:
                    chunks.append(candidate)
                current_sentences = [sentences[index]]
                continue

            current_sentences.append(sentences[index])

        final_chunk = " ".join(current_sentences).strip()
        if final_chunk:
            chunks.append(final_chunk)
        return self._cleanup_chunks(chunks)

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Prepend the tail of the previous chunk as context.

        FIX: overlap is taken from the raw chunk text (preserving newlines)
        BEFORE any whitespace collapsing, so the prefix still carries
        structural signals (bullet markers, section headers) into the next chunk.
        We also ensure the overlap boundary lands on a full line, not mid-word.
        """
        if self.overlap_chars <= 0 or len(chunks) < 2:
            return chunks

        with_overlap: list[str] = [chunks[0]]
        for index in range(1, len(chunks)):
            tail = chunks[index - 1][-self.overlap_chars:]
            # Snap to start of a line so overlap doesn't begin mid-bullet/mid-word.
            newline_pos = tail.find("\n")
            if newline_pos != -1:
                tail = tail[newline_pos + 1:]
            tail = tail.strip()
            current = chunks[index]
            combined = f"{tail}\n{current}" if tail else current
            with_overlap.append(combined.strip())
        return with_overlap

    def _split_by_size(self, text: str, max_chars: int) -> list[str]:
        """Split into passages that always end on a clean line boundary.

        FIX (original bug): the old version split only on \\n\\n (double newlines).
        Retail/structured KBs use single \\n between list items, so passages were
        cut mid-bullet producing fragments like '** — per piece | Tags: none'.

        New strategy: accumulate lines greedily and cut only when the next line
        would exceed max_chars, always ending on a full line.
        """
        lines = text.split("\n")
        passages: list[str] = []
        current_lines: list[str] = []
        current_chars = 0

        for line in lines:
            # +1 for the newline we'll re-add
            line_len = len(line) + 1
            if current_lines and current_chars + line_len > max_chars:
                passages.append("\n".join(current_lines))
                # Carry the last heading/section line into the next passage
                # so context isn't lost when a section header sits at the cut point.
                carry = self._find_carry_line(current_lines)
                current_lines = [carry, line] if carry else [line]
                current_chars = sum(len(l) + 1 for l in current_lines)
            else:
                current_lines.append(line)
                current_chars += line_len

        if current_lines:
            passages.append("\n".join(current_lines))

        return [p for p in passages if p.strip()]

    @staticmethod
    def _find_carry_line(lines: list[str]) -> str:
        """Return the last heading line from a passage to carry into the next,
        preserving section context at passage boundaries."""
        for line in reversed(lines):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("##") or stripped.startswith("###"):
                return line
        return ""

    def _split_by_tokens(self, text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
        tokenizer = self.llm_tokenizer
        if tokenizer is None:
            return self._split_by_size(text, self.llm_passage_chars)

        # FIX: after decoding each token window, snap the boundary to the
        # nearest preceding newline so passages never start mid-line.
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            return []

        chunks: list[str] = []
        start = 0
        step = max(max_tokens - max(overlap_tokens, 0), 1)

        while start < len(token_ids):
            end = min(start + max_tokens, len(token_ids))
            window_ids = token_ids[start:end]
            chunk_text = tokenizer.decode(window_ids, skip_special_tokens=True)

            # Snap to last newline if we're not at end of document
            if end < len(token_ids):
                last_nl = chunk_text.rfind("\n")
                if last_nl > len(chunk_text) // 2:  # only snap if newline is in the second half
                    chunk_text = chunk_text[:last_nl]

            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append(chunk_text)
            if end >= len(token_ids):
                break
            start += step

        return chunks

    # ──────────────────────────────────────────────────────────────────
    # Marker-based chunking helpers
    # ──────────────────────────────────────────────────────────────────

    def _detect_document_profile(self, text: str) -> dict:
        """One-shot LLM call on the first 2000 chars.

        Instead of classifying into a fixed domain taxonomy, we ask the LLM
        to describe the document in its own words and explain what a good
        semantic boundary looks like for THIS specific document. The result
        is passed verbatim into every marker prompt so the LLM has its own
        prior context when deciding where to cut.
        """
        sample = text[:2000]
        prompt = (
            "Read the following document excerpt carefully.\n\n"
            f"DOCUMENT EXCERPT:\n{sample}\n\n"
            "Answer these two questions in a JSON object with exactly these keys:\n"
            '  "description": In one sentence, what kind of document is this and how is it organized?\n'
            '  "boundary_hint": In one sentence, where do meaningful topic shifts occur '
            "in this document that would make good boundaries between independent knowledge chunks?\n\n"
            "Return ONLY the JSON object, nothing else."
        )
        try:
            raw = self.llm_text_generator(prompt, 128, 0.0)
            logger.info("[CHUNKER] Profile detection raw: %r", raw[:300])
            match = re.search(r"\{[\s\S]*?\}", raw)
            if match:
                profile = json.loads(match.group(0))
                if {"description", "boundary_hint"} <= set(profile.keys()):
                    result = {k: str(profile[k]) for k in ("description", "boundary_hint")}
                    logger.info("[CHUNKER] Document profile: %s | %s", result["description"], result["boundary_hint"])
                    return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CHUNKER] Profile detection failed (%s), using generic defaults", exc)
        defaults = {
            "description": "A document with sections of related content.",
            "boundary_hint": "Split where the subject matter shifts to a clearly different topic.",
        }
        logger.info("[CHUNKER] Using generic document profile")
        return defaults

    def _build_marker_prompt(self, numbered_text: str, profile: dict, n_lines: int) -> str:
        """Build a document-agnostic chunking prompt.

        The LLM receives:
          1. Its own description of what the document is (from profile detection).
          2. Its own judgment of where good boundaries lie.
          3. The numbered lines to read and split.

        No format-specific rules are hardcoded — no mention of markdown, bullets,
        or any structure hints. The LLM reads the actual content and decides.
        """
        return (
            "You are preparing a document for a retrieval-augmented generation (RAG) system.\n"
            "Your task is to identify where the document should be split into self-contained knowledge chunks.\n\n"
            f"About this document: {profile['description']}\n"
            f"Where to split: {profile['boundary_hint']}\n\n"
            "Below is the document passage with each line numbered [N].\n"
            "Read the content carefully and decide which line numbers should START a new chunk.\n"
            "Each chunk should cover one coherent topic so it can be retrieved independently.\n\n"
            "Output rules:\n"
            "- Line 0 must always be included\n"
            "- Choose split points where the meaning or subject changes significantly\n"
            "- Do not output any explanation or document text — only the array\n"
            "- Return ONLY a JSON integer array. Example: [0, 12, 28, 45]\n\n"
            f"NUMBERED LINES ({n_lines} total):\n{numbered_text}"
        )

    @staticmethod
    def _number_lines(text: str, preview_chars: int = 120) -> tuple[list[str], str]:
        """Number each line for the LLM marker prompt.

        Each line is truncated to `preview_chars`. The LLM needs enough text
        to understand what the line is about — it does not need every word of
        every bullet to find where the topic changes. Shorter lines mean a
        smaller prompt and less KV cache pressure on the GPU.
        """
        lines = text.split("\n")
        numbered = "\n".join(
            f"[{i}] {line[:preview_chars]}" for i, line in enumerate(lines)
        )
        return lines, numbered

    @staticmethod
    def _parse_line_markers(raw: str, n_lines: int) -> list[int]:
        match = re.search(r"\[[\d,\s]+\]", raw)
        if match:
            try:
                parsed = json.loads(match.group(0))
                valid = sorted({int(x) for x in parsed if 0 <= int(x) < n_lines})
                if valid:
                    return valid
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        nums = [int(m) for m in re.findall(r"\b(\d+)\b", raw) if 0 <= int(m) < n_lines]
        valid = sorted(set(nums))
        return valid if len(valid) >= 2 else []

    @staticmethod
    def _split_by_line_markers(lines: list[str], start_lines: list[int]) -> list[str]:
        if not start_lines or start_lines[0] != 0:
            start_lines = [0] + list(start_lines)
        start_lines = sorted(set(start_lines))
        chunks: list[str] = []
        for i, start in enumerate(start_lines):
            end = start_lines[i + 1] if i + 1 < len(start_lines) else len(lines)
            chunk = "\n".join(lines[start:end]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _save_debug_chunks(self, chunks: list[str]) -> None:
        save_dir = self.save_chunks_debug  # type: ignore[arg-type]
        if not os.path.isabs(save_dir):
            service_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_dir = os.path.join(service_root, save_dir.lstrip("./"))
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(save_dir, f"chunks_{ts}_{uuid.uuid4().hex[:8]}.jsonl")
        try:
            with open(fname, "w", encoding="utf-8") as fh:
                for index, chunk in enumerate(chunks):
                    fh.write(
                        json.dumps(
                            {"chunk_index": index, "chars": len(chunk), "text": chunk},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            logger.info("[CHUNKER] Saved %d chunks for review → %s", len(chunks), fname)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CHUNKER] Failed to save debug chunks: %s", exc)

    def _cleanup_chunks(self, chunks: list[str]) -> list[str]:
        """Clean up chunks while preserving structural whitespace (newlines).

        FIX (original bug): the old version ran re.sub(r"\\s+", " ", chunk) which
        collapsed ALL whitespace including newlines into a single space. This
        destroyed markdown structure — headers lost their ## prefix context,
        bullet lists became run-on sentences, making embedding similarity worse
        and LLM answers less accurate.

        New approach: only collapse runs of spaces/tabs on a single line;
        never touch newlines. Also collapse 3+ consecutive blank lines to 2.
        """
        result: list[str] = []
        for chunk in chunks:
            if not chunk or not chunk.strip():
                continue
            # Collapse horizontal whitespace (spaces/tabs) only, not newlines
            lines = chunk.split("\n")
            lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in lines]
            # Collapse 3+ consecutive blank lines to a single blank line
            cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
            if cleaned:
                result.append(cleaned)

        # Merge only genuinely tiny fragments (not just short headings)
        merged: list[str] = []
        for chunk in result:
            # Only merge if tiny AND doesn't start with a heading marker
            is_heading_start = chunk.lstrip().startswith("#")
            if merged and len(chunk) < self.min_chunk_chars and not is_heading_start:
                merged[-1] = f"{merged[-1]}\n{chunk}".strip()
            else:
                merged.append(chunk)
        return merged

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        normalized = text.replace("\n", " ")
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _normalize_text(text: str) -> str:
        # FIX: only collapse horizontal whitespace per line; preserve newlines
        lines = text.split("\n")
        lines = [re.sub(r"[ \t]+", " ", line) for line in lines]
        normalized = "\n".join(lines)
        # Collapse 3+ blank lines to 2
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    @staticmethod
    def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))
    