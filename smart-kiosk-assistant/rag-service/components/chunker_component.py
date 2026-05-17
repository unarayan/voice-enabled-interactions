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
        # Optional: path to save produced chunks as JSONL for manual review
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
        logger.info(
            "[CHUNKER] Done | total_chunks=%d | elapsed=%.1fs",
            len(records),
            elapsed,
        )
        return records

    # Marker-only output: just a JSON integer array of line-split indices.
    # Output is ~50-100 tokens regardless of passage size — no OOM risk.
    _MARKER_MAX_TOKENS = 256

    def _semantic_llm_chunks(self, text: str) -> list[str]:
        # ── Phase 0: detect document domain & structure once per ingest ──
        profile = self._detect_document_profile(text)

        # ── Phase 1: coarse token-window splitting ──
        if self.llm_tokenizer is not None and self.llm_passage_tokens > 0:
            coarse_passages = self._split_by_tokens(
                text, self.llm_passage_tokens, self.llm_passage_overlap_tokens,
            )
        else:
            coarse_passages = self._split_by_size(text, self.llm_passage_chars)

        total = len(coarse_passages)
        logger.info(
            "[CHUNKER] semantic_llm | passages=%d | domain=%s | %s",
            total, profile["domain"], profile["structure"],
        )

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

            # ── Phase 2: number the lines, ask LLM for split markers only ──
            lines, numbered = self._number_lines(passage)
            logger.info(
                "[CHUNKER] Passage %d/%d | chars=%d | lines=%d | requesting split markers",
                p_idx, total, passage_chars, len(lines),
            )
            logger.info("[CHUNKER] Passage %d/%d | preview: %r", p_idx, total, passage[:200])

            prompt = self._build_marker_prompt(numbered, profile, len(lines))
            prompt_chars = len(prompt)
            prompt_tokens: int | None = None
            if self.llm_tokenizer is not None:
                try:
                    prompt_tokens = len(self.llm_tokenizer.encode(prompt))
                except Exception:  # noqa: BLE001
                    pass
            logger.info(
                "[CHUNKER] Passage %d/%d | prompt_chars=%d | prompt_tokens=%s | max_new_tokens=%d",
                p_idx, total, prompt_chars, prompt_tokens, self._MARKER_MAX_TOKENS,
            )
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
        if self.overlap_chars <= 0 or len(chunks) < 2:
            return chunks

        with_overlap: list[str] = [chunks[0]]
        for index in range(1, len(chunks)):
            prefix = chunks[index - 1][-self.overlap_chars :].strip()
            current = chunks[index]
            combined = f"{prefix}\n{current}" if prefix else current
            with_overlap.append(combined.strip())
        return with_overlap

    def _split_by_size(self, text: str, max_chars: int) -> list[str]:
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
        if not paragraphs:
            return []

        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
            if len(paragraph) <= max_chars:
                current = paragraph
                continue

            sentence_buffer = ""
            for sentence in self._split_sentences(paragraph):
                sentence_candidate = f"{sentence_buffer} {sentence}".strip() if sentence_buffer else sentence
                if len(sentence_candidate) <= max_chars:
                    sentence_buffer = sentence_candidate
                    continue
                if sentence_buffer:
                    chunks.append(sentence_buffer)
                sentence_buffer = sentence
            current = sentence_buffer

        if current:
            chunks.append(current)
        return chunks

    def _split_by_tokens(self, text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
        tokenizer = self.llm_tokenizer
        if tokenizer is None:
            return self._split_by_size(text, self.llm_passage_chars)

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            return []

        chunks: list[str] = []
        start = 0
        step = max(max_tokens - max(overlap_tokens, 0), 1)

        while start < len(token_ids):
            end = min(start + max_tokens, len(token_ids))
            window_ids = token_ids[start:end]
            chunk_text = tokenizer.decode(window_ids, skip_special_tokens=True).strip()
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
        """One-shot LLM call on the first 2000 chars to identify domain and structure.
        Returns a dict with 'domain', 'structure', 'split_guidance' — fully dynamic,
        no hardcoded domain keywords in the detection logic."""
        sample = text[:2000]
        prompt = (
            "Analyze this document excerpt. Return ONLY a JSON object with these exact fields:\n"
            '  "domain": category such as "retail_store", "quick_service_restaurant", '
            '"banking", "airline", "hospital", "e_commerce", "generic"\n'
            '  "structure": one sentence describing how this document is organized '
            '(sections, headings, pattern)\n'
            '  "split_guidance": one sentence on what constitutes a natural chunk boundary '
            'for RAG knowledge retrieval\n\n'
            f"DOCUMENT EXCERPT:\n{sample}\n\n"
            "Return ONLY the JSON object, nothing else."
        )
        try:
            raw = self.llm_text_generator(prompt, 256, 0.0)
            logger.info("[CHUNKER] Profile detection raw: %r", raw[:300])
            match = re.search(r"\{[\s\S]*?\}", raw)
            if match:
                profile = json.loads(match.group(0))
                if {"domain", "structure", "split_guidance"} <= set(profile.keys()):
                    result = {k: str(profile[k]) for k in ("domain", "structure", "split_guidance")}
                    logger.info(
                        "[CHUNKER] Document profile: domain=%s | %s",
                        result["domain"], result["structure"],
                    )
                    return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CHUNKER] Profile detection failed (%s), using generic defaults", exc)
        defaults = {
            "domain": "generic",
            "structure": "The document has sections with headings followed by detailed content.",
            "split_guidance": "Split where the topic or section changes significantly.",
        }
        logger.info("[CHUNKER] Using generic document profile")
        return defaults

    def _build_marker_prompt(self, numbered_text: str, profile: dict, n_lines: int) -> str:
        """Dynamic chunking prompt that adapts to the detected document domain."""
        return (
            f"You are chunking a {profile['domain']} document for a RAG knowledge-base.\n"
            f"Document structure: {profile['structure']}\n"
            f"Chunking guidance: {profile['split_guidance']}\n\n"
            "Each line below is labeled [N]. Identify which line numbers should START a new knowledge chunk.\n\n"
            "Rules:\n"
            "- Always include 0 (line 0 always starts the first chunk)\n"
            "- Split where the topic, section, or entity changes meaningfully\n"
            f"- Target chunk size: {self.min_chunk_chars}\u2013{self.max_chunk_chars} characters of content\n"
            "- Return ONLY a JSON integer array, nothing else. Example: [0, 12, 28, 45]\n"
            "- Do NOT reproduce any text from the document\n\n"
            f"NUMBERED TEXT ({n_lines} lines):\n{numbered_text}"
        )

    @staticmethod
    def _number_lines(text: str) -> tuple[list[str], str]:
        """Prepend [N] to each line so the LLM can reference line positions."""
        lines = text.split("\n")
        numbered = "\n".join(f"[{i}] {line}" for i, line in enumerate(lines))
        return lines, numbered

    @staticmethod
    def _parse_line_markers(raw: str, n_lines: int) -> list[int]:
        """Extract sorted integer split-line indices from LLM output."""
        # Primary: canonical JSON array of integers
        match = re.search(r"\[[\d,\s]+\]", raw)
        if match:
            try:
                parsed = json.loads(match.group(0))
                valid = sorted({int(x) for x in parsed if 0 <= int(x) < n_lines})
                if valid:
                    return valid
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        # Graceful fallback: collect all in-range integers from the raw text
        nums = [int(m) for m in re.findall(r"\b(\d+)\b", raw) if 0 <= int(m) < n_lines]
        valid = sorted(set(nums))
        return valid if len(valid) >= 2 else []

    @staticmethod
    def _split_by_line_markers(lines: list[str], start_lines: list[int]) -> list[str]:
        """Slice the original lines at the detected split positions."""
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
        """Persist the final chunks that will be embedded for manual inspection."""
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
        cleaned = [re.sub(r"\s+", " ", chunk).strip() for chunk in chunks if chunk and chunk.strip()]
        merged: list[str] = []
        for chunk in cleaned:
            if merged and len(chunk) < self.min_chunk_chars:
                merged[-1] = f"{merged[-1]} {chunk}".strip()
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
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))
