from __future__ import annotations

import ctypes
import gc
import logging
import pathlib
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Generator

from components.ov_ir_llm import OVIRTextGenPipeline
from langchain_chroma import Chroma
from langchain_core.documents import Document
from transformers import AutoTokenizer

from components.chunker_component import SemanticChunker
from components.embedding_component import EmbeddingComponent
from utils.config_loader import config
from utils.ensure_model import ensure_llm_model, get_llm_model_path


logger = logging.getLogger(__name__)

_SHARED_PIPELINE: "RagPipeline | None" = None
_SHARED_PIPELINE_LOCK = threading.Lock()



class ChromaEmbeddingAdapter:
    def __init__(self, component: EmbeddingComponent) -> None:
        self.component = component

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.component.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.component.embed_query(text)


@dataclass(slots=True)
class RetrievalRecord:
    source: str
    content: str
    score: float | None
    metadata: dict


class RagPipeline:
    def __init__(self) -> None:
        # Ensure model is exported to OpenVINO IR before anything else
        ensure_llm_model()

        self.embedding_component = EmbeddingComponent()

        storage_cfg = config.storage
        self.persist_directory = storage_cfg.persist_directory
        self.collection_name = storage_cfg.collection_name
        self.top_k = int(getattr(config.retrieval, "top_k", 3))
        self.fetch_k = int(getattr(config.retrieval, "fetch_k", 6))
        self.max_context_chars = int(getattr(config.retrieval, "max_context_chars", 16000))
        self.score_threshold = getattr(config.retrieval, "score_threshold", None)
        self.include_source_markers = bool(getattr(config.answering, "include_source_markers", False))

        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=ChromaEmbeddingAdapter(self.embedding_component),
        )

        llm_cfg = config.models.llm
        self._llm_cfg = llm_cfg
        self._model_path = get_llm_model_path()
        self._device = str(getattr(llm_cfg, "device", "CPU")).upper()
        self._temperature = float(getattr(llm_cfg, "temperature", 0.0))
        self._default_max_new_tokens = int(getattr(config.answering, "max_tokens", 192))
        self._max_generations_before_reload = int(
            getattr(config.answering, "max_generations_before_reload", 25)
        )
        self._generation_timeout = float(
            getattr(config.answering, "generation_timeout_secs", 90.0)
        )
        self._generations_since_reload = 0

        # Tokenizer and pipeline are loaded once at startup and shared behind a lock.
        logger.info(
            "Loading HF tokenizer for %s (model path: %s, device: %s)",
            llm_cfg.hf_id, self._model_path, self._device,
        )
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, fix_mistral_regex=True)
        except TypeError:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._llm_lock = threading.RLock()
        self._llm = self._load_llm()

        self.chunker = SemanticChunker(
            self.embedding_component,
            self._generate_text,
            llm_tokenizer=self._tokenizer,
        )

    def _build_ov_config(self) -> dict:
        cfg: dict[str, str] = {}
        if self._device == "GPU":
            # cfg["KV_CACHE_PRECISION"] = "u8"
            # cfg["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = "32"
            # cfg["NUM_STREAMS"] = "1"
            # cfg["GPU_HOST_TASK_PRIORITY"] = "HIGH"
            pass
        # cfg["PERFORMANCE_HINT"] = "LATENCY"
        cache_dir = getattr(self._llm_cfg, "cache_dir", None)
        if cache_dir:
            cache_path = pathlib.Path(cache_dir).expanduser().resolve()
            cache_path.mkdir(parents=True, exist_ok=True)
            cfg["CACHE_DIR"] = str(cache_path)
        logger.info("[LLM] ov_config=%s", cfg)
        return cfg

    def _load_llm(self) -> OVIRTextGenPipeline:
        logger.info(
            "[LLM] Loading OVIRTextGenPipeline from %s on %s",
            self._model_path, self._device,
        )
        return OVIRTextGenPipeline(
            model_path=self._model_path,
            tokenizer=self._tokenizer,
            device=self._device,
            ov_config=self._build_ov_config(),
            generation_timeout=self._generation_timeout,
        )

    def _destroy_llm(self, model: OVIRTextGenPipeline) -> None:
        try:
            model.destroy()
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:  # noqa: BLE001
                pass
            logger.info("[LLM] Pipeline destroyed, memory reclaimed")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[LLM] Failed to fully destroy pipeline: %s", exc)

    def close(self) -> None:
        with self._llm_lock:
            if getattr(self, "_llm", None) is not None:
                self._destroy_llm(self._llm)
                self._llm = None

    @staticmethod
    def _is_resource_exhaustion(exc: Exception) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        message = str(exc).upper()
        return any(
            marker in message
            for marker in (
                "CL_OUT_OF_RESOURCES",
                "OUT OF MEMORY",
                "NOT ENOUGH MEMORY",
                "ALLOCATE",
                "EXCEEDED MAX SIZE OF MEMORY ALLOCATION",
            )
        )

    def _reload_llm_locked(self) -> None:
        if getattr(self, "_llm", None) is not None:
            self._destroy_llm(self._llm)
            self._llm = None
        # Give the GPU driver time to actually reclaim pages before reloading.
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
        self._llm = self._load_llm()
        self._generations_since_reload = 0

    def _post_generation_locked(self) -> None:
        """Cleanup after a successful generation.

        OVIRTextGenPipeline has no persistent GPU state — each generate() call
        allocates and frees its own InferRequest.  We still run gc.collect and
        malloc_trim to promptly return Python/libc heap pages to the OS, and
        proactively reload the compiled model at the configured threshold to
        prevent any long-term GPU allocator fragmentation.
        """
        self._generations_since_reload += 1
        try:
            gc.collect()
        except Exception:  # noqa: BLE001
            pass
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:  # noqa: BLE001
            pass
        if (
            self._max_generations_before_reload > 0
            and self._generations_since_reload >= self._max_generations_before_reload
        ):
            logger.info(
                "[LLM] Reached %d generations; recycling pipeline proactively",
                self._generations_since_reload,
            )
            self._reload_llm_locked()

    def ingest_text(self, text: str, source: str = "api", metadata: dict | None = None) -> int:
        logger.info("[INGEST] Starting | source=%s | input_chars=%d", source, len(text))
        t0 = time.monotonic()

        # Remove any existing docs for this source so re-ingestion replaces
        # rather than accumulates (prevents stale duplicates across runs).
        try:
            collection = getattr(self.vectorstore, "_collection", None)
            if collection is not None:
                existing = collection.get(where={"source": source}, include=[])
                if existing["ids"]:
                    collection.delete(ids=existing["ids"])
                    logger.info(
                        "[INGEST] Removed %d stale docs for source=%s",
                        len(existing["ids"]), source,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[INGEST] Could not purge old docs for source=%s: %s", source, exc)

        chunks = self.chunker.chunk_text(text)
        t_chunk = time.monotonic()
        logger.info(
            "[INGEST] Chunking done | chunks=%d | elapsed=%.1fs",
            len(chunks), t_chunk - t0,
        )

        if not chunks:
            logger.warning("[INGEST] No chunks produced — ingestion aborted")
            return 0

        docs = [
            Document(
                page_content=chunk.text,
                metadata={
                    "source": source,
                    "chunk_index": chunk.index,
                    **(metadata or {}),
                },
                id=str(uuid.uuid4()),
            )
            for chunk in chunks
        ]

        logger.info("[INGEST] Embedding + upserting %d docs into vectorstore...", len(docs))
        self.vectorstore.add_documents(docs)
        t_done = time.monotonic()
        logger.info(
            "[INGEST] Done | docs_added=%d | embed+upsert=%.1fs | total=%.1fs",
            len(docs), t_done - t_chunk, t_done - t0,
        )
        return len(docs)

    def clear_context(self) -> None:
        client = getattr(self.vectorstore, "_client", None)
        if client is None:
            raise RuntimeError("Vector store client is not available")
        try:
            client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001
            logger.info("Collection %s did not exist yet during clear_context", self.collection_name)
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=ChromaEmbeddingAdapter(self.embedding_component),
        )

    def get_stats(self) -> dict:
        collection = getattr(self.vectorstore, "_collection", None)
        count = collection.count() if collection is not None else None
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "document_count": count,
            "chunking_strategy": "semantic_llm",
            "llm_model": config.models.llm.hf_id,
            "embedding_model": config.models.embedding.hf_id,
        }

    def answer_question(
        self,
        question: str,
        context_text: str | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> dict:
        prompt, sources = self.plan_answer(
            question,
            context_text=context_text,
            top_k=top_k,
            system_prompt=system_prompt,
        )
        answer = self.generate_from_prompt(prompt, max_tokens=max_tokens, temperature=temperature)
        return {
            "answer": answer.strip(),
            "sources": self.source_payloads(sources),
        }

    def stream_answer(
        self,
        question: str,
        context_text: str | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> Generator[str, None, None]:
        prompt, _ = self.plan_answer(
            question,
            context_text=context_text,
            top_k=top_k,
            system_prompt=system_prompt,
        )
        yield from self.stream_from_prompt(prompt, max_tokens=max_tokens, temperature=temperature)

    def plan_answer(
        self,
        question: str,
        context_text: str | None = None,
        top_k: int | None = None,
        system_prompt: str | None = None,
    ) -> tuple[str, list[RetrievalRecord]]:
        sources = self.retrieve(question, top_k=top_k)
        prompt = self._build_prompt(question, sources, context_text=context_text, system_prompt=system_prompt)
        return prompt, sources

    def generate_from_prompt(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        return self._generate_text(prompt, max_tokens=max_tokens, temperature=temperature)

    def stream_from_prompt(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Generator[str, None, None]:
        yield from self._stream_generate(prompt, max_tokens=max_tokens, temperature=temperature)

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievalRecord]:
        desired_k = top_k or self.top_k
        docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=max(desired_k, self.fetch_k))
        records: list[RetrievalRecord] = []
        for document, score in docs_with_scores:
            if self.score_threshold is not None and score is not None and score > self.score_threshold:
                continue
            records.append(
                RetrievalRecord(
                    source=str(document.metadata.get("source", "context")),
                    content=document.page_content,
                    score=float(score) if score is not None else None,
                    metadata=document.metadata,
                )
            )
            if len(records) >= desired_k:
                break
        return records

    def _build_prompt(
        self,
        question: str,
        sources: list[RetrievalRecord],
        context_text: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        prompt_system = system_prompt or config.answering.system_prompt
        retrieved_context = self._build_context_block(sources)
        extra_context = (context_text or "").strip()
        fallback_hint = (
            "If the retrieved store context is insufficient, you may use general retail knowledge but state uncertainty clearly."
            if bool(getattr(config.answering, "fallback_to_general_knowledge", True))
            else "If the context is insufficient, say you do not have enough store context."
        )

        prompt = [prompt_system.strip(), "", f"Customer question:\n{question.strip()}"]
        if retrieved_context:
            prompt.extend(["", f"Retrieved store context:\n{retrieved_context}"])
        if extra_context:
            prompt.extend(["", f"Runtime context passed by caller:\n{extra_context}"])
        prompt.extend(["", fallback_hint, "Answer:"])
        return "\n".join(prompt).strip()

    def _build_context_block(self, sources: list[RetrievalRecord]) -> str:
        parts: list[str] = []
        total_chars = 0
        for index, record in enumerate(sources, start=1):
            label = f"[{index}] {record.source}" if self.include_source_markers else record.source
            block = f"### SOURCE {label}\n{record.content.strip()}"
            if total_chars + len(block) > self.max_context_chars:
                break
            parts.append(block)
            total_chars += len(block)
        return "\n\n".join(parts)

    def _generation_kwargs(self, max_tokens: int | None, temperature: float | None) -> dict:
        temp = temperature if temperature is not None else self._temperature
        return {
            "max_new_tokens": max_tokens if max_tokens is not None else self._default_max_new_tokens,
            "temperature": temp,
            "do_sample": temp > 0.0,
        }

    def _generate_text(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> str:
        gen_kwargs = self._generation_kwargs(max_tokens=max_tokens, temperature=temperature)

        with self._llm_lock:
            try:
                result = self._llm.generate(prompt, **gen_kwargs)
            except Exception as exc:  # noqa: BLE001
                if not self._is_resource_exhaustion(exc):
                    raise
                logger.warning("[LLM] Generation hit resource exhaustion or timeout; recycling pipeline and retrying once: %s", exc)
                self._reload_llm_locked()
                result = self._llm.generate(prompt, **gen_kwargs)
            self._post_generation_locked()
        return str(result)

    def _stream_generate(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> Generator[str, None, None]:
        gen_kwargs = self._generation_kwargs(max_tokens=max_tokens, temperature=temperature)
        with self._llm_lock:
            try:
                streamer = self._llm.generate_stream(prompt, **gen_kwargs)
                try:
                    for token in streamer:
                        yield token
                except queue.Empty:
                    # TextIteratorStreamer raises queue.Empty when no token arrives
                    # within its timeout window — treat it the same as a GPU hang.
                    raise TimeoutError(
                        f"LLM streaming exceeded {self._generation_timeout:.0f}s — GPU may be hung"
                    )
            except Exception as exc:  # noqa: BLE001
                if not self._is_resource_exhaustion(exc):
                    raise
                logger.warning(
                    "[LLM] Streaming hit resource exhaustion or timeout; recycling and falling back: %s", exc
                )
                self._reload_llm_locked()
                result = self._llm.generate(prompt, **gen_kwargs)
                if result:
                    yield result
            finally:
                self._post_generation_locked()

    @staticmethod
    def source_payload(record: RetrievalRecord) -> dict:
        return {
            "source": record.source,
            "score": record.score,
            "metadata": record.metadata,
            "content": record.content,
        }

    def source_payloads(self, records: list[RetrievalRecord]) -> list[dict]:
        return [self.source_payload(record) for record in records]
def close_shared_pipeline() -> None:
    global _SHARED_PIPELINE
    with _SHARED_PIPELINE_LOCK:
        if _SHARED_PIPELINE is not None:
            _SHARED_PIPELINE.close()
            _SHARED_PIPELINE = None


def get_shared_pipeline() -> RagPipeline:
    global _SHARED_PIPELINE
    if _SHARED_PIPELINE is not None:
        return _SHARED_PIPELINE

    with _SHARED_PIPELINE_LOCK:
        if _SHARED_PIPELINE is None:
            _SHARED_PIPELINE = RagPipeline()
        return _SHARED_PIPELINE
