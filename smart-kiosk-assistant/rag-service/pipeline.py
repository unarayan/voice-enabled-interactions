"""RAG pipeline — retrieval + OpenVINO LLM generation.

This module owns:

* A shared singleton ``RagPipeline`` that holds the LLM, embedding model,
  Chroma vector store and chunker.
* Ingest / retrieve / answer / stream-answer APIs used by the HTTP layer.
* An ``ov_genai`` text streamer adapter so token-by-token output can be
  yielded over Server-Sent Events.
"""

from __future__ import annotations

import gc
import logging
import pathlib
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Generator

import openvino_genai as ov_genai
from langchain_chroma import Chroma
from langchain_core.documents import Document

from components.chunker_component import SemanticChunker
from components.embedding_component import EmbeddingComponent
from utils.config_loader import config
from utils.ensure_model import ensure_llm_model, get_llm_model_path


logger = logging.getLogger(__name__)

_RESOURCE_ERROR_MARKERS = (
    "CL_OUT_OF_RESOURCES",
    "OUT OF MEMORY",
    "NOT ENOUGH MEMORY",
    "ALLOCATE",
)

_SHARED_PIPELINE: "RagPipeline | None" = None
_SHARED_PIPELINE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Token streaming
# ---------------------------------------------------------------------------
class _TokenStream:
    """Producer/consumer queue used as ``ov_genai`` streamer callback.

    ``ov_genai.LLMPipeline.generate(streamer=...)`` accepts any callable
    of the form ``Callable[[str], bool | StreamingStatus]``. The callable
    is invoked with each decoded text chunk; returning ``False`` (or
    ``StreamingStatus.RUNNING``) keeps generation going. We push chunks
    onto an internal queue so the consuming HTTP handler can iterate
    over us to receive tokens as they arrive.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._ended = False
        self.exc: Exception | None = None

    def __call__(self, chunk: str) -> bool:
        if chunk:
            self._queue.put(chunk)
        return False  # keep generating

    def end(self) -> None:
        if self._ended:
            return
        self._ended = True
        self._queue.put(None)

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            yield item


# ---------------------------------------------------------------------------
# Vector store glue
# ---------------------------------------------------------------------------
class _ChromaEmbeddingAdapter:
    def __init__(self, component: EmbeddingComponent) -> None:
        self._component = component

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._component.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._component.embed_query(text)


@dataclass(slots=True)
class RetrievalRecord:
    source: str
    content: str
    score: float | None
    metadata: dict


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class RagPipeline:
    """Singleton holding the LLM, embeddings, vector store, and chunker."""

    def __init__(self) -> None:
        ensure_llm_model()

        # Embedding + chunker (CPU, cheap to keep loaded).
        # Pass the LLM generate function only when strategy=llm; it uses the
        # same pipeline that answers questions, so the LLM must be loaded first.
        self.embedding_component = EmbeddingComponent()
        self._plugin_config = self._build_plugin_config(config.models.llm)
        self._llm_lock = threading.RLock()
        self._llm = self._load_llm()

        chunking_strategy = str(getattr(config.chunking, "strategy", "llm")).lower()
        llm_fn = self._chunker_llm_fn if chunking_strategy == "llm" else None
        self.chunker = SemanticChunker(self.embedding_component, llm_generate_fn=llm_fn)

        # Vector store
        store_cfg = config.storage
        self.persist_directory = store_cfg.persist_directory
        self.collection_name = store_cfg.collection_name
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=_ChromaEmbeddingAdapter(self.embedding_component),
        )

        # Retrieval params
        retr_cfg = config.retrieval
        self.top_k = int(getattr(retr_cfg, "top_k", 3))
        self.fetch_k = int(getattr(retr_cfg, "fetch_k", 6))
        self.max_context_chars = int(getattr(retr_cfg, "max_context_chars", 8000))
        self.score_threshold = getattr(retr_cfg, "score_threshold", None)
        self.include_source_markers = bool(getattr(config.answering, "include_source_markers", False))

        # LLM params
        llm_cfg = config.models.llm
        self.device = str(getattr(llm_cfg, "device", "CPU")).upper()
        self._model_path = get_llm_model_path()
        self._temperature = float(getattr(llm_cfg, "temperature", 0.0))
        self._default_max_new_tokens = int(getattr(config.answering, "max_tokens", 192))

    # ----- LLM lifecycle ------------------------------------------------
    def _build_plugin_config(self, llm_cfg) -> dict[str, str]:
        cfg: dict[str, str] = {}
        if self.device == "GPU":
            # f16 inference + f16 KV cache halves GPU memory vs f32 defaults;
            # NUM_STREAMS=1 avoids parallel execution buffers that multiply VRAM.
            cfg["INFERENCE_PRECISION_HINT"] = "f16"
            cfg["KV_CACHE_PRECISION"] = "f16"
            cfg["NUM_STREAMS"] = "1"

        cache_dir = getattr(llm_cfg, "cache_dir", None)
        if cache_dir:
            cache_path = pathlib.Path(cache_dir).expanduser().resolve()
            cache_path.mkdir(parents=True, exist_ok=True)
            cfg["CACHE_DIR"] = str(cache_path)
            logger.info("[llm] OpenVINO model cache at %s", cache_path)
        return cfg

    def _load_llm(self) -> ov_genai.LLMPipeline:
        logger.info("[llm] loading pipeline from %s on %s", self._model_path, self.device)
        if self._plugin_config:
            return ov_genai.LLMPipeline(self._model_path, self.device, **self._plugin_config)
        return ov_genai.LLMPipeline(self._model_path, self.device)

    def _reload_llm_locked(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
            gc.collect()
        self._llm = self._load_llm()

    def close(self) -> None:
        with self._llm_lock:
            if getattr(self, "_llm", None) is not None:
                del self._llm
                self._llm = None
                gc.collect()

    @staticmethod
    def _is_resource_error(exc: Exception) -> bool:
        message = str(exc).upper()
        return any(m in message for m in _RESOURCE_ERROR_MARKERS)

    def _chunker_llm_fn(self, prompt: str, max_tokens: int) -> str:
        """Thin wrapper that lets the chunker call the LLM without knowing
        about locks or generation kwargs."""
        return self._generate(prompt, max_tokens=max_tokens, temperature=0.0)

    # ----- Ingest -------------------------------------------------------
    def ingest_text(self, text: str, source: str = "api", metadata: dict | None = None) -> int:
        logger.info("[ingest] source=%s chars=%d", source, len(text))
        t0 = time.monotonic()

        records = self.chunker.chunk_text(text)
        if not records:
            logger.warning("[ingest] no chunks produced, aborting")
            return 0
        t_chunk = time.monotonic()

        docs = [
            Document(
                page_content=r.text,
                metadata={"source": source, "chunk_index": r.index, **(metadata or {})},
                id=str(uuid.uuid4()),
            )
            for r in records
        ]
        self.vectorstore.add_documents(docs)
        t_done = time.monotonic()
        logger.info(
            "[ingest] done chunks=%d chunking=%.1fs embed+upsert=%.1fs total=%.1fs",
            len(docs), t_chunk - t0, t_done - t_chunk, t_done - t0,
        )
        return len(docs)

    def clear_context(self) -> None:
        client = getattr(self.vectorstore, "_client", None)
        if client is not None:
            try:
                client.delete_collection(self.collection_name)
            except Exception:  # noqa: BLE001
                logger.info("[context] collection %s did not exist", self.collection_name)
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=_ChromaEmbeddingAdapter(self.embedding_component),
        )

    def get_stats(self) -> dict:
        collection = getattr(self.vectorstore, "_collection", None)
        count = collection.count() if collection is not None else None
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "document_count": count,
            "chunking_strategy": self.chunker.strategy,
            "llm_model": config.models.llm.hf_id,
            "embedding_model": config.models.embedding.hf_id,
        }

    # ----- Retrieve -----------------------------------------------------
    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievalRecord]:
        desired_k = top_k or self.top_k
        results = self.vectorstore.similarity_search_with_score(
            question, k=max(desired_k, self.fetch_k),
        )
        records: list[RetrievalRecord] = []
        for doc, score in results:
            if self.score_threshold is not None and score is not None and score > self.score_threshold:
                continue
            records.append(
                RetrievalRecord(
                    source=str(doc.metadata.get("source", "context")),
                    content=doc.page_content,
                    score=float(score) if score is not None else None,
                    metadata=doc.metadata,
                )
            )
            if len(records) >= desired_k:
                break
        return records

    # ----- Answering ----------------------------------------------------
    def plan_answer(
        self,
        question: str,
        context_text: str | None = None,
        top_k: int | None = None,
        system_prompt: str | None = None,
    ) -> tuple[str, list[RetrievalRecord]]:
        sources = self.retrieve(question, top_k=top_k)
        prompt = self._build_prompt(
            question, sources, context_text=context_text, system_prompt=system_prompt,
        )
        return prompt, sources

    def answer_question(
        self,
        question: str,
        context_text: str | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> dict:
        prompt, sources = self.plan_answer(question, context_text, top_k, system_prompt)
        answer = self.generate_from_prompt(prompt, max_tokens=max_tokens, temperature=temperature)
        return {"answer": answer.strip(), "sources": [self._source_payload(s) for s in sources]}

    def stream_answer(
        self,
        question: str,
        context_text: str | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> Generator[str, None, None]:
        prompt, _ = self.plan_answer(question, context_text, top_k, system_prompt)
        yield from self.stream_from_prompt(prompt, max_tokens=max_tokens, temperature=temperature)

    def generate_from_prompt(
        self, prompt: str, max_tokens: int | None = None, temperature: float | None = None,
    ) -> str:
        return self._generate(prompt, max_tokens, temperature)

    def stream_from_prompt(
        self, prompt: str, max_tokens: int | None = None, temperature: float | None = None,
    ) -> Generator[str, None, None]:
        yield from self._stream(prompt, max_tokens, temperature)

    def source_payloads(self, records: list[RetrievalRecord]) -> list[dict]:
        return [self._source_payload(r) for r in records]

    # ----- Prompt construction -----------------------------------------
    def _build_prompt(
        self,
        question: str,
        sources: list[RetrievalRecord],
        context_text: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        sys = (system_prompt or config.answering.system_prompt).strip()
        retrieved = self._build_context_block(sources)
        extra = (context_text or "").strip()
        fallback = (
            "If the retrieved context is insufficient, you may use general knowledge "
            "but state uncertainty clearly."
            if bool(getattr(config.answering, "fallback_to_general_knowledge", True))
            else "If the context is insufficient, say you do not have enough context."
        )

        parts = [sys, "", f"Question:\n{question.strip()}"]
        if retrieved:
            parts.extend(["", f"Retrieved context:\n{retrieved}"])
        if extra:
            parts.extend(["", f"Additional context:\n{extra}"])
        parts.extend(["", fallback, "Answer:"])
        return "\n".join(parts).strip()

    def _build_context_block(self, sources: list[RetrievalRecord]) -> str:
        parts: list[str] = []
        total = 0
        for i, rec in enumerate(sources, start=1):
            label = f"[{i}] {rec.source}" if self.include_source_markers else rec.source
            block = f"### SOURCE {label}\n{rec.content.strip()}"
            if total + len(block) > self.max_context_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts)

    @staticmethod
    def _source_payload(record: RetrievalRecord) -> dict:
        return {
            "source": record.source,
            "score": record.score,
            "metadata": record.metadata,
            "content": record.content,
        }

    # ----- Generation core ---------------------------------------------
    def _gen_kwargs(self, max_tokens: int | None, temperature: float | None) -> dict:
        temp = temperature if temperature is not None else self._temperature
        return {
            "temperature": max(temp, 1e-7),
            "do_sample": temp > 0.0,
            "max_new_tokens": max_tokens if max_tokens is not None else self._default_max_new_tokens,
        }

    def _generate(self, prompt: str, max_tokens: int | None, temperature: float | None) -> str:
        kwargs = self._gen_kwargs(max_tokens, temperature)
        with self._llm_lock:
            try:
                return str(self._llm.generate(prompt, **kwargs))
            except Exception as exc:  # noqa: BLE001
                if not self._is_resource_error(exc):
                    raise
                logger.warning("[llm] resource error, reloading pipeline and retrying: %s", exc)
                self._reload_llm_locked()
                return str(self._llm.generate(prompt, **kwargs))

    def _stream(
        self, prompt: str, max_tokens: int | None, temperature: float | None,
    ) -> Generator[str, None, None]:
        kwargs = self._gen_kwargs(max_tokens, temperature)
        streamer = _TokenStream()

        def _run() -> None:
            try:
                with self._llm_lock:
                    try:
                        self._llm.generate(prompt, streamer=streamer, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        if not self._is_resource_error(exc):
                            streamer.exc = exc
                            return
                        logger.warning("[llm] resource error during stream, reloading: %s", exc)
                        self._reload_llm_locked()
                        try:
                            self._llm.generate(prompt, streamer=streamer, **kwargs)
                        except Exception as retry_exc:  # noqa: BLE001
                            streamer.exc = retry_exc
            except Exception as exc:  # noqa: BLE001
                streamer.exc = exc
            finally:
                streamer.end()

        threading.Thread(target=_run, daemon=True).start()

        for token in streamer:
            yield token
        if streamer.exc is not None:
            raise streamer.exc


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------
def get_shared_pipeline() -> RagPipeline:
    global _SHARED_PIPELINE
    if _SHARED_PIPELINE is not None:
        return _SHARED_PIPELINE
    with _SHARED_PIPELINE_LOCK:
        if _SHARED_PIPELINE is None:
            _SHARED_PIPELINE = RagPipeline()
        return _SHARED_PIPELINE


def close_shared_pipeline() -> None:
    global _SHARED_PIPELINE
    with _SHARED_PIPELINE_LOCK:
        if _SHARED_PIPELINE is not None:
            _SHARED_PIPELINE.close()
            _SHARED_PIPELINE = None
