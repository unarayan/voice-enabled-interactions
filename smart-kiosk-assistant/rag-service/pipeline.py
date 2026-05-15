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
from transformers import AutoTokenizer

from components.chunker_component import SemanticChunker
from components.embedding_component import EmbeddingComponent
from utils.config_loader import config
from utils.ensure_model import ensure_llm_model, get_llm_model_path


logger = logging.getLogger(__name__)

_SHARED_PIPELINE: "RagPipeline | None" = None
_SHARED_PIPELINE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Streaming helper — matches the smart-classroom ov_genai_util pattern exactly,
# using put() which is the correct openvino_genai.StreamerBase interface.
# ---------------------------------------------------------------------------
class YieldingTextStreamer(ov_genai.StreamerBase):
    def __init__(self, tokenizer, skip_special_tokens: bool = True) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._token_cache: list[int] = []
        self._print_len = 0
        self._exc: Exception | None = None
        self._ended = False  # FIX: guard against double end()

    def put(self, token_id: int) -> bool:
        self._token_cache.append(token_id)
        text = self.tokenizer.decode(self._token_cache, skip_special_tokens=self.skip_special_tokens)
        new_text = text[self._print_len:]
        if not new_text:
            return False
        if self._is_safe_to_emit(new_text):
            self._queue.put(new_text)
            self._print_len = len(text)
        else:
            last_token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            if last_token_text.startswith(" "):
                prev_chunk = text[self._print_len: len(text) - len(last_token_text)]
                if prev_chunk:
                    self._queue.put(prev_chunk)
                    self._print_len += len(prev_chunk)
        return False

    def end(self) -> None:
        # FIX: ov_genai may call end() internally; guard against double-call
        # to avoid pushing two None sentinels which corrupts the iterator.
        if self._ended:
            return
        self._ended = True
        if self._token_cache:
            text = self.tokenizer.decode(self._token_cache, skip_special_tokens=self.skip_special_tokens)
            remaining = text[self._print_len:]
            if remaining:
                self._queue.put(remaining)
        self._queue.put(None)
        self._token_cache.clear()
        self._print_len = 0

    def __iter__(self):
        while True:
            token = self._queue.get()
            if token is None:
                break
            yield token

    @staticmethod
    def _is_safe_to_emit(text: str) -> bool:
        last = text[-1]
        cp = ord(last)
        return (
            last.isspace()
            or last == "\n"
            or last in {".", ",", "!", "?", ";", ":"}
            or 0x4E00 <= cp <= 0x9FFF  # CJK
        )


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
        self._model_path = get_llm_model_path()
        self._device = str(getattr(llm_cfg, "device", "CPU")).upper()
        self._temperature = float(getattr(llm_cfg, "temperature", 0.0))
        self._default_max_new_tokens = int(getattr(config.answering, "max_tokens", 192))
        self._chunker_max_new_tokens = int(getattr(config.answering, "chunker_max_tokens", 128))
        self._max_generations_before_reload = int(
            getattr(config.answering, "max_generations_before_reload", 25)
        )
        self._generations_since_reload = 0

        # Plugin properties — tuned for iGPU to reduce memory pressure.
        # FIX: f16 inference + f16 KV cache halves GPU memory vs f32 defaults.
        # FIX: NUM_STREAMS=1 prevents iGPU from allocating parallel execution
        #      buffers which multiply peak VRAM usage.
        self._plugin_config: dict[str, str] = {}

        if self._device == "GPU":
            self._plugin_config["INFERENCE_PRECISION_HINT"] = "f16"
            self._plugin_config["KV_CACHE_PRECISION"] = "f16"
            self._plugin_config["NUM_STREAMS"] = "1"
            logger.info(
                "[LLM] iGPU detected — applied memory-saving plugin config: %s",
                self._plugin_config,
            )

        cache_dir = getattr(llm_cfg, "cache_dir", None)
        if cache_dir:
            cache_path = pathlib.Path(cache_dir).expanduser().resolve()
            cache_path.mkdir(parents=True, exist_ok=True)
            self._plugin_config["CACHE_DIR"] = str(cache_path)
            logger.info("[LLM] OpenVINO model cache enabled at %s", cache_path)

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

        # FIX: Use a dedicated wrapper with capped max_new_tokens for chunking.
        # The chunker only needs short completions (section titles, boundaries),
        # so passing max_tokens=None → _default_max_new_tokens was wasteful and
        # caused the LLMPipeline to pre-allocate large KV cache on first call,
        # triggering CL_OUT_OF_RESOURCES on iGPU before any real work happened.
        self.chunker = SemanticChunker(
            self.embedding_component,
            self._chunker_generate,
            llm_tokenizer=self._tokenizer,
        )

    def _chunker_generate(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> str:
        """Thin wrapper used exclusively by SemanticChunker.

        Caps max_new_tokens to _chunker_max_new_tokens so the iGPU KV cache
        allocation stays well within available memory. The caller-supplied
        max_tokens is respected only if it is smaller than the cap.
        Temperature defaults to 0 for deterministic chunking decisions.
        """
        capped = self._chunker_max_new_tokens
        if max_tokens is not None:
            capped = min(max_tokens, capped)
        return self._generate_text(
            prompt,
            max_tokens=capped,
            temperature=temperature if temperature is not None else 0.0,
        )

    def _load_llm(self) -> ov_genai.LLMPipeline:
        logger.info(
            "[LLM] Loading ov_genai.LLMPipeline from %s on %s (plugin_config=%s)",
            self._model_path, self._device, self._plugin_config or "{}",
        )
        if self._plugin_config:
            return ov_genai.LLMPipeline(self._model_path, self._device, **self._plugin_config)
        return ov_genai.LLMPipeline(self._model_path, self._device)

    def _destroy_llm(self, model: ov_genai.LLMPipeline) -> None:
        try:
            del model
            gc.collect()
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
        message = str(exc).upper()
        return any(
            marker in message
            for marker in (
                "CL_OUT_OF_RESOURCES",
                "OUT OF MEMORY",
                "NOT ENOUGH MEMORY",
                "ALLOCATE",
            )
        )

    def _reload_llm_locked(self) -> None:
        if getattr(self, "_llm", None) is not None:
            self._destroy_llm(self._llm)
            self._llm = None
        self._llm = self._load_llm()
        self._generations_since_reload = 0

    def _post_generation_locked(self) -> None:
        """Run cleanup after a successful generation while holding _llm_lock.

        Increments the generation counter, runs gc.collect to release Python
        references to intermediate tensors, and recycles the LLMPipeline once
        the configured threshold is reached to avoid GPU memory fragmentation
        / KV cache buildup that eventually triggers CL_OUT_OF_RESOURCES.
        """
        self._generations_since_reload += 1
        try:
            gc.collect()
        except Exception:  # noqa: BLE001
            pass
        if (
            self._max_generations_before_reload > 0
            and self._generations_since_reload >= self._max_generations_before_reload
        ):
            logger.info(
                "[LLM] Reached %d generations since last reload; recycling pipeline proactively",
                self._generations_since_reload,
            )
            self._reload_llm_locked()

    def ingest_text(self, text: str, source: str = "api", metadata: dict | None = None) -> int:
        logger.info("[INGEST] Starting | source=%s | input_chars=%d", source, len(text))
        t0 = time.monotonic()

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
        kwargs: dict = {
            "temperature": max(temp, 1e-7),
            "do_sample": temp > 0.0,
        }
        kwargs["max_new_tokens"] = max_tokens if max_tokens is not None else self._default_max_new_tokens
        return kwargs

    def _generate_text(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> str:
        gen_kwargs = self._generation_kwargs(max_tokens=max_tokens, temperature=temperature)

        with self._llm_lock:
            try:
                result = self._llm.generate(prompt, **gen_kwargs)
            except Exception as exc:  # noqa: BLE001
                if not self._is_resource_exhaustion(exc):
                    raise
                logger.warning("[LLM] Generation hit resource exhaustion; recycling pipeline and retrying once: %s", exc)
                self._reload_llm_locked()
                result = self._llm.generate(prompt, **gen_kwargs)
            self._post_generation_locked()
        return str(result)

    def _stream_generate(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> Generator[str, None, None]:
        gen_kwargs = self._generation_kwargs(max_tokens=max_tokens, temperature=temperature)

        streamer = YieldingTextStreamer(self._tokenizer)

        def _run_generation() -> None:
            # FIX: Do NOT call streamer.end() in finally — ov_genai calls it
            # internally when generation completes normally. Calling it again
            # pushed a second None sentinel into the queue, which either caused
            # the iterator to stop one token early or (in some OV builds) triggered
            # a second forward pass leading to a spurious OOM on the GPU.
            # We only manually signal end() in the error path where ov_genai
            # may not have had a chance to call it itself.
            try:
                with self._llm_lock:
                    try:
                        self._llm.generate(prompt, streamer=streamer, **gen_kwargs)
                    except Exception as exc:  # noqa: BLE001
                        if not self._is_resource_exhaustion(exc):
                            logger.error("[LLM] Stream generation failed: %s", exc)
                            streamer._exc = exc
                            streamer.end()  # ensure iterator unblocks on error
                            return
                        logger.warning(
                            "[LLM] Stream generation hit resource exhaustion; recycling pipeline and retrying once: %s", exc
                        )
                        self._reload_llm_locked()
                        try:
                            self._llm.generate(prompt, streamer=streamer, **gen_kwargs)
                        except Exception as retry_exc:  # noqa: BLE001
                            logger.error("[LLM] Stream generation failed after retry: %s", retry_exc)
                            streamer._exc = retry_exc
                            streamer.end()  # ensure iterator unblocks on error
                            return
                    self._post_generation_locked()
            except Exception as exc:  # noqa: BLE001
                logger.error("[LLM] Unexpected stream generation error: %s", exc)
                streamer._exc = exc
                streamer.end()  # ensure iterator unblocks on error

        thread = threading.Thread(target=_run_generation, daemon=True)
        thread.start()

        for token in streamer:
            yield token

        if streamer._exc is not None:
            raise streamer._exc

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