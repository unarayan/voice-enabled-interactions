from __future__ import annotations

import gc
import logging
import pathlib
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Generator

from langchain_chroma import Chroma
from langchain_core.documents import Document
from transformers import TextIteratorStreamer

from components.chunker_component import SemanticChunker
from components.embedding_component import EmbeddingComponent
from components.llm_component import OVLLMComponent
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
        self._model_path = get_llm_model_path()
        self._device = str(getattr(llm_cfg, "device", "CPU")).upper()
        self._temperature = float(getattr(llm_cfg, "temperature", 0.0))
        self._default_max_new_tokens = int(getattr(config.answering, "max_tokens", 192))
        self._max_generations_before_reload = int(
            getattr(config.answering, "max_generations_before_reload", 25)
        )
        self._generations_since_reload = 0

        # Plugin properties — primarily CACHE_DIR so periodic reloads skip
        # kernel compilation on the GPU and complete in seconds.
        self._plugin_config: dict[str, str] = {}
        cache_dir = getattr(llm_cfg, "cache_dir", None)
        if cache_dir:
            cache_path = pathlib.Path(cache_dir).expanduser().resolve()
            cache_path.mkdir(parents=True, exist_ok=True)
            self._plugin_config["CACHE_DIR"] = str(cache_path)
            logger.info("[LLM] OpenVINO model cache enabled at %s", cache_path)

        # Tokenizer and pipeline are loaded once at startup and shared behind a lock.
        self._llm_lock = threading.RLock()
        self._llm = self._load_llm()
        # Tokenizer is owned by OVLLMComponent; expose it for SemanticChunker.
        self._tokenizer = self._llm.tokenizer

        self.chunker = SemanticChunker(
            self.embedding_component,
            self._generate_text,
            llm_tokenizer=self._tokenizer,
        )

    def _load_llm(self) -> OVLLMComponent:
        return OVLLMComponent(
            model_path=self._model_path,
            device=self._device,
            ov_config=self._plugin_config if self._plugin_config else None,
        )

    def _destroy_llm(self, model: OVLLMComponent) -> None:
        model.destroy()

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
        self._tokenizer = self._llm.tokenizer
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
        t0 = time.monotonic()
        logger.info(
            "[LLM] _generate_text | prompt_chars=%d | max_tokens=%s | temperature=%s",
            len(prompt), max_tokens, temperature,
        )
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
        logger.info(
            "[LLM] _generate_text | done | result_chars=%d | elapsed=%.2fs",
            len(result), time.monotonic() - t0,
        )
        return result

    def _stream_generate(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> Generator[str, None, None]:
        gen_kwargs = self._generation_kwargs(max_tokens=max_tokens, temperature=temperature)

        streamer = TextIteratorStreamer(self._tokenizer, skip_special_tokens=True, skip_prompt=True)
        error_holder: list[Exception | None] = [None]

        def _run_generation() -> None:
            try:
                with self._llm_lock:
                    try:
                        self._llm.generate_with_streamer(prompt, streamer, **gen_kwargs)
                    except Exception as exc:  # noqa: BLE001
                        if not self._is_resource_exhaustion(exc):
                            raise
                        logger.warning("[LLM] Stream generation hit resource exhaustion; recycling pipeline and retrying once: %s", exc)
                        self._reload_llm_locked()
                        self._llm.generate_with_streamer(prompt, streamer, **gen_kwargs)
                    self._post_generation_locked()
            except Exception as exc:  # noqa: BLE001
                logger.error("[LLM] Stream generation failed: %s", exc)
                error_holder[0] = exc
            finally:
                # Unblock the consumer if generation exited before sending the stop signal.
                try:
                    streamer.text_queue.put(streamer.stop_signal)
                except Exception:  # noqa: BLE001
                    pass

        thread = threading.Thread(target=_run_generation, daemon=True)
        thread.start()

        for token in streamer:
            yield token

        if error_holder[0] is not None:
            raise error_holder[0]

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
