# Configuration

The service loads `config.yaml` at the repo root and (optionally) merges YAML
overlay files listed in `SMART_KIOSK_RAG_CONFIG_OVERRIDE_PATHS`, then applies
environment overrides whose names start with `SMART_KIOSK_RAG__`.

For example, `config.container.yaml` only sets the keys that differ from
`config.yaml` for the Docker deployment.

## `server`

| Key | Default | Description |
|---|---|---|
| `host` | `0.0.0.0` | Bind address |
| `port` | `8020` | Service port |

## `api`

| Key | Default | Description |
|---|---|---|
| `cors_allow_origins` | `http://127.0.0.1`, `http://localhost` | Allowed browser origins |
| `openai_model_name` | `smart-kiosk-rag` | Model name returned by the OpenAI-compatible endpoints |

## `models.llm`

| Key | Default | Description |
|---|---|---|
| `hf_id` | `Qwen/Qwen2.5-7B-Instruct` | Hugging Face model ID |
| `device` | `GPU` | OpenVINO device (`GPU`, `CPU`, `AUTO`) |
| `weight_format` | `int8` | Export precision (`fp32`, `fp16`, `int8`, `int4`) |
| `models_base_path` | `./models/llm` | Local cache root |
| `temperature` | `0.0` | Default sampling temperature |
| `cache_dir` | `./storage/ov_cache` | OpenVINO compiled-kernel cache (speeds up restarts) |

## `models.embedding`

| Key | Default | Description |
|---|---|---|
| `hf_id` | `BAAI/bge-large-en-v1.5` | Hugging Face embedding model ID |
| `device` | `CPU` | `CPU` or `cuda` (only used by `sentence-transformers`) |
| `models_base_path` | `./models/embeddings` | Local cache root |
| `normalize_embeddings` | `true` | Normalize vectors for cosine retrieval |

## `storage`

| Key | Default | Description |
|---|---|---|
| `persist_directory` | `./storage/vector_db` | Chroma persistence directory |
| `collection_name` | `smart-kiosk-assistant-bge-large` | Chroma collection name |

## `retrieval`

| Key | Default | Description |
|---|---|---|
| `top_k` | `3` | Chunks returned to the prompt |
| `fetch_k` | `6` | Candidates fetched from the vector store |
| `max_context_chars` | `5000` (host) / `8000` (container) | Character budget for the retrieved-context block |
| `score_threshold` | `null` | Optional cutoff on Chroma's distance score |

## `chunking`

| Key | Default | Description |
|---|---|---|
| `strategy` | `semantic` | `semantic` (embedding-based, header-aware) or `fixed` (recursive char split) |
| `max_chunk_chars` | `1200` | Upper bound per chunk |
| `min_chunk_chars` | `200` | Tiny-chunk merge threshold |
| `overlap_chars` | `150` | Character overlap added between adjacent chunks |
| `semantic_similarity_threshold` | `0.72` | Cosine boundary for the `semantic` strategy |

### Picking a strategy

* **`semantic`** — Best for narrative or mixed content. Honors markdown
  headers as hard boundaries, then uses embedding similarity to detect
  topic shifts inside each section. Embedding runs on CPU; ingest is
  GPU-free.
* **`fixed`** — Deterministic recursive split on paragraph → line →
  sentence → word boundaries. Best for tabular or code-like content,
  or when you want fully reproducible chunking.

## `answering`

| Key | Default | Description |
|---|---|---|
| `system_prompt` | retail assistant prompt | Base instruction prepended to every answer prompt |
| `fallback_to_general_knowledge` | `true` | Permit non-context answers when retrieval is weak |
| `include_source_markers` | `false` | Add `[N]` markers before each source block |
| `max_tokens` | `192` | Default generation cap |

## Environment Overrides

Use double underscores to target nested keys:

```bash
SMART_KIOSK_RAG__MODELS__LLM__DEVICE=CPU
SMART_KIOSK_RAG__RETRIEVAL__TOP_K=8
SMART_KIOSK_RAG__CHUNKING__STRATEGY=fixed
```

Use `SMART_KIOSK_RAG_CONFIG_OVERRIDE_PATHS` for a comma-separated list of
YAML overlay files (relative paths are resolved against the service root).
