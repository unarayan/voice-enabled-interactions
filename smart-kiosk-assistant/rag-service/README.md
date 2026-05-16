# RAG Service

A FastAPI retrieval-augmented question-answering service built on
**OpenVINO GenAI**, **Chroma**, and **sentence-transformers**. It powers
the Smart Kiosk Assistant but is content- and domain-agnostic — point
it at any text corpus and it works.

## Highlights

* **Streaming kiosk API** — `POST /api/v1/query` returns SSE token stream.
* **OpenAI-compatible chat completions** — `POST /v1/chat/completions`
  (both streaming and non-streaming).
* **Document-agnostic ingestion** — accepts `.txt` or `.md` files, plain
  text, or JSON payloads. No assumptions about document structure.
* **Two pluggable chunking strategies**:
  * `semantic` (default) — embedding-based, header-aware.
  * `fixed` — deterministic recursive char split.
* **OpenVINO LLM** — Qwen2.5-7B-Instruct int8 by default. Runs on iGPU,
  dGPU, or CPU. First-run export is automatic.
* **GPU-friendly defaults** — KV cache in f16, single inference stream,
  persistent compiled-kernel cache, post-load warmup inference.

## Intel GPU Driver (one-time host setup)

The service runs the LLM on the host GPU (passed into the container via
`/dev/dri`). The driver must be installed on the **host**, not inside the
image.

```bash
# Add Intel GPU apt repo (Ubuntu 24.04 noble packages work on 24.10 too)
wget -qO- https://repositories.intel.com/gpu/intel-graphics.key \
    | sudo gpg --yes --dearmor -o /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
https://repositories.intel.com/gpu/ubuntu noble unified" \
    | sudo tee /etc/apt/sources.list.d/intel-gpu.list

# If a previous install included intel-level-zero-gpu, remove it first.
# Its libigc1 dependency conflicts with the libigc2 used by intel-opencl-icd 25+:
sudo apt-get remove -y intel-level-zero-gpu 2>/dev/null || true

# Install the OpenCL driver (pulls in libigc2/libigdfcl2 automatically).
# intel-level-zero-gpu is NOT needed — OpenVINO GPU uses the OpenCL backend.
sudo apt-get update && sudo apt-get install -y intel-opencl-icd

# Add your user to the GPU groups, then re-login (or run: newgrp render)
sudo usermod -aG video,render "$USER"
```

> **Driver version note:** `intel-opencl-icd` ≤ 24.39 causes `generate()` to
> hang on prompts longer than ~3 000 tokens. Version 25.x+ (installed by the
> commands above) handles long prompts correctly.

See the [Intel GPU Driver Installation Guide](https://dgpu-docs.intel.com/driver/installation.html)
for other distros and platforms.

## Quickstart

### Docker (recommended)

```bash
# (Install GPU driver above first, then re-login)
docker compose up -d --build
curl --noproxy '*' http://127.0.0.1:8020/health
```

The container reuses the host's `/dev/dri` GPU device.

### Host (Python venv)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

For host GPU inference, install the driver first — see the
[Intel GPU Driver section above](#intel-gpu-driver-one-time-host-setup).

Full host instructions: [docs/run-standalone.md](docs/run-standalone.md).

## API at a Glance

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness probe |
| `POST /api/v1/context` | Ingest raw text |
| `POST /api/v1/context/file` | Ingest a `.txt` / `.md` upload (≤ 10 MB) |
| `GET  /api/v1/context/stats` | Vector store + model summary |
| `DELETE /api/v1/context` | Reset the active collection |
| `POST /api/v1/query` | Streaming RAG answer (SSE) |
| `POST /v1/chat/completions` | OpenAI-compatible chat |

Full payload examples: [docs/api.md](docs/api.md).

## Configuration

All settings live in `config.yaml` and can be overridden via overlay YAML
files (`SMART_KIOSK_RAG_CONFIG_OVERRIDE_PATHS`) or environment variables
(`SMART_KIOSK_RAG__SECTION__KEY=...`).

Full reference: [docs/configuration.md](docs/configuration.md).

## Layout

```
rag-service/
├── main.py                       FastAPI app entry
├── pipeline.py                   RAG pipeline (LLM + retrieval + streaming)
├── config.yaml                   Default config (host)
├── config.container.yaml         Container overrides (merged on top)
├── Dockerfile / docker-compose.yml
├── api/                          HTTP routers
├── components/
│   ├── chunker_component.py      semantic / fixed chunkers
│   └── embedding_component.py    sentence-transformers wrapper
├── dto/query_dto.py              Pydantic request / response models
├── utils/
│   ├── config_loader.py          YAML + env config
│   ├── ensure_model.py           OpenVINO export / HF download
│   ├── preload_models.py         GPU warmup at boot
│   └── logger_config.py
├── tests/                        pytest unit tests
├── scripts/                      eval & batch tooling (not in image)
└── docs/                         user-facing docs
```

## Development

```bash
pip install -r requirements.txt
pytest tests/ -q
```
