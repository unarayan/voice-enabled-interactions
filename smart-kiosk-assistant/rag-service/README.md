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

Tested on: **Ubuntu 24.04 LTS (noble) / 24.10 (oracular)** with
**Intel Arc / Meteor Lake / Raptor Lake** iGPU.

### Why the apt repo alone is not enough

The Intel GPU apt repository (`repositories.intel.com/gpu/ubuntu noble`) lags
upstream by several months. As of May 2026 it still ships `intel-opencl-icd
25.18`, while upstream (`intel/compute-runtime` on GitHub) is at `26.18`.
Additionally, the repo version of `intel-level-zero-gpu` depends on `libigc1`,
which **conflicts** with `libigc2` already pulled in by `intel-opencl-icd 25+`.
The procedure below installs directly from the upstream GitHub releases to
avoid both issues.

### Step 1 — Add the Intel GPU apt repo and upgrade available packages

```bash
# Add Intel GPU apt repo (if not already present)
wget -qO- https://repositories.intel.com/gpu/intel-graphics.key \
    | sudo gpg --yes --dearmor \
    -o /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
https://repositories.intel.com/gpu/ubuntu noble unified" \
    | sudo tee /etc/apt/sources.list.d/intel-gpu.list

sudo apt update

# Upgrade Level Zero loader libs and media stack via apt
# (do NOT install intel-level-zero-gpu from apt — see note above)
sudo apt install -y libze1 libze-dev intel-media-va-driver-non-free libvpl2

# Add your user to the GPU groups, then re-login (or: newgrp render)
sudo usermod -aG video,render "$USER"
```

### Step 2 — Install compute runtime 26.18 from GitHub

This installs `intel-opencl-icd 26.18`, the updated Intel Graphics Compiler
(`libigc2`), GMM library, and `libze-intel-gpu1` (the Level Zero GPU ICD —
replaces the old `intel-level-zero-gpu` package name).

```bash
mkdir -p /tmp/neo && cd /tmp/neo

# Intel Graphics Compiler (IGC) v2.34.4
wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.34.4/intel-igc-core-2_2.34.4+21428_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.34.4/intel-igc-opencl-2_2.34.4+21428_amd64.deb

# Compute runtime 26.18.38308.1 (OpenCL ICD + Level Zero GPU ICD + GMM)
wget https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/intel-opencl-icd_26.18.38308.1-0_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/intel-ocloc_26.18.38308.1-0_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/libze-intel-gpu1_26.18.38308.1-0_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/libigdgmm12_22.10.0_amd64.deb

# Verify checksums before installing
wget https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/ww18.sum
sha256sum -c ww18.sum

# Install
sudo dpkg -i *.deb
sudo apt install -f    # resolve any remaining dependencies
```

### Step 3 — Reboot

```bash
sudo reboot
```

A reboot ensures the new kernel (if upgraded) and all driver changes take
effect cleanly.

### Step 4 — Verify

```bash
# OpenVINO should report CPU, GPU, and NPU
python3 -c "import openvino as ov; core = ov.Core(); print(core.available_devices)"
# Expected: ['CPU', 'GPU', 'NPU']

# Confirm driver version
dpkg -l intel-opencl-icd libze-intel-gpu1 | awk '{print $2, $3}'
```

### Troubleshooting: `libigc1` / `libigc2` conflict

If `apt` refuses to install `intel-level-zero-gpu` with a message like
`Conflicts: libigc1 but libigc2 is installed`, **do not try to force-install
the apt package**. It is too old. The GitHub `.deb` packages in Step 2 ship
`libze-intel-gpu1` which is built against `libigc2` and is the correct
replacement.

### NPU driver

The NPU driver (`intel-level-zero-npu`) must be installed separately from
[github.com/intel/linux-npu-driver/releases](https://github.com/intel/linux-npu-driver/releases).
It is not included in the compute-runtime bundle above.

### Driver version reference

| Component | Recommended version | Source |
|---|---|---|
| `intel-opencl-icd` | `26.18.38308.1` | GitHub compute-runtime |
| `libze-intel-gpu1` | `26.18.38308.1` | GitHub compute-runtime |
| `intel-igc-core-2` | `2.34.4` | GitHub IGC |
| `libze1` / `libze-dev` | `1.21.9.0+` | Intel apt repo |
| `intel-level-zero-npu` | latest | GitHub linux-npu-driver |

See also: [Intel GPU Driver Installation Guide](https://dgpu-docs.intel.com/driver/installation.html)

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
