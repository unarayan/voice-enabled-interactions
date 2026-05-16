# Run Without Docker

Use this path when you want to run the service directly with Python on the host.

## 1 — Intel GPU Driver Setup (required for GPU inference)

Install the Intel GPU driver stack from Intel's official apt repository:

> **[Intel GPU Driver Installation Guide](https://dgpu-docs.intel.com/driver/installation.html)**

The quick path on Ubuntu 24.04:

```bash
wget -qO- https://repositories.intel.com/gpu/intel-graphics.key \
    | sudo gpg --yes --dearmor -o /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified" \
    | sudo tee /etc/apt/sources.list.d/intel-gpu.list
# If a previous install included intel-level-zero-gpu, remove it first — its
# libigc1 dependency conflicts with the libigc2 used by intel-opencl-icd 25+:
# sudo apt-get remove -y intel-level-zero-gpu
sudo apt-get update && sudo apt-get install -y intel-opencl-icd
# intel-opencl-icd pulls in libigc2/libigdfcl2 automatically;
# intel-level-zero-gpu is not needed — OpenVINO GPU uses the OpenCL backend.
```

Then add your user to the GPU access groups and re-login:

```bash
sudo usermod -aG video,render $USER
# log out and back in (or: newgrp render in the current shell)
```

> **Driver version note:** Driver 24.39 and older cause `generate()` to hang on prompts above ~3 000 tokens.
> Driver **26.18+** (installed by the commands above) supports 5 000-token prompts correctly.

## 2 — Python Setup

From the `rag-service/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3 — Config

- Start from `config.yaml`.
- Use `SMART_KIOSK_RAG_CONFIG_OVERRIDE_PATHS` for one or more YAML override files.
- Use `SMART_KIOSK_RAG__...` environment variables for targeted overrides.
- For Intel GPU use, set `models.llm.device: GPU` (already the default).

## 4 — Start

```bash
source .venv/bin/activate
python main.py
```

Default bind address:

- host: `0.0.0.0`
- port: `8020`

Equivalent `uvicorn` command:

```bash
uvicorn main:app --host 0.0.0.0 --port 8020
```

## Verify

```bash
curl --noproxy '*' http://127.0.0.1:8020/health
```

## Notes

- Model bootstrap runs on startup through `utils/ensure_model.py`.
- LLM and embedding assets are cached under `models/` by default.
- Ingested vectors and Chroma files are stored under `storage/vector_db/`.
- The GPU driver install script is idempotent — re-running it upgrades to the latest version.
