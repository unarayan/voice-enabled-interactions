# Run in Docker

The container uses the host's `/dev/dri` GPU device. The Intel GPU driver
must be installed on the **host** (not inside the image).

## Prerequisites

* Docker 24+ with the `docker compose` plugin
* An Intel GPU exposed at `/dev/dri/renderD128`
* Intel GPU compute-runtime **26.18+** installed on the host
  (see the [Intel GPU Driver Installation Guide](https://dgpu-docs.intel.com/driver/installation.html))
* The host user in the `video` and `render` groups:
  ```bash
  sudo usermod -aG video,render "$USER"
  newgrp render   # or log out and back in
  ```

### One-time driver install on the host (Ubuntu 24.04)

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

> Driver versions ≤ 24.39 hang on prompts > ~3 k tokens. Driver **26.18+**
> handles 5 k-token prompts cleanly on Arrow Lake iGPU.

## Start

From `rag-service/`:

```bash
docker compose up -d --build
```

Published endpoint: `http://127.0.0.1:8020`.

## Verify

```bash
curl --noproxy '*' http://127.0.0.1:8020/health
```

## Notes

* `/dev/dri` is mounted and both `video` + `render` groups are added so
  OpenVINO GPU is reachable from inside the container.
* `config.container.yaml` is mounted read-only and merged on top of
  `config.yaml`.
* Model cache, vector storage, and the HF cache are persisted through
  bind mounts (`models/`, `storage/`, `.cache/huggingface/`).
* If GPU is unavailable inside the container, confirm
  `ls -l /dev/dri/renderD128` is readable by your user, the host driver
  is installed, and the `video`/`render` groups are set.
