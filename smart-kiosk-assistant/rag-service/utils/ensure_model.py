"""Model download / export utilities for the rag-service.

Export strategy (in order):
  1. ``optimum-cli export openvino`` — verified, GPU-compatible path (uses the
     venv-local binary when available, otherwise PATH).
  2. HF Hub snapshot download of the matching pre-converted OpenVINO repo —
     fallback when optimum-cli is not installed.

Public API (drop-in compatible with the original):
  get_llm_model_path()             -> str
  get_embedding_model_path()       -> str
  ensure_llm_model(force)          -> str
  ensure_embedding_model(force)    -> str
  ensure_model(force)              -> None
  resolve_embedding_model_source() -> str
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys

from huggingface_hub import snapshot_download

from utils.config_loader import config

logger = logging.getLogger(__name__)

_WEIGHT_FORMATS = {"fp32", "fp16", "int8", "int4"}
_SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── path helpers ──────────────────────────────────────────────────────────────

def _slugify(model_name: str, suffix: str | None = None) -> str:
    slug = model_name.replace("/", "_")
    return f"{slug}__{suffix}" if suffix else slug


def _resolve_service_path(path: str) -> str:
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(_SERVICE_ROOT, path))


def _reset_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# ── readiness checks ──────────────────────────────────────────────────────────

def _llm_model_exists(output_dir: str) -> bool:
    for name in ("openvino_model.xml", "openvino_model.bin"):
        p = os.path.join(output_dir, name)
        if not (os.path.isfile(p) and os.path.getsize(p) > 0):
            return False
    return True


def _llm_tokenizer_exists(output_dir: str) -> bool:
    for name in (
        "openvino_tokenizer.xml",
        "openvino_tokenizer.bin",
        "openvino_detokenizer.xml",
        "openvino_detokenizer.bin",
    ):
        p = os.path.join(output_dir, name)
        if not (os.path.isfile(p) and os.path.getsize(p) > 0):
            return False
    return True


def _llm_export_ready(output_dir: str) -> bool:
    return _llm_model_exists(output_dir) and _llm_tokenizer_exists(output_dir)


# ── export logic ──────────────────────────────────────────────────────────────

def _find_optimum_cli() -> str | None:
    """Return path to optimum-cli, preferring the active venv's bin directory."""
    candidate = os.path.join(os.path.dirname(sys.executable), "optimum-cli")
    if os.path.isfile(candidate):
        return candidate
    return shutil.which("optimum-cli")


def _export_via_optimum_cli(
    model_name: str,
    output_dir: str,
    weight_format: str,
    task: str = "text-generation-with-past",
) -> None:
    cli = _find_optimum_cli()
    if cli is None:
        raise FileNotFoundError(
            "optimum-cli not found. "
            "Install it with: pip install optimum-intel[openvino]"
        )
    cmd = [
        cli, "export", "openvino",
        "--model", model_name,
        "--weight-format", weight_format,
        "--task", task,
        "--trust-remote-code",
        output_dir,
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"optimum-cli export failed for {model_name} (exit {result.returncode})"
        )


def _download_preconverted(model_name: str, weight_format: str, output_dir: str) -> None:
    """Download a pre-converted OpenVINO model from HuggingFace Hub.

    Naming convention used by the OpenVINO HF org:
        ``Owner/ModelName`` -> ``OpenVINO/ModelName-{weight_format}-ov``
    """
    ov_repo_id = f"OpenVINO/{model_name.split('/')[-1]}-{weight_format}-ov"
    logger.info(
        "Downloading pre-converted OpenVINO model %s -> %s",
        ov_repo_id,
        output_dir,
    )
    snapshot_download(repo_id=ov_repo_id, local_dir=output_dir)


def _export_llm(model_name: str, output_dir: str, weight_format: str) -> None:
    """Export an LLM to OpenVINO IR, trying optimum-cli first then HF Hub fallback."""
    if weight_format not in _WEIGHT_FORMATS:
        raise ValueError(
            f"Unsupported weight_format: {weight_format!r}. "
            f"Choose from {sorted(_WEIGHT_FORMATS)}."
        )

    _reset_dir(output_dir)

    if _find_optimum_cli():
        logger.info(
            "Exporting %s -> %s  [optimum-cli, weight_format=%s]",
            model_name,
            output_dir,
            weight_format,
        )
        _export_via_optimum_cli(model_name, output_dir, weight_format)
    else:
        logger.warning(
            "optimum-cli not found -- falling back to HF Hub pre-converted download. "
            "Install optimum-intel[openvino] for a local export."
        )
        _download_preconverted(model_name, weight_format, output_dir)

    if not _llm_model_exists(output_dir):
        raise RuntimeError(
            f"Export failed for {model_name}: "
            f"openvino_model.xml / openvino_model.bin missing in {output_dir}"
        )
    if not _llm_tokenizer_exists(output_dir):
        raise RuntimeError(
            f"Export failed for {model_name}: "
            f"OpenVINO tokenizer IR files missing in {output_dir}"
        )


# ── public API ────────────────────────────────────────────────────────────────

def get_llm_model_path() -> str:
    llm_cfg = config.models.llm
    return os.path.join(
        _resolve_service_path(llm_cfg.models_base_path),
        _slugify(llm_cfg.hf_id, getattr(llm_cfg, "weight_format", "int8")),
    )


def get_embedding_model_path() -> str:
    emb_cfg = config.models.embedding
    return os.path.join(
        _resolve_service_path(emb_cfg.models_base_path),
        "sentence_transformers",
        _slugify(emb_cfg.hf_id),
    )


def ensure_llm_model(force: bool = False) -> str:
    llm_cfg = config.models.llm
    output_dir = get_llm_model_path()
    weight_format = getattr(llm_cfg, "weight_format", "int8")

    if not force and _llm_export_ready(output_dir):
        logger.info("Using cached LLM export at %s", output_dir)
        return output_dir

    logger.info(
        "Exporting %s -> OpenVINO IR at %s  (weight_format=%s). "
        "This takes a few minutes on first run.",
        llm_cfg.hf_id,
        output_dir,
        weight_format,
    )
    _export_llm(llm_cfg.hf_id, output_dir, weight_format)
    logger.info("LLM ready at %s", output_dir)
    return output_dir


def ensure_embedding_model(force: bool = False) -> str:
    emb_cfg = config.models.embedding
    output_dir = get_embedding_model_path()

    if not force and os.path.isdir(output_dir) and any(os.scandir(output_dir)):
        logger.info("Using cached embedding model at %s", output_dir)
        return output_dir

    logger.info(
        "Downloading embedding model %s -> %s",
        emb_cfg.hf_id,
        output_dir,
    )
    os.makedirs(output_dir, exist_ok=True)
    snapshot_download(repo_id=emb_cfg.hf_id, local_dir=output_dir)
    return output_dir


def ensure_model(force: bool = False) -> None:
    ensure_llm_model(force=force)
    ensure_embedding_model(force=force)


def resolve_embedding_model_source() -> str:
    output_dir = get_embedding_model_path()
    if os.path.isdir(output_dir) and any(os.scandir(output_dir)):
        return output_dir
    return config.models.embedding.hf_id
