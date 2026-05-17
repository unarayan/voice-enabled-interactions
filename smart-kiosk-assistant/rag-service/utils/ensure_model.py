from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys

from huggingface_hub import snapshot_download

from utils.config_loader import config

logger = logging.getLogger(__name__)

_SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _slugify(model_name: str, suffix: str | None = None) -> str:
    slug = model_name.replace("/", "_")
    return f"{slug}__{suffix}" if suffix else slug


def _resolve_service_path(path: str) -> str:
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(_SERVICE_ROOT, path))


def _llm_export_ready(output_dir: str) -> bool:
    required = (
        "openvino_model.xml",
        "openvino_model.bin",
        "openvino_tokenizer.xml",
        "openvino_tokenizer.bin",
        "openvino_detokenizer.xml",
        "openvino_detokenizer.bin",
    )
    return all(
        os.path.isfile(os.path.join(output_dir, f))
        and os.path.getsize(os.path.join(output_dir, f)) > 0
        for f in required
    )


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

    cli = os.path.join(os.path.dirname(sys.executable), "optimum-cli")
    if not os.path.isfile(cli):
        cli = shutil.which("optimum-cli")
    if not cli:
        raise FileNotFoundError(
            "optimum-cli not found in venv or PATH. "
            "Install with: pip install optimum-intel[openvino]==1.24.0 optimum==1.26.1"
        )

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        cli, "export", "openvino",
        "--model", llm_cfg.hf_id,
        "--weight-format", weight_format,
        "--task", "text-generation-with-past",
        "--trust-remote-code",
        output_dir,
    ]
    logger.info(
        "Exporting %s -> %s  (weight_format=%s)",
        llm_cfg.hf_id, output_dir, weight_format,
    )
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"optimum-cli export failed for {llm_cfg.hf_id} (exit {result.returncode})"
        )

    if not _llm_export_ready(output_dir):
        raise RuntimeError(
            f"Export incomplete for {llm_cfg.hf_id}: "
            f"one or more OpenVINO IR files missing or empty in {output_dir}"
        )

    logger.info("LLM ready at %s", output_dir)
    return output_dir


def ensure_embedding_model(force: bool = False) -> str:
    emb_cfg = config.models.embedding
    output_dir = get_embedding_model_path()

    if not force and os.path.isdir(output_dir) and any(os.scandir(output_dir)):
        logger.info("Using cached embedding model at %s", output_dir)
        return output_dir

    logger.info("Downloading embedding model %s -> %s", emb_cfg.hf_id, output_dir)
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
