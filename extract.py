"""Extracción de cv.json y jd.json con Ollama, validación y caché.

Flujo de una extracción:
  1. prompt = plantilla de prompts/ + texto del CV/oferta
  2. si ya existe caché para (modelo + prompt) -> devolver sin llamar al modelo
  3. llamar a Ollama con JSON Schema (la gramática obliga a la forma)
  4. validar en Python
  5. auditar contra el texto original (posibles invenciones)
  6. guardar en caché solo si la validación pasa

La caché es lo que hace usable un modelo local lento: el CV se extrae una vez
y se reutiliza para todas las ofertas. Su clave incluye el prompt y el modelo,
así que editar el prompt o cambiar de modelo invalida automáticamente.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from config import CACHE_DIR, MODEL_CV, MODEL_JD
from llm import LLMError, chat_json
from rules import cv_semantic_errors, cv_semantic_warnings
from schemas import CV_SCHEMA, JD_SCHEMA, audit_cv, audit_jd, validate

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


class ExtractionError(RuntimeError):
    """La extracción falló o no cumplió el contrato de datos."""

    def __init__(
        self,
        message: str,
        *,
        errors: list[str] | None = None,
        audit: list[str] | None = None,
        raw_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.errors = errors or []
        self.audit = audit or []
        self.raw_path = raw_path


def extract_cv(
    cv_text: str,
    *,
    model: str | None = None,
    use_cache: bool = True,
    on_progress=None,
) -> dict:
    """Devuelve {"data": {...}, "meta": {...}} a partir del texto del CV."""
    return _extract(
        task="cv",
        text=cv_text,
        model=model or MODEL_CV,
        prompt_name="extract_cv.txt",
        schema=CV_SCHEMA,
        auditor=audit_cv,
        use_cache=use_cache,
        on_progress=on_progress,
    )


def extract_jd(
    jd_text: str,
    *,
    model: str | None = None,
    use_cache: bool = True,
    on_progress=None,
) -> dict:
    """Devuelve {"data": {...}, "meta": {...}} a partir del texto de la oferta."""
    return _extract(
        task="jd",
        text=jd_text,
        model=model or MODEL_JD,
        prompt_name="extract_jd.txt",
        schema=JD_SCHEMA,
        auditor=audit_jd,
        use_cache=use_cache,
        on_progress=on_progress,
    )


def _extract(
    *,
    task: str,
    text: str,
    model: str,
    prompt_name: str,
    schema: dict,
    auditor,
    use_cache: bool,
    on_progress,
) -> dict:
    source = (text or "").strip()
    if len(source) < 20:
        raise ExtractionError(f"El texto de {task} es demasiado corto para extraer nada ({len(source)} chars).")

    template = (PROMPT_DIR / prompt_name).read_text(encoding="utf-8")
    keyword = "cv" if task == "cv" else "jd"
    prompt = template.format(**{keyword: source})

    # Mismo modelo + mismo prompt + mismo contenido = mismo resultado.
    # El espacio se normaliza para que el mismo CV en PDF, DOCX o TXT no se
    # extraiga dos veces por diferencias de líneas en blanco.
    key = _cache_key(model, template, source)
    cache_file = CACHE_DIR / task / f"{key}.json"
    # Compatibilidad con entradas creadas antes de normalizar la clave.
    legacy_file = CACHE_DIR / task / f"{_legacy_key(model, prompt)}.json"

    if use_cache and (cache_file.exists() or legacy_file.exists()):
        read_path = cache_file if cache_file.exists() else legacy_file
        cached = _read_cache(read_path)
        if cached:
            cached_data = cached.get("data") or {}
            shape_errors = validate(cached_data, schema)
            semantic_errors = _semantic_errors(task, cached_data)

            # Si la forma cambió (nuevo campo obligatorio), hay que regenerar.
            # Si solo fallan reglas semánticas, el modelo determinista daría lo
            # mismo: se falla rápido y se deja la caché como evidencia.
            if not shape_errors and semantic_errors:
                raise ExtractionError(
                    f"{task.upper()} cacheado con {model} incumple las reglas actuales. "
                    "Usa --no-cache para forzar una llamada nueva.",
                    errors=semantic_errors,
                    audit=auditor(cached_data, source),
                    raw_path=read_path,
                )

            if not shape_errors and not semantic_errors:
                if read_path != cache_file:
                    # Migración perezosa: la entrada antigua pasa a la clave
                    # normalizada, así el mismo CV en otro formato la reutiliza.
                    migrated = {
                        **cached,
                        "meta": {
                            **cached.get("meta", {}),
                            "cached": False,
                            "cache_file": str(cache_file),
                            "key": key,
                        },
                    }
                    _write_cache(cache_file, migrated)
                cached.setdefault("meta", {})
                cached["meta"]["cached"] = True
                cached["meta"]["cache_file"] = str(cache_file)
                # Se recalculan: las heurísticas pueden haber mejorado.
                cached["meta"]["audit"] = auditor(cached["data"], source)
                cached["meta"]["warnings"] = _semantic_warnings(task, cached["data"])
                return cached

    try:
        result = chat_json(prompt, model=model, schema=schema, on_progress=on_progress)
    except LLMError as exc:
        raise ExtractionError(str(exc)) from exc

    errors = validate(result.data, schema) + _semantic_errors(task, result.data)
    audit = auditor(result.data, source)
    warnings = _semantic_warnings(task, result.data)

    if errors:
        raw_path = _save_failed(task, key, result.raw)
        raise ExtractionError(
            f"{task.upper()} no cumple el contrato con {model}.",
            errors=errors,
            audit=audit + warnings,
            raw_path=raw_path,
        )

    payload = {
        "task": task,
        "data": result.data,
        "meta": {
            "model": model,
            "seconds": round(result.seconds, 1),
            "tokens": result.tokens,
            "tps": round(result.tps, 1),
            "cached": False,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "audit": audit,
            "warnings": warnings,
            "errors": [],
            "cache_file": str(cache_file),
            "key": key,
        },
    }
    _write_cache(cache_file, payload)
    return payload


def _semantic_errors(task: str, data: dict) -> list[str]:
    return cv_semantic_errors(data) if task == "cv" else []


def _semantic_warnings(task: str, data: dict) -> list[str]:
    return cv_semantic_warnings(data) if task == "cv" else []


def _read_cache(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None  # caché corrupta: se ignora y se regenera
    return payload if isinstance(payload, dict) and payload.get("data") else None


def _cache_key(model: str, template: str, source: str) -> str:
    """Clave estable: modelo + plantilla + contenido sin diferencias de formato.

    Normalizar el espacio evita extraer dos veces el mismo CV subido en PDF,
    DOCX y TXT (los lectores no producen exactamente el mismo texto).
    """
    canonical = re.sub(r"\s+", " ", source).strip()
    return hashlib.sha256(f"{model}\x00{template}\x00{canonical}".encode("utf-8")).hexdigest()[:24]


def _legacy_key(model: str, prompt: str) -> str:
    """Clave anterior (modelo + prompt ya formateado). Solo para leer caché vieja."""
    return hashlib.sha256(f"{model}\x00{prompt}".encode("utf-8")).hexdigest()[:24]


def _write_cache(path: Path, payload: dict) -> None:
    """Escritura atómica: si se corta a mitad, no queda un JSON roto."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _save_failed(task: str, key: str, raw: str) -> Path:
    """Guarda la salida inválida para poder depurar el prompt con datos reales."""
    path = CACHE_DIR / "failed" / f"{task}_{key}_{int(time.time())}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw, encoding="utf-8")
    return path
