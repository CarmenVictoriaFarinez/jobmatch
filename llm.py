"""Cliente mínimo de Ollama. Sin SDK: HTTP y JSON, nada más.

Contrato: `chat_json()` devuelve un dict ya parseado o lanza LLMError.
Todo lo específico de Ollama (URL, streaming, keep_alive) vive aquí, así que
migrar a Gemini/Groq en Fase 4 es reescribir este archivo, no el pipeline.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

import requests

from config import KEEP_ALIVE, NUM_CTX, NUM_PREDICT, OLLAMA_URL, TIMEOUT

ProgressCallback = Callable[[float, int], None]


class LLMError(RuntimeError):
    """Fallo del modelo o de la conexión, con mensaje accionable."""


@dataclass
class LLMResult:
    data: dict
    raw: str
    model: str
    seconds: float
    tokens: int = 0
    tps: float = 0.0
    meta: dict = field(default_factory=dict)


def chat_json(
    prompt: str,
    *,
    model: str,
    schema: dict | None = None,
    temperature: float = 0.0,
    num_ctx: int = NUM_CTX,
    num_predict: int = NUM_PREDICT,
    timeout: int = TIMEOUT,
    keep_alive: str = KEEP_ALIVE,
    on_progress: ProgressCallback | None = None,
) -> LLMResult:
    """Pide al modelo un objeto JSON y lo parsea.

    - `schema`: si se pasa, Ollama usa gramática restringida (structured output)
      y es mucho más difícil que la forma se rompa.
    - `on_progress(segundos, caracteres)`: se llama ~1 vez por segundo mientras
      genera. En CPU una extracción tarda minutos; sin esto parece colgada.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "format": schema if schema is not None else "json",
        "keep_alive": keep_alive,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }

    started = time.perf_counter()
    pieces: list[str] = []
    final: dict = {}
    last_notice = 0.0

    try:
        with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=(10, timeout)) as response:
            if response.status_code == 404:
                raise LLMError(f"Modelo no instalado: {model}. Ejecuta: ollama pull {model}")
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if "error" in event:
                    raise LLMError(f"Ollama: {event['error']}")
                chunk = (event.get("message") or {}).get("content", "")
                if chunk:
                    pieces.append(chunk)
                elapsed = time.perf_counter() - started
                if on_progress and elapsed - last_notice >= 1.0:
                    on_progress(elapsed, sum(map(len, pieces)))
                    last_notice = elapsed
                if event.get("done"):
                    final = event
                    break
    except requests.exceptions.ConnectionError as exc:
        raise LLMError(
            f"No se puede conectar con Ollama en {OLLAMA_URL}. "
            "Abre la app de Ollama o ejecuta `ollama serve`."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise LLMError(f"El modelo {model} superó el timeout de {timeout}s.") from exc

    seconds = time.perf_counter() - started
    raw = "".join(pieces).strip()
    if not raw:
        raise LLMError(f"{model} devolvió una respuesta vacía.")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"{model} no devolvió JSON válido ({exc}). Primeros 300 chars:\n{raw[:300]}"
        ) from exc

    if not isinstance(data, dict):
        raise LLMError(f"{model} devolvió un JSON que no es un objeto: {type(data).__name__}.")

    eval_count = final.get("eval_count") or 0
    eval_ns = final.get("eval_duration") or 0
    return LLMResult(
        data=data,
        raw=raw,
        model=model,
        seconds=seconds,
        tokens=eval_count,
        tps=(eval_count / (eval_ns / 1e9)) if eval_ns else 0.0,
    )


def list_models() -> list[str]:
    """Modelos instalados. Útil para validar la configuración al arrancar."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise LLMError(f"Ollama no responde en {OLLAMA_URL}: {exc}") from exc
    return [m["name"] for m in response.json().get("models", [])]
