"""Comprueba si Ollama mantiene DOS modelos cargados a la vez.

Para qué: `OLLAMA_MAX_LOADED_MODELS=2` solo se lee al arrancar el servidor.
Si Ollama no se ha reiniciado desde que se definió la variable, sigue en 1 y
cada cambio de modelo paga entre 16 y 55 s de recarga.

El truco para ser rápido: pide 1 solo token a cada modelo (cargarlo cuesta,
generar no). Después consulta /api/ps, que es la lista real de modelos en RAM.

Uso:
    .venv\\Scripts\\python.exe scripts\\check_ollama_parallel.py
"""

from __future__ import annotations

import sys
import time

import requests

OLLAMA = "http://localhost:11434"
MODELS = ["qwen2.5-coder:7b", "llama3.2:3b"]


def cargados() -> list[str]:
    response = requests.get(f"{OLLAMA}/api/ps", timeout=10)
    response.raise_for_status()
    return [m["name"] for m in response.json().get("models", [])]


def tocar(modelo: str) -> float:
    """Carga el modelo pidiendo un token. Devuelve segundos."""
    started = time.perf_counter()
    response = requests.post(
        f"{OLLAMA}/api/generate",
        json={
            "model": modelo,
            "prompt": "ok",
            "stream": False,
            "keep_alive": "10m",
            "options": {"num_predict": 1, "temperature": 0.0},
        },
        timeout=(10, 300),
    )
    if response.status_code == 404:
        raise RuntimeError(f"Modelo no instalado: {modelo}")
    response.raise_for_status()
    return time.perf_counter() - started


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    try:
        print("En RAM antes de empezar:", cargados() or "(ninguno)")
    except Exception as exc:
        print(f"[X] Ollama no responde: {exc}")
        return 2

    for modelo in MODELS:
        print(f"-> cargando {modelo} ...", flush=True)
        try:
            seconds = tocar(modelo)
        except Exception as exc:
            print(f"   [X] {exc}")
            return 1
        print(f"   {seconds:.1f}s | en RAM ahora: {cargados()}")

    final = cargados()
    print()
    if len(final) >= 2:
        print(f"OK: {len(final)} modelos cargados a la vez -> OLLAMA_MAX_LOADED_MODELS activo.")
        return 0
    print(f"SOLO {len(final)} modelo en RAM -> la variable NO está activa en este servidor.")
    print("Cierra Ollama del todo (bandeja) y vuelve a abrirlo. Comprueba con:")
    print("  Get-Process ollama* | Select Name, StartTime")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
