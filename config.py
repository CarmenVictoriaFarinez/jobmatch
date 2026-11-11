"""Configuración central. Todo por variable de entorno, con default local.

Por qué un archivo aparte: los modelos y la URL son decisiones, no constantes
escondidas en el código. En Fase 4 cambias JOBFIT_OLLAMA_URL por la API en la
nube y el resto del pipeline no se entera.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Servidor local de Ollama (el default que instala Ollama).
OLLAMA_URL = os.environ.get("JOBFIT_OLLAMA_URL", "http://localhost:11434")

# Modelo por tarea, elegido con las pruebas de scripts/probe_ollama.py:
# - CV: objetos anidados -> el 3b rompía el esquema.
# - JD: listas planas -> el 3b las extrae perfectas y va al doble de velocidad.
MODEL_CV = os.environ.get("JOBFIT_MODEL_CV", "qwen2.5-coder:7b")
MODEL_JD = os.environ.get("JOBFIT_MODEL_JD", "llama3.2:3b")

# Mantener el modelo cargado en RAM entre llamadas. Sin esto, cada extracción
# paga 20-50 s de carga desde disco.
KEEP_ALIVE = os.environ.get("JOBFIT_KEEP_ALIVE", "30m")

# Caché de extracciones: mismo texto + mismo prompt + mismo modelo = mismo JSON.
CACHE_DIR = ROOT / "cache"

# Límites de contexto/salida. Los samples caben de sobra; subir num_ctx encarece
# el prompt en CPU.
NUM_CTX = int(os.environ.get("JOBFIT_NUM_CTX", "4096"))
NUM_PREDICT = int(os.environ.get("JOBFIT_NUM_PREDICT", "1400"))
TIMEOUT = int(os.environ.get("JOBFIT_TIMEOUT", "900"))
