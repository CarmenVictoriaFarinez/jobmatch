"""Utilidades de texto para comparar salidas del LLM con el texto fuente.

Son heurísticas, no verdad absoluta: sirven para detectar alucinaciones
gruesas (una empresa que no existe) y para priorizar qué revisar a mano.
Se usan en `schemas.py` (auditoría) y en scripts/probe_ollama.py.
"""

from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    """Minúsculas, sin acentos y sin puntuación. Así "NorteDatos SL" y
    "nortedatos" se comparan igual, sin falsos negativos por tildes."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())


def looks_invented(value: str, source_norm: str) -> bool:
    """¿El valor aparece en el texto original?

    Acepta coincidencia parcial por tokens ("NorteDatos SL" vs "NorteDatos")
    porque los modelos reformulan nombres. Si no comparte ni las dos primeras
    palabras significativas, lo marcamos como sospechoso.
    """
    value_norm = normalize(value).strip()
    if not value_norm:
        return False
    if value_norm in source_norm:
        return False
    tokens = [t for t in value_norm.split() if len(t) > 3]
    return not (tokens and all(t in source_norm for t in tokens[:2]))


def unsupported_requirement(item: str, source_norm: str) -> bool:
    """Para requisitos largos no exigimos copia literal (se reformulan).
    Pedimos al menos dos palabras significativas presentes en la oferta."""
    words = [t for t in normalize(item).split() if len(t) > 4]
    if not words:
        return False
    return sum(1 for t in words if t in source_norm) < 2
