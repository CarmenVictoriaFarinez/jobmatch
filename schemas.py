"""Contratos de datos de la Fase 1: esquema + validación + auditoría.

Tres capas distintas, y conviene no confundirlas:
  1. SCHEMA  -> se pasa a Ollama en `format`. La gramática del modelo obliga a
                que la salida tenga esa forma (tipos, claves obligatorias).
  2. validate -> comprueba otra vez en Python. La gramática evita formas raras,
                pero no garantiza contenido útil (p. ej. "company": "").
  3. audit    -> señala valores que no aparecen en el texto original. Es la
                base del Critic de Fase 2: no inventar es el producto.

Las reglas que el esquema no puede expresar (un proyecto no es un empleo)
viven en `rules.py`, no aquí.
"""

from __future__ import annotations

from jsonschema import Draft202012Validator

from textutil import looks_invented, normalize, unsupported_requirement

CV_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "headline", "location", "summary", "skills", "experience", "projects", "education"],
    "properties": {
        "name": {"type": "string"},
        "headline": {
            "type": "string",
            "description": "Titular corto del CV, máximo 100 caracteres. No el párrafo de perfil.",
        },
        "location": {"type": "string"},
        "summary": {"type": "string"},
        "skills": {"type": "array", "items": {"type": "string"}},
        "experience": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["company", "role", "start", "end", "bullets"],
                "properties": {
                    "company": {"type": "string"},
                    "role": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "projects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "bullets"],
                "properties": {
                    "name": {"type": "string"},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "education": {"type": "array", "items": {"type": "string"}},
    },
}

JD_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["role", "company", "location", "modality", "must_have", "nice_to_have", "responsibilities", "seniority"],
    "properties": {
        "role": {"type": "string"},
        "company": {"type": "string"},
        "location": {
            "type": "string",
            "description": "Ciudad o país de la oferta, tal cual. No la modalidad.",
        },
        "modality": {
            "type": "string",
            "description": "remoto, híbrido o presencial, tal como lo diga la oferta. '' si no lo dice.",
        },
        "must_have": {"type": "array", "items": {"type": "string"}},
        "nice_to_have": {"type": "array", "items": {"type": "string"}},
        "responsibilities": {"type": "array", "items": {"type": "string"}},
        "seniority": {"type": "string"},
    },
}


def validate(data: dict, schema: dict) -> list[str]:
    """Errores de forma, en texto legible. Lista vacía = contrato cumplido."""
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    messages = []
    for error in errors:
        where = "/".join(str(p) for p in error.path) or "(raíz)"
        messages.append(f"{where}: {error.message}")
    return messages


def audit_cv(data: dict, source: str) -> list[str]:
    """Valores del CV extraído que no se sostienen con el texto original."""
    source_norm = normalize(source)
    suspicious: list[str] = []

    for index, job in enumerate(data.get("experience") or []):
        for field in ("company", "role"):
            value = job.get(field) or ""
            if value and looks_invented(value, source_norm):
                suspicious.append(f"experience[{index}].{field} = {value!r}")

    for skill in data.get("skills") or []:
        if skill and looks_invented(skill, source_norm):
            suspicious.append(f"skill = {skill!r}")

    # Aviso de forma, no de invención: un headline larguísimo suele ser el
    # párrafo de perfil copiado (fallo observado en llama3.2:3b).
    headline = (data.get("headline") or "").strip()
    if len(headline) > 110:
        suspicious.append(f"headline demasiado largo ({len(headline)} chars): parece el perfil")
    if headline and headline.strip() == (data.get("summary") or "").strip():
        suspicious.append("headline y summary son idénticos")

    return suspicious


def audit_jd(data: dict, source: str) -> list[str]:
    """Requisitos de la oferta sin apoyo en el texto pegado."""
    source_norm = normalize(source)
    suspicious: list[str] = []

    for field in ("role", "company"):
        value = data.get(field) or ""
        if value and looks_invented(value, source_norm):
            suspicious.append(f"{field} = {value!r}")

    for field in ("must_have", "nice_to_have", "responsibilities"):
        for item in data.get(field) or []:
            if item and unsupported_requirement(item, source_norm):
                suspicious.append(f"{field} = {item[:70]!r}")

    return suspicious
