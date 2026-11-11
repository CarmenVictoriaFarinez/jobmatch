"""Score explicable y tabla de huecos. Determinista: sin LLM.

Por qué sin LLM: el score es la parte que el usuario va a discutir. Si sale de
un modelo, no hay forma de explicar por qué sube o baja, ni de testearlo. Aquí
cada punto tiene una razón rastreable hasta una línea del CV.

Reglas acordadas:
  - must-have pesa 70 %, nice-to-have 30 %.
  - "nivel inicial" declarado en skills -> parcial.
  - evidencia solo en proyectos o perfil -> parcial (no es experiencia laboral).
  - covered = 1 punto, partial = 0.5, missing = 0.

La detección de conceptos usa un léxico de tecnologías con alias ES/EN. No
intenta entender lenguaje natural: intenta no inventar.
"""

from __future__ import annotations

import re

from textutil import normalize

WEIGHTS = {"must": 0.7, "nice": 0.3}
POINTS = {"covered": 1.0, "partial": 0.5, "missing": 0.0}
RANK_TO_STATUS = {2: "covered", 1: "partial", 0: "missing"}

# Cualificador que rebaja una skill a parcial: la persona la declara, pero no
# la demuestra en experiencia.
QUALIFIERS = (
    "nivel inicial",
    "nivel basico",
    "basico",
    "principiante",
    "en curso",
    "familiar",
    "learning",
    "beginner",
)

# Léxico mínimo, ampliable. canonical -> alias (todo se normaliza antes).
# No es una ontología: es la lista de cosas que aparecen en ofertas técnicas.
ALIASES: dict[str, tuple[str, ...]] = {
    "python": ("python",),
    "sql": ("sql", "postgres", "postgresql", "mysql", "sqlite"),
    "json": ("json",),
    "git": ("git", "github", "gitlab"),
    "docker": ("docker", "contenedores", "containers"),
    "kubernetes": ("kubernetes", "k8s", "eks", "aks", "gke"),
    "terraform": ("terraform",),
    "helm": ("helm",),
    "prometheus": ("prometheus",),
    "grafana": ("grafana",),
    "aws": ("aws", "amazon web services"),
    "gcp": ("gcp", "google cloud"),
    "azure": ("azure",),
    "istio": ("istio",),
    "linkerd": ("linkerd",),
    "golang": ("golang", "go"),
    "java": ("java",),
    "javascript": ("javascript", "js"),
    "typescript": ("typescript", "ts"),
    "react": ("react",),
    "node": ("node", "nodejs", "node js"),
    "django": ("django",),
    "flask": ("flask",),
    "fastapi": ("fastapi", "fast api"),
    "streamlit": ("streamlit",),
    "pytest": ("pytest",),
    "testing": ("pruebas", "testing", "test", "tests", "pruebas unitarias", "unit testing"),
    "ci cd": ("ci cd", "ci/cd", "integracion continua", "continuous integration"),
    "linux": ("linux", "bash", "shell"),
    "airflow": ("airflow",),
    "spark": ("spark", "pyspark"),
    "kafka": ("kafka",),
    "redis": ("redis",),
    "mongodb": ("mongodb", "mongo"),
    "rest api": ("api rest", "rest api", "apis rest", "api", "apis"),
    "microservices": ("microservicios", "microservices"),
    "rag": ("rag", "retrieval augmented generation"),
    "embeddings": ("embeddings", "embedding", "vectores"),
    "bm25": ("bm25",),
    "hybrid search": ("busqueda hibrida", "hybrid search"),
    "llm": ("llm", "llms", "large language model"),
    "langgraph": ("langgraph",),
    "langchain": ("langchain",),
    "ollama": ("ollama",),
    "hugging face": ("hugging face", "huggingface"),
    "pytorch": ("pytorch",),
    "tensorflow": ("tensorflow",),
    "machine learning": ("machine learning", "aprendizaje automatico", "ml"),
    "mlops": ("mlops",),
    "pipeline": ("pipeline", "pipelines", "flujo de datos", "data pipeline", "etl"),
    "automation": ("automatizacion", "automatizar", "automation", "automatizaciones"),
    "devops": ("devops", "sre"),
    "observability": ("observabilidad", "observability", "monitorizacion", "monitoring"),
    "networking": ("redes", "networking", "cni", "networkpolicies"),
    "incident response": ("respuesta a incidentes", "incident response", "guardias", "on call", "p1"),
    "security": ("seguridad", "security"),
    "cloud": ("cloud", "nube"),
    "agile": ("agile", "scrum", "kanban"),
    "cka": ("cka", "certificacion kubernetes"),
}

# Alias ordenados de más largo a más corto: "api rest" antes que "api".
_ALIAS_INDEX: list[tuple[str, str]] = sorted(
    ((alias, canonical) for canonical, aliases in ALIASES.items() for alias in aliases),
    key=lambda pair: len(pair[0]),
    reverse=True,
)


def match_profile(cv: dict, jd: dict) -> dict:
    """Compara el CV extraído con la oferta extraída.

    `cv` y `jd` son los dicts de datos (`result["cv"]["data"]`), no los wrappers
    con metadatos. Devuelve score, desglose por requisito y huecos.
    """
    must = [str(item) for item in (jd.get("must_have") or []) if str(item).strip()]
    nice = [str(item) for item in (jd.get("nice_to_have") or []) if str(item).strip()]

    requirements = [_evaluate(text, "must", cv) for text in must]
    requirements += [_evaluate(text, "nice", cv) for text in nice]

    must_items = [r for r in requirements if r["kind"] == "must"]
    nice_items = [r for r in requirements if r["kind"] == "nice"]

    must_score = _mean(must_items)
    nice_score = _mean(nice_items)

    if must_items and nice_items:
        weights = dict(WEIGHTS)
    elif must_items:
        weights = {"must": 1.0, "nice": 0.0}
    elif nice_items:
        weights = {"must": 0.0, "nice": 1.0}
    else:
        weights = {"must": 0.0, "nice": 0.0}

    score = round(100 * (weights["must"] * must_score + weights["nice"] * nice_score))
    score = max(0, min(100, score))

    return {
        "score": score,
        "weights": weights,
        "must_score": round(must_score, 3),
        "nice_score": round(nice_score, 3),
        "counts": {
            "must": _counts(must_items),
            "nice": _counts(nice_items),
        },
        "requirements": requirements,
        "gaps": [r for r in requirements if r["status"] != "covered"],
    }


def _evaluate(text: str, kind: str, cv: dict) -> dict:
    alternatives = _split_alternatives(text)
    best_rank = 0
    evidence: list[dict] = []
    ranked: dict[str, int] = {}
    notes: list[str] = []

    for alternative in alternatives:
        found = _concepts_in(alternative)
        if not found:
            found = _fallback_concepts(alternative, cv)

        ranks: list[int] = []
        for concept in sorted(found):
            rank, concept_evidence = _rank_concept(concept, cv)
            ranks.append(rank)
            ranked[concept] = max(ranked.get(concept, 0), rank)
            evidence.extend(concept_evidence)
            if rank == 1:
                notes.append(_why_partial(concept, concept_evidence))
            elif rank == 0:
                notes.append(f"{concept}: sin evidencia")

        rank = _aggregate(ranks)
        best_rank = max(best_rank, rank)

    status = RANK_TO_STATUS[best_rank]
    return {
        "text": text,
        "kind": kind,
        "status": status,
        "points": POINTS[status],
        "concepts": sorted(ranked),
        "evidence": _dedupe_evidence(evidence),
        "reason": _reason(status, ranked, notes),
        "action": _action(status),
    }


def _split_alternatives(text: str) -> list[str]:
    """'FastAPI o Flask' son dos caminos válidos; 'JSON/SQL' son dos exigencias.

    Solo se separa por ' o ' / ' or '. La barra se deja como conjunción para no
    dar por cubierto JSON/SQL cuando solo hay SQL.
    """
    parts = re.split(r"\s+o\s+|\s+or\s+", text, flags=re.IGNORECASE)
    return [part.strip() for part in parts if part.strip()] or [text]


def _concepts_in(text: str) -> set[str]:
    normalized = f" {normalize(text)} "
    found: set[str] = set()
    for alias, canonical in _ALIAS_INDEX:
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized):
            found.add(canonical)
    return found


def _fallback_concepts(text: str, cv: dict) -> set[str]:
    """Si el léxico no reconoce nada, busca palabras del requisito que sí
    existan en el CV. Así una tecnología desconocida no se da por cubierta."""
    cv_tokens = _cv_tokens(cv)
    return {
        token
        for token in normalize(text).split()
        if len(token) > 4 and token in cv_tokens
    }


def _cv_tokens(cv: dict) -> set[str]:
    parts = [cv.get("summary") or ""]
    parts += [str(skill) for skill in cv.get("skills") or []]
    for job in cv.get("experience") or []:
        parts.append(f"{job.get('company', '')} {job.get('role', '')}")
        parts += [str(bullet) for bullet in job.get("bullets") or []]
    for project in cv.get("projects") or []:
        parts.append(str(project.get("name") or ""))
        parts += [str(bullet) for bullet in project.get("bullets") or []]
    parts += [str(item) for item in cv.get("education") or []]
    return {token for part in parts for token in normalize(part).split() if len(token) > 2}


def _rank_concept(concept: str, cv: dict) -> tuple[int, list[dict]]:
    """2 = demostrado, 1 = declarado/débil, 0 = ausente."""
    aliases = ALIASES.get(concept, (concept,))
    evidence: list[dict] = []

    for job in cv.get("experience") or []:
        for bullet in job.get("bullets") or []:
            if _has_any(bullet, aliases):
                evidence.append({
                    "where": "experience",
                    "term": concept,
                    "quote": f"{job.get('company', '')}: {bullet}",
                })

    for skill in cv.get("skills") or []:
        if _has_any(str(skill), aliases):
            evidence.append({"where": "skills", "term": concept, "quote": str(skill)})

    for project in cv.get("projects") or []:
        for bullet in project.get("bullets") or []:
            if _has_any(bullet, aliases):
                evidence.append({
                    "where": "projects",
                    "term": concept,
                    "quote": f"{project.get('name', '')}: {bullet}",
                })

    summary = cv.get("summary") or ""
    if summary and _has_any(summary, aliases):
        evidence.append({"where": "summary", "term": concept, "quote": summary})

    if any(item["where"] == "experience" for item in evidence):
        return 2, evidence

    clean_skills = [
        item for item in evidence
        if item["where"] == "skills" and not _qualified(item["quote"])
    ]
    if clean_skills:
        return 2, evidence

    if evidence:
        return 1, evidence
    return 0, evidence


def _has_any(text: str, aliases) -> bool:
    normalized = f" {normalize(text)} "
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized)
        for alias in aliases
    )


def _qualified(skill: str) -> bool:
    normalized = normalize(skill)
    return any(qualifier in normalized for qualifier in QUALIFIERS)


def _aggregate(ranks: list[int]) -> int:
    if not ranks:
        return 0
    if all(rank >= 1 for rank in ranks) and any(rank == 2 for rank in ranks):
        return 2
    if any(rank >= 1 for rank in ranks):
        return 1
    return 0


def _mean(items: list[dict]) -> float:
    if not items:
        return 0.0
    return sum(item["points"] for item in items) / len(items)


def _counts(items: list[dict]) -> dict:
    return {
        status: sum(1 for item in items if item["status"] == status)
        for status in ("covered", "partial", "missing")
    }


def _why_partial(concept: str, evidence: list[dict]) -> str:
    if any(item["where"] == "skills" and _qualified(item["quote"]) for item in evidence):
        return f"{concept}: declarado con nivel inicial/básico"
    if any(item["where"] == "projects" for item in evidence):
        return f"{concept}: solo en proyectos"
    if any(item["where"] == "summary" for item in evidence):
        return f"{concept}: solo en el perfil"
    return f"{concept}: cobertura parcial"


def _reason(status: str, ranked: dict[str, int], notes: list[str]) -> str:
    if status == "covered":
        # Solo se nombran los conceptos con evidencia fuerte. En "FastAPI o
        # Flask", Flask no aparece y no debe insinuarse que sí.
        strong = sorted(concept for concept, rank in ranked.items() if rank == 2)
        return "Evidencia en experiencia o skills: " + (", ".join(strong) or "—")
    if status == "partial":
        detail = "; ".join(dict.fromkeys(notes)) or "cobertura incompleta"
        return f"Parcial — {detail}"
    missing = sorted(ranked) or []
    if missing:
        return "Sin evidencia de: " + ", ".join(missing)
    return "No aparece en el CV."


def _action(status: str) -> str:
    if status == "covered":
        return "Mantenlo visible en el CV y la carta."
    if status == "partial":
        return "Refuérzalo con un ejemplo concreto o no lo destaques."
    return "No lo reclames. Si es imprescindible, valora no aplicar."


def _dedupe_evidence(evidence: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for item in evidence:
        key = (item["where"], item["quote"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique
