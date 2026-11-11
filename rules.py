"""Reglas semánticas: lo que un JSON Schema no puede expresar.

Un esquema garantiza FORMA. No garantiza que un proyecto no esté disfrazado
de empleo. Ese fue el fallo real de llama3.2:3b: con la gramática obligada a
rellenar company/role/start/end, inventó un puesto llamado "JobMatch".

Estas reglas son deterministas (0 s, 0 €) y corren antes de cachear. Atrapan
lo que la auditoría por subcadena no ve: el valor existe en el CV, pero está
en la sección equivocada.

Criterio de gravedad:
  - ERROR: indica invención de empleo. Se rechaza y no se cachea.
  - AVISO: puede ser legítimo (CV sin fechas). Se muestra para revisar.
"""

from __future__ import annotations

from textutil import normalize

# Roles que, normalizados, son un proyecto y no un puesto de trabajo.
# Lista cerrada a propósito: "Project Manager" NO debe caer aquí.
PROJECT_ROLES = {
    "proyecto propio",
    "proyecto personal",
    "personal project",
    "side project",
}


def cv_semantic_errors(data: dict) -> list[str]:
    """Invenciones que rompen el producto: un proyecto presentado como empleo."""
    errors: list[str] = []

    project_names = {
        normalize(p.get("name") or "").strip()
        for p in (data.get("projects") or [])
        if (p.get("name") or "").strip()
    }
    project_names = {name for name in project_names if len(name) > 3}

    for index, job in enumerate(data.get("experience") or []):
        company = normalize(job.get("company") or "").strip()
        role = normalize(job.get("role") or "").strip()

        if company and company in project_names:
            errors.append(
                f"experience[{index}]: '{job.get('company')}' es un proyecto del CV, "
                "no una empresa. Un proyecto no se declara como empleo."
            )

        if role in PROJECT_ROLES or role.startswith("proyecto "):
            errors.append(
                f"experience[{index}].role = {job.get('role')!r}: describe un proyecto, "
                "no un puesto de trabajo."
            )

    return errors


def cv_semantic_warnings(data: dict) -> list[str]:
    """Señales dudosas que no justifican rechazar la extracción."""
    warnings: list[str] = []

    experience_bullets = [
        normalize(bullet).strip()
        for job in (data.get("experience") or [])
        for bullet in (job.get("bullets") or [])
        if len(bullet.strip()) > 15
    ]

    for index, job in enumerate(data.get("experience") or []):
        company = (job.get("company") or "").strip()
        start = (job.get("start") or "").strip()
        end = (job.get("end") or "").strip()

        if not company:
            warnings.append(f"experience[{index}]: sin empresa.")
        if not start and not end:
            warnings.append(f"experience[{index}] ({company or 'sin empresa'}): sin fechas.")
        if not (job.get("bullets") or []):
            warnings.append(f"experience[{index}] ({company or 'sin empresa'}): sin bullets.")

    for project in data.get("projects") or []:
        for bullet in project.get("bullets") or []:
            normalized = normalize(bullet).strip()
            if len(normalized) <= 15:
                continue
            # Texto repetido en experiencia y proyectos: señal de clasificación doble.
            for other in experience_bullets:
                if normalized in other or other in normalized:
                    warnings.append(
                        f"proyecto '{project.get('name')}': bullet repetido en experiencia "
                        f"({bullet[:60]!r})."
                    )
                    break

    return warnings
