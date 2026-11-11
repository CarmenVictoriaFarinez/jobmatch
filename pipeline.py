"""Pipeline de Fase 1: leer -> validar -> extraer (CV y oferta).

No hay agentes todavía. Es una función lineal a propósito: cada paso puede
fallar y devolver un error legible, sin que la UI tenga que saber por qué.

`on_progress(fracción, mensaje)` permite pintar una barra real en Streamlit.
La fracción es una estimación por caracteres generados, no una verdad: el
modelo no sabe cuánto le queda, y fingir precisión sería peor que aproximar.
"""

from __future__ import annotations

from typing import Callable

from config import MODEL_CV, MODEL_JD
from extract import ExtractionError, extract_cv, extract_jd
from readers import ReadError, read_cv_file

MIN_JD_CHARS = 80
ProgressFn = Callable[[float, str], None]

# Caracteres esperados de salida, medidos con los samples. Solo se usan para
# estimar la barra; si el modelo escribe más, la barra se queda en el 95 %.
CV_EXPECTED_CHARS = 1500
JD_EXPECTED_CHARS = 900


def run_pipeline(
    cv_bytes: bytes | None,
    cv_filename: str | None,
    jd_text: str,
    *,
    language: str = "es",
    role: str = "",
    on_progress: ProgressFn | None = None,
) -> dict:
    notify = on_progress or (lambda _fraction, _message: None)

    errors: list[str] = []
    warnings: list[str] = []
    cv_text = ""

    notify(0.02, "Validando entradas…")

    if not cv_bytes or not cv_filename:
        errors.append("Sube un CV en PDF o DOCX.")
    else:
        try:
            cv_text, read_warnings = read_cv_file(cv_bytes, cv_filename)
            warnings.extend(read_warnings)
        except ReadError as exc:
            errors.append(str(exc))

    jd_text = (jd_text or "").strip()
    if len(jd_text) < MIN_JD_CHARS:
        errors.append(
            f"Pega una oferta con al menos {MIN_JD_CHARS} caracteres "
            "(el anuncio completo, no solo el título)."
        )

    if not cv_text:
        errors.append("El CV quedó vacío tras la lectura. Revisa que tenga texto seleccionable.")

    result = {
        "stage": "blocked",
        "language": language,
        "role": role.strip(),
        "cv_filename": cv_filename,
        "cv_text": cv_text,
        "jd_text": jd_text,
        "cv": None,
        "jd": None,
        "match": None,
        "artifacts": None,
        "warnings": warnings,
        "errors": errors,
        "next_step": "corregir entradas",
    }
    if errors:
        return result

    # --- CV -------------------------------------------------------------
    notify(0.05, f"Extrayendo CV con {MODEL_CV} (la primera vez tarda 1–3 min)…")
    try:
        cv_result = extract_cv(
            cv_text,
            on_progress=_bar(notify, start=0.05, span=0.50, expected=CV_EXPECTED_CHARS, label="Extrayendo CV"),
        )
    except ExtractionError as exc:
        _record_error(errors, "CV", exc)
        result["next_step"] = "revisar el CV o probar otro modelo"
        return result

    result["cv"] = cv_result
    notify(
        0.55,
        "CV desde caché" if cv_result["meta"].get("cached") else f"CV listo en {cv_result['meta'].get('seconds')} s",
    )

    # --- Oferta ---------------------------------------------------------
    notify(0.57, f"Extrayendo oferta con {MODEL_JD}…")
    try:
        jd_result = extract_jd(
            jd_text,
            on_progress=_bar(notify, start=0.57, span=0.40, expected=JD_EXPECTED_CHARS, label="Extrayendo oferta"),
        )
    except ExtractionError as exc:
        _record_error(errors, "Oferta", exc)
        result["next_step"] = "revisar la oferta o probar otro modelo"
        return result

    result["jd"] = jd_result
    notify(
        0.97,
        "Oferta desde caché" if jd_result["meta"].get("cached") else f"Oferta lista en {jd_result['meta'].get('seconds')} s",
    )
    result["stage"] = "extracted"
    result["next_step"] = "match y score (días 5–7)"
    notify(1.0, "Listo")
    return result


def _bar(notify: ProgressFn, *, start: float, span: float, expected: int, label: str) -> ProgressFn:
    """Traduce (segundos, caracteres) del modelo a una fracción de la barra."""

    def callback(elapsed: float, chars: int) -> None:
        fraction = start + span * min(chars / expected, 1.0)
        notify(fraction, f"{label}… {elapsed:.0f} s")

    return callback


def _record_error(errors: list[str], task: str, exc: ExtractionError) -> None:
    errors.append(f"{task}: {exc}")
    for detail in exc.errors:
        errors.append(f"  · {detail}")
    if exc.raw_path:
        errors.append(f"  archivo: {exc.raw_path}")
