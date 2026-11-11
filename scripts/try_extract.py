"""Prueba de extracción de punta a punta, sin Streamlit.

Uso (desde la raíz del proyecto):
    .venv\\Scripts\\python.exe scripts\\try_extract.py
    .venv\\Scripts\\python.exe scripts\\try_extract.py --task cv --model llama3.2:3b
    .venv\\Scripts\\python.exe scripts\\try_extract.py --no-cache

La segunda ejecución debe ser instantánea: viene de caché. Si tarda lo mismo,
la clave de caché no está funcionando y hay que mirarlo.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from extract import ExtractionError, extract_cv, extract_jd  # noqa: E402

CV_PATH = ROOT / "samples" / "cv_ana_ruiz.txt"
JD_PATH = ROOT / "samples" / "oferta_alineada.txt"


def progreso(segundos: float, chars: int) -> None:
    marca = int(segundos) // 10
    if marca != progreso.ultima:
        progreso.ultima = marca
        print(f"    ... {segundos:.0f}s, {chars} chars", flush=True)


progreso.ultima = -1  # type: ignore[attr-defined]


def resumen_cv(data: dict) -> None:
    print(f"  name       : {data.get('name')!r}")
    print(f"  headline   : {data.get('headline')!r}")
    print(f"  location   : {data.get('location')!r}")
    print(f"  skills     : {len(data.get('skills') or [])}")
    for job in data.get("experience") or []:
        print(f"  experiencia: {job.get('company')} | {job.get('role')} "
              f"({job.get('start')}-{job.get('end')}) | {len(job.get('bullets') or [])} bullets")
    for project in data.get("projects") or []:
        print(f"  proyecto   : {project.get('name')} | {len(project.get('bullets') or [])} bullets")
    for item in data.get("education") or []:
        print(f"  formación  : {item}")


def resumen_jd(data: dict) -> None:
    print(f"  role        : {data.get('role')!r}")
    print(f"  company     : {data.get('company')!r}")
    print(f"  location    : {data.get('location')!r}")
    print(f"  modality    : {data.get('modality')!r}")
    print(f"  seniority   : {data.get('seniority')!r}")
    print(f"  must_have   : {len(data.get('must_have') or [])}")
    print(f"  nice_to_have: {len(data.get('nice_to_have') or [])}")
    print(f"  respuestas  : {len(data.get('responsibilities') or [])}")


def correr(tarea: str, modelo: str | None, use_cache: bool) -> bool:
    source = (CV_PATH if tarea == "cv" else JD_PATH).read_text(encoding="utf-8")
    extractor = extract_cv if tarea == "cv" else extract_jd
    resumen = resumen_cv if tarea == "cv" else resumen_jd

    print(f"-> {tarea.upper()} | modelo={modelo or 'por defecto'} | caché={'on' if use_cache else 'off'}")
    print("   (en CPU puede tardar 1-3 min la primera vez)", flush=True)
    try:
        result = extractor(source, model=modelo, use_cache=use_cache, on_progress=progreso)
    except ExtractionError as exc:
        print(f"   [X] {exc}")
        for error in exc.errors:
            print(f"       - {error}")
        for warning in exc.audit:
            print(f"       ? {warning}")
        if exc.raw_path:
            print(f"       archivo: {exc.raw_path}")
        return False

    meta = result["meta"]
    origen = "CACHÉ" if meta.get("cached") else f"{meta['seconds']}s | {meta['tps']} tok/s"
    print(f"   OK ({origen})")
    for warning in meta.get("audit") or []:
        print(f"   ? auditoría: {warning}")
    for warning in meta.get("warnings") or []:
        print(f"   ? aviso    : {warning}")
    resumen(result["data"])
    print(f"   guardado en: {meta.get('cache_file')}")
    print()
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Extrae CV y oferta de los samples")
    ap.add_argument("--task", choices=["cv", "jd", "both"], default="both")
    ap.add_argument("--model", help="forzar modelo (por defecto: config.py)")
    ap.add_argument("--no-cache", action="store_true", help="ignorar y no leer caché")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    tareas = ["cv", "jd"] if args.task == "both" else [args.task]
    ok = True
    for tarea in tareas:
        progreso.ultima = -1  # type: ignore[attr-defined]
        ok = correr(tarea, args.model, not args.no_cache) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
