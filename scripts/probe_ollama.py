"""
Banco de pruebas de Ollama para JobMatch (Fase 1).

No forma parte de la app: sirve para decidir CON DATOS si tus modelos
locales sirven para extraer JSON de un CV y de una oferta, y para medir
cuánto tardan en tu máquina.

Qué comprueba por cada modelo:
  1. Si devuelve JSON parseable (format="json").
  2. Si trae los campos obligatorios del esquema.
  3. Si "inventa": empresas/skills que no aparecen literalmente en el texto.
  4. Velocidad: segundos totales y tokens/s.

Uso (desde la raíz del proyecto, PowerShell):
    .venv\\Scripts\\python.exe scripts\\probe_ollama.py

Opciones útiles:
    --task cv                 solo el CV (más rápido para iterar)
    --models llama3.2:3b      solo un modelo
    --num-ctx 4096            contexto (los samples caben de sobra)
    --timeout 600             segundos máximos por petición

Salidas en scripts/out/: el JSON crudo y el parseado de cada prueba.
Si el JSON no parsea, se guarda igual para que puedas ver QUÉ falló.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

import requests

OLLAMA = "http://localhost:11434"
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "scripts" / "out"

# El script debe poder correr suelto con `python scripts/probe_ollama.py`.
sys.path.insert(0, str(ROOT))

from textutil import looks_invented, normalize, unsupported_requirement  # noqa: E402

CV_PATH = ROOT / "samples" / "cv_ana_ruiz.txt"
JD_PATH = ROOT / "samples" / "oferta_alineada.txt"

# Los esquemas se declaran como texto en el prompt (no como JSON Schema)
# a propósito: así medimos si el modelo respeta la forma por sí solo.
# Si un modelo falla aquí, un JSON Schema estricto en `format` lo taparía
# y no sabríamos si el problema era el prompt o el modelo.
SCHEMA_CV = """{
  "name": str,
  "headline": str,
  "location": str,
  "summary": str,
  "skills": [str],
  "experience": [{"company": str, "role": str, "start": str, "end": str, "bullets": [str]}],
  "projects": [{"name": str, "bullets": [str]}],
  "education": [str]
}"""

SCHEMA_JD = """{
  "role": str,
  "company": str,
  "must_have": [str],
  "nice_to_have": [str],
  "responsibilities": [str],
  "seniority": str
}"""

PROMPT_CV = """Extrae el CV a JSON valido. Devuelve SOLO el objeto JSON, sin markdown.
Esquema exacto (todas las claves obligatorias):
{schema}

Reglas:
- Copia literal del CV. No inventes empresas, fechas ni skills.
- Si un campo no aparece, usa "" o [].
- "skills" solo lo que esté en la sección SKILLS más tecnologías citadas en experiencia.

CV:
{text}
"""

PROMPT_JD = """Extrae la oferta a JSON valido. Devuelve SOLO el objeto JSON, sin markdown.
Esquema exacto (todas las claves obligatorias):
{schema}

Reglas:
- Copia literal de la oferta, sin parafrasear requisitos.
- Si un campo no aparece, usa "" o [].

OFERTA:
{text}
"""

REQUIRED = {
    "cv": ["name", "headline", "location", "summary", "skills", "experience", "projects", "education"],
    "jd": ["role", "company", "must_have", "nice_to_have", "responsibilities", "seniority"],
}


def listar_modelos() -> list[str]:
    r = requests.get(f"{OLLAMA}/api/tags", timeout=10)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


def generar(modelo: str, prompt: str, *, num_ctx: int, num_predict: int, timeout: int) -> dict:
    """Llama a /api/chat en streaming y devuelve texto + métricas.

    Streaming no es capricho: en CPU una extracción tarda minutos y sin
    progreso parece colgada. Además así puedes cortar con Ctrl+C sin esperar.
    """
    payload = {
        "model": modelo,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "format": "json",
        "options": {
            "temperature": 0.0,   # extracción = reproducibilidad, no creatividad
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    inicio = time.perf_counter()
    trozos: list[str] = []
    stats: dict = {}
    ultimo_aviso = 0.0

    with requests.post(f"{OLLAMA}/api/chat", json=payload, stream=True, timeout=(10, timeout)) as r:
        r.raise_for_status()
        for linea in r.iter_lines():
            if not linea:
                continue
            evento = json.loads(linea)
            if "error" in evento:
                raise RuntimeError(evento["error"])
            pieza = (evento.get("message") or {}).get("content", "")
            if pieza:
                trozos.append(pieza)
            transcurrido = time.perf_counter() - inicio
            if transcurrido - ultimo_aviso >= 5:
                print(f"    ... {transcurrido:.0f}s, {sum(map(len, trozos))} chars", flush=True)
                ultimo_aviso = transcurrido
            if evento.get("done"):
                stats = evento
                break

    total = time.perf_counter() - inicio
    eval_count = stats.get("eval_count") or 0
    eval_ns = stats.get("eval_duration") or 0
    tps = eval_count / (eval_ns / 1e9) if eval_ns else 0.0
    return {
        "texto": "".join(trozos).strip(),
        "segundos": total,
        "tokens": eval_count,
        "tps": tps,
    }


def evaluar(tarea: str, modelo: str, respuesta: dict, fuente: str) -> dict:
    """Convierte la respuesta en un veredicto legible + los fallos concretos."""
    resultado = {
        "tarea": tarea,
        "modelo": modelo,
        "segundos": respuesta["segundos"],
        "tps": respuesta["tps"],
        "json_ok": False,
        "faltan": [],
        "inventados": [],
        "datos": None,
    }

    texto = respuesta["texto"]
    try:
        datos = json.loads(texto)
    except json.JSONDecodeError as exc:
        resultado["error"] = f"JSON inválido: {exc}"
        resultado["raw"] = texto
        return resultado

    resultado["json_ok"] = True
    resultado["datos"] = datos
    resultado["faltan"] = [c for c in REQUIRED[tarea] if c not in datos]

    fuente_norm = normalize(fuente)
    sospechosos: list[str] = []
    if tarea == "cv":
        for exp in datos.get("experience") or []:
            if isinstance(exp, dict):
                for clave in ("company", "role"):
                    if exp.get(clave) and looks_invented(exp[clave], fuente_norm):
                        sospechosos.append(f"{clave}={exp[clave]}")
        for skill in datos.get("skills") or []:
            if isinstance(skill, str) and looks_invented(skill, fuente_norm):
                sospechosos.append(f"skill={skill}")
    else:
        for clave in ("role", "company"):
            if datos.get(clave) and looks_invented(datos[clave], fuente_norm):
                sospechosos.append(f"{clave}={datos[clave]}")
        for item in (datos.get("must_have") or []) + (datos.get("nice_to_have") or []):
            if isinstance(item, str) and unsupported_requirement(item, fuente_norm):
                sospechosos.append(f"req={item[:60]}")

    resultado["inventados"] = sospechosos
    return resultado


def guardar(tarea: str, modelo: str, respuesta: dict, resultado: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seguro = re.sub(r"[^a-zA-Z0-9._-]+", "_", modelo)
    (OUT / f"{tarea}__{seguro}.raw.txt").write_text(respuesta["texto"], encoding="utf-8")
    if resultado["datos"] is not None:
        (OUT / f"{tarea}__{seguro}.json").write_text(
            json.dumps(resultado["datos"], ensure_ascii=False, indent=2), encoding="utf-8"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Prueba extracción JSON con modelos de Ollama")
    ap.add_argument("--models", nargs="*", help="modelos a probar (por defecto, todos los instalados)")
    ap.add_argument("--task", choices=["cv", "jd", "both"], default="both")
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--num-predict", type=int, default=1200)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    # Consola de Windows en cp1252: sin esto, los acentos rompen el print.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    try:
        instalados = listar_modelos()
    except Exception as exc:
        print(f"[X] Ollama no responde en {OLLAMA}: {exc}")
        print("    Abre Ollama (o `ollama serve`) y vuelve a intentarlo.")
        return 2
    if not instalados:
        print("[X] No hay modelos instalados: `ollama pull llama3.2:3b`")
        return 2

    modelos = args.models or instalados
    desconocidos = [m for m in modelos if m not in instalados]
    if desconocidos:
        print(f"[!] No instalados (se omiten): {', '.join(desconocidos)}")
        modelos = [m for m in modelos if m in instalados]
    if not modelos:
        print(f"[X] Nada que probar. Instalados: {', '.join(instalados)}")
        return 2

    tareas = ["cv", "jd"] if args.task == "both" else [args.task]
    fuentes = {
        "cv": CV_PATH.read_text(encoding="utf-8"),
        "jd": JD_PATH.read_text(encoding="utf-8"),
    }
    prompts = {
        "cv": PROMPT_CV.format(schema=SCHEMA_CV, text=fuentes["cv"]),
        "jd": PROMPT_JD.format(schema=SCHEMA_JD, text=fuentes["jd"]),
    }

    print(f"Modelos: {', '.join(modelos)}")
    print(f"Tareas : {', '.join(tareas)} | ctx={args.num_ctx} | timeout={args.timeout}s\n")

    resultados: list[dict] = []
    for modelo in modelos:
        for tarea in tareas:
            print(f"-> {modelo} | {tarea.upper()} ...", flush=True)
            try:
                respuesta = generar(
                    modelo, prompts[tarea],
                    num_ctx=args.num_ctx,
                    num_predict=args.num_predict,
                    timeout=args.timeout,
                )
            except Exception as exc:
                print(f"   [X] Falló la llamada: {exc}\n")
                resultados.append({"tarea": tarea, "modelo": modelo, "error": str(exc)})
                continue

            resultado = evaluar(tarea, modelo, respuesta, fuentes[tarea])
            guardar(tarea, modelo, respuesta, resultado)
            resultados.append(resultado)

            print(f"   {respuesta['segundos']:.1f}s | {respuesta['tps']:.1f} tok/s | "
                  f"JSON={'OK' if resultado['json_ok'] else 'MAL'}")
            if resultado["faltan"]:
                print(f"   faltan: {', '.join(resultado['faltan'])}")
            if resultado["inventados"]:
                print(f"   sospechosos: {'; '.join(resultado['inventados'][:6])}")
            print()

    # Resumen final: lo que de verdad importa para decidir.
    print("=" * 68)
    print(f"{'tarea':4} {'modelo':22} {'seg':>6} {'tok/s':>6} {'JSON':>5} {'inventa':>8}")
    print("-" * 68)
    for r in resultados:
        if "segundos" not in r:
            print(f"{r['tarea']:4} {r['modelo']:22} {'-':>6} {'-':>6} {'-':>5} {'ERROR':>8}")
            continue
        print(
            f"{r['tarea']:4} {r['modelo']:22} {r['segundos']:>6.1f} {r['tps']:>6.1f} "
            f"{('si' if r['json_ok'] else 'no'):>5} {len(r['inventados']):>8}"
        )
    print("=" * 68)
    print(f"Salidas en: {OUT}")

    fallos = [r for r in resultados if not r.get("json_ok")]
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
