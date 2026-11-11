"""Prueba el score contra una oferta, sin Streamlit.

Usa la caché de extracción: si el CV o la oferta no están extraídos, la
primera ejecución pagará el tiempo del modelo (1–3 min). Después es instantáneo.

Uso:
    .venv\\Scripts\\python.exe scripts\\try_match.py
    .venv\\Scripts\\python.exe scripts\\try_match.py --offer samples\\oferta_desalineada.txt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from extract import ExtractionError, extract_cv, extract_jd  # noqa: E402
from match import match_profile  # noqa: E402

ICONS = {"covered": "OK ", "partial": "~~ ", "missing": "-- "}


def main() -> int:
    ap = argparse.ArgumentParser(description="Score determinista CV vs oferta")
    ap.add_argument("--cv", default=str(ROOT / "samples" / "cv_ana_ruiz.txt"))
    ap.add_argument("--offer", default=str(ROOT / "samples" / "oferta_alineada.txt"))
    ap.add_argument("--json", action="store_true", help="volcar el match completo")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cv_text = pathlib.Path(args.cv).read_text(encoding="utf-8")
    jd_text = pathlib.Path(args.offer).read_text(encoding="utf-8")

    try:
        cv = extract_cv(cv_text)
        jd = extract_jd(jd_text)
    except ExtractionError as exc:
        print(f"[X] {exc}")
        for detail in exc.errors:
            print(f"    - {detail}")
        return 1

    print(f"CV    : {args.cv} | modelo {cv['meta']['model']} | caché={cv['meta']['cached']}")
    print(f"Oferta: {args.offer} | modelo {jd['meta']['model']} | caché={jd['meta']['cached']}")

    match = match_profile(cv["data"], jd["data"])
    if args.json:
        print(json.dumps(match, ensure_ascii=False, indent=2))
        return 0

    print()
    print(f"SCORE: {match['score']}/100")
    print(f"must {match['must_score']:.2f} / nice {match['nice_score']:.2f} "
          f"| pesos {match['weights']['must']:.0%}/{match['weights']['nice']:.0%}")
    print(f"must: {match['counts']['must']}")
    print(f"nice: {match['counts']['nice']}")
    print()
    for item in match["requirements"]:
        print(f"{ICONS[item['status']]}[{item['kind']:4}] {item['text']}")
        print(f"        {item['reason']}")
    print()
    print(f"Huecos: {len(match['gaps'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
