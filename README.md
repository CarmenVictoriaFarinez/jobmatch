# JobMatch

Workspace para adaptar un CV a una oferta: score explicable, huecos y
borrador de carta **anclado al CV**. No es un chatbot.

Estado actual: la UI lee PDF/DOCX, extrae CV y oferta a JSON, calcula un
**score explicable** con tabla de huecos y lo muestra en **Análisis**.
Todavía no hay bullets, carta ni Critic.

## Arranque local (Windows, PowerShell)

La `.venv` ya existe y apunta a Python 3.14.

```powershell
# 1. Instalar dependencias (una vez, o cuando cambie requirements.txt)
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Lanzar la app
.venv\Scripts\python.exe -m streamlit run app.py
```

Abre `http://localhost:8501`. Para parar: `Ctrl+C`.

Si hay que recrear el entorno desde cero:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Probar la UI (Fase 1)

1. Pestaña **Entrada**.
2. Sube `samples/cv_ana_ruiz.docx` (el `.txt` no vale: el uploader pide PDF/DOCX).
3. Pega el contenido de `samples/oferta_alineada.txt`.
4. Pulsa **Analizar**. Verás la barra de progreso y el modelo en uso.
   - La primera vez con un CV nuevo: 1–3 min. Después sale de caché.
5. Pestaña **Análisis**: score 0–100, desglose must-have / nice-to-have, tabla
   de requisitos y "qué falta y por qué". Debajo, CV y oferta estructurados,
   con avisos si algo no se sostiene con el texto original.
6. Pestaña **Artefactos**: aviso de que el Writer llega después del score.

Comprobación sin navegador:

```powershell
.venv\Scripts\python.exe -c "from pipeline import run_pipeline; r=run_pipeline(open('samples/cv_ana_ruiz.docx','rb').read(),'cv_ana_ruiz.docx',open('samples/oferta_alineada.txt',encoding='utf-8').read()); print(r['stage'], r['match']['score'], r['errors'])"
```

Debe imprimir `matched 76 []`.

## Extracción con Ollama

Requiere Ollama arrancado (`ollama list` debe responder en `localhost:11434`).

```powershell
# Extrae CV y oferta de los samples
.venv\Scripts\python.exe scripts\try_extract.py

# Solo el CV, con otro modelo (útil para comparar)
.venv\Scripts\python.exe scripts\try_extract.py --task cv --model llama3.2:3b

# Ignorar la caché y forzar una llamada nueva
.venv\Scripts\python.exe scripts\try_extract.py --no-cache
```

La **segunda ejecución es instantánea**: viene de `cache/`. La clave incluye
modelo + plantilla + contenido normalizado, así que editar un prompt o cambiar
de modelo invalida solo lo afectado, y el mismo CV en PDF, DOCX o TXT reutiliza
la misma extracción. Las entradas antiguas se migran solas al leerlas.

### Modelos: medidos en esta máquina (CPU, sin GPU)

| Tarea | Modelo | Tiempo | Calidad observada |
|---|---|---|---|
| Oferta | `llama3.2:3b` | 46–62 s | correcta, con `location` y `modality` |
| CV | `qwen2.5-coder:7b` | 166–185 s | estructura correcta |
| CV | `llama3.2:3b` | ~95 s | **rompe la semántica**: mete el proyecto en `experience` |

Decisión: `qwen2.5-coder:7b` para CV, `llama3.2:3b` para oferta. Se cambia por
variable de entorno sin tocar código:

```powershell
$env:JOBFIT_MODEL_CV = "llama3.2:3b"
```

Otras variables: `JOBFIT_MODEL_JD`, `JOBFIT_OLLAMA_URL`, `JOBFIT_KEEP_ALIVE`,
`JOBFIT_NUM_CTX`, `JOBFIT_NUM_PREDICT`, `JOBFIT_TIMEOUT`.

### Cómo se protege de alucinaciones

Cinco capas, en este orden:

1. **JSON Schema** pasado a Ollama en `format`: la gramática restringe la forma.
2. **Validación en Python** (`schemas.validate`): tipos, claves, sin campos extra.
3. **Reglas semánticas** (`rules.py`): un proyecto del CV no puede aparecer como
   empresa en `experience`; un rol tipo "Proyecto propio" se rechaza. Esto
   atrapa invenciones que el esquema no ve porque el texto sí existe en el CV.
4. **Auditoría contra el texto original** (`schemas.audit_cv` / `audit_jd`):
   marca empresas, skills o requisitos que no aparecen en la fuente.
5. **Caché revalidada**: antes de servir una entrada se vuelve a validar con las
   reglas actuales. Cambiar el contrato invalida lo viejo sin borrar nada.

Errores vs avisos: un error (esquema o regla semántica) rechaza la extracción y
guarda el crudo en `cache/failed/`. Un aviso (posible invención, CV sin fechas)
se muestra pero no bloquea. La auditoría son heurísticas: avisan para revisar.

### Score y huecos (determinista, sin LLM)

`match.py` compara los dos JSON y da un score 0–100 con razones rastreables.
No usa el modelo: un score que no se puede explicar no sirve para decidir.

| Estado | Puntos | Cuándo |
|---|---|---|
| `covered` | 1,0 | aparece en experiencia laboral, o en skills sin cualificador |
| `partial` | 0,5 | solo en proyectos o perfil, o skill con "nivel inicial" |
| `missing` | 0,0 | no aparece en ninguna parte del CV |

Pesos: **70 % must-have / 30 % nice-to-have**. Si una lista está vacía, el peso
pasa a la otra. Los requisitos se parten por " o " (FastAPI o Flask = camino
válido alternativo), pero no por "/" (JSON/SQL = dos exigencias).

Resultado con los samples:

| Oferta | Score | Detalle |
|---|---|---|
| Alineada (Python/IA) | **76** | 2 must cubiertos, 2 parciales, 0 huecos |
| Desalineada (SRE/Kubernetes) | **0** | 7 requisitos sin evidencia |

Límite honesto: el léxico de `match.py` es una lista, no comprensión del
lenguaje. Reconoce tecnologías y alias ES/EN; una herramienta rara caerá al
modo "solo palabras que aparezcan en el CV". Ampliar `ALIASES` es barato y no
requiere tocar el resto.

```powershell
.venv\Scripts\python.exe scripts\try_match.py
.venv\Scripts\python.exe scripts\try_match.py --offer samples\oferta_desalineada.txt
```

### Rendimiento: evitar el cambio de modelo

Ollama solo mantiene un modelo en RAM a la vez en CPU. Pasar de 3b a 7b cuesta
entre 16 y 55 s antes del primer carácter. Con 19 GB de RAM caben los dos:

```powershell
setx OLLAMA_MAX_LOADED_MODELS 2   # ya aplicado en esta máquina
```

Requiere reiniciar Ollama (salir de la bandeja y volver a abrirlo). Se comprueba
con `ollama ps` mientras corre una extracción: deben aparecer dos modelos.

### Banco de pruebas de modelos

```powershell
.venv\Scripts\python.exe scripts\probe_ollama.py --task cv --models llama3.2:3b
```

Mide JSON válido, campos que faltan, posibles inventos y tokens/s. Salidas en
`scripts/out/`.

## Estructura

| Archivo | Rol | Estado |
|---|---|---|
| `app.py` | 3 pantallas + barra de progreso | funciona (lectura + extracción) |
| `pipeline.py` | leer → validar → extraer | funciona |
| `readers.py` | PDF/DOCX → texto | funciona, sin OCR |
| `config.py` | modelos, URL y límites por entorno | funciona |
| `llm.py` | cliente HTTP de Ollama | funciona |
| `schemas.py` | contratos JSON + validación + auditoría | funciona |
| `rules.py` | reglas semánticas (proyecto ≠ empleo) | funciona |
| `textutil.py` | normalización y comparación con la fuente | funciona |
| `extract.py` | CV/oferta → JSON, con caché | funciona |
| `match.py` | score + tabla de huecos (sin LLM) | funciona |
| `write.py` | 5 bullets + carta | `NotImplementedError` |
| `prompts/` | plantillas de extracción | CV y oferta listas |
| `scripts/` | diagnóstico y pruebas manuales | — |

## Qué no hace aún

Bullets, carta, Critic, RAG/BM25/embeddings, LangGraph, historial SQLite,
export DOCX y deploy. El PDF imagen-sin-texto no se OCR.

## Nota de entorno (Windows)

Esta máquina tiene Application Control activo y **bloquea una DLL de pandas**:

```
ImportError: DLL load failed while importing ops:
Una directiva de Control de aplicaciones bloqueó este archivo.
```

`st.dataframe` importa pandas y falla, así que la tabla de requisitos se pinta
en Markdown. Si añades gráficos o tablas interactivas, comprueba antes que
pandas importa; si no, usa alternativas sin pandas.

## Privacidad

`samples/` es sintético y con Ollama el CV no sale de tu máquina. No subas
CVs reales a una demo pública: en Fase 4 se documentará el aviso de PII.
