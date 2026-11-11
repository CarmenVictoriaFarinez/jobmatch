import streamlit as st

from config import MODEL_CV, MODEL_JD
from pipeline import run_pipeline

st.set_page_config(page_title="JobMatch", layout="wide")
st.title("JobMatch")
st.caption(
    "Fase 1 · lee tu CV y la oferta, y extrae los datos con un modelo local. "
    "Cada dato sale del texto: los avisos marcan lo que no se sostiene."
)

if "result" not in st.session_state:
    st.session_state.result = None

with st.sidebar:
    st.caption("**Modelos locales (Ollama)**")
    st.code(f"CV:     {MODEL_CV}\nOferta: {MODEL_JD}", language=None)
    st.caption("La primera extracción tarda 1–3 min. Después sale de caché.")

tab_in, tab_an, tab_art = st.tabs(["Entrada", "Análisis", "Artefactos"])


def meta_caption(meta: dict | None) -> str:
    if not meta:
        return ""
    model = meta.get("model", "?")
    if meta.get("cached"):
        return f"`{model}` · desde caché"
    return f"`{model}` · {meta.get('seconds', '?')} s · {meta.get('tps', '?')} tok/s"


def show_warnings(result: dict) -> None:
    for key in ("cv", "jd"):
        item = result.get(key)
        if not item:
            continue
        for message in item["meta"].get("audit") or []:
            st.warning(f"Auditoría: {message}")
        for message in item["meta"].get("warnings") or []:
            st.warning(f"Aviso: {message}")


def render_cv(cv_result: dict) -> None:
    data = cv_result["data"]
    st.caption(meta_caption(cv_result["meta"]))
    st.subheader(data.get("name") or "CV")
    if data.get("headline"):
        st.markdown(f"**{data['headline']}**")
    if data.get("location"):
        st.write(data["location"])
    if data.get("summary"):
        st.write(data["summary"])

    skills = data.get("skills") or []
    if skills:
        st.markdown("**Skills**")
        st.write(" · ".join(skills))

    experience = data.get("experience") or []
    if experience:
        st.markdown("**Experiencia**")
        for job in experience:
            st.markdown(
                f"**{job.get('company', '')}** — {job.get('role', '')}  \n"
                f"_{job.get('start', '')}–{job.get('end', '')}_"
            )
            for bullet in job.get("bullets") or []:
                st.markdown(f"- {bullet}")

    projects = data.get("projects") or []
    if projects:
        st.markdown("**Proyectos**")
        for project in projects:
            st.markdown(f"**{project.get('name', '')}**")
            for bullet in project.get("bullets") or []:
                st.markdown(f"- {bullet}")

    education = data.get("education") or []
    if education:
        st.markdown("**Formación**")
        for item in education:
            st.markdown(f"- {item}")


def render_jd(jd_result: dict) -> None:
    data = jd_result["data"]
    st.caption(meta_caption(jd_result["meta"]))
    st.subheader(data.get("role") or "Oferta")
    if data.get("company"):
        st.write(f"**{data['company']}**")
    chips = [value for value in (data.get("location"), data.get("modality"), data.get("seniority")) if value]
    if chips:
        st.caption(" · ".join(chips))

    blocks = [
        ("Must-have", data.get("must_have")),
        ("Nice-to-have", data.get("nice_to_have")),
        ("Responsabilidades", data.get("responsibilities")),
    ]
    for title, items in blocks:
        if items:
            st.markdown(f"**{title}**")
            for item in items:
                st.markdown(f"- {item}")


def render_match(match: dict) -> None:
    must_total = sum(match["counts"]["must"].values())
    col_score, col_must, col_gaps = st.columns(3)
    col_score.metric("Score", f"{match['score']}/100")
    col_must.metric(
        "Must-have cubiertos",
        f"{match['counts']['must']['covered']}/{must_total}",
        help="Un must-have parcial cuenta 0,5; un hueco, 0.",
    )
    col_gaps.metric("Huecos", len(match["gaps"]))
    st.progress(
        match["score"] / 100,
        text=(
            f"must-have {match['must_score']:.0%} · nice-to-have {match['nice_score']:.0%} "
            f"(pesos {match['weights']['must']:.0%}/{match['weights']['nice']:.0%})"
        ),
    )

    icons = {"covered": "✅ cubierto", "partial": "🟡 parcial", "missing": "❌ hueco"}
    labels = {"must": "Must-have", "nice": "Nice-to-have"}

    def esc(text: str) -> str:
        return text.replace("|", "\\|").replace("\n", " ")

    lines = ["| Tipo | Requisito | Estado | Por qué |", "|---|---|---|---|"]
    for item in match["requirements"]:
        lines.append(
            f"| {labels[item['kind']]} | {esc(item['text'])} | {icons[item['status']]} | {esc(item['reason'])} |"
        )
    # Markdown y no st.dataframe: pandas está bloqueado por la política de
    # Application Control de Windows en esta máquina, y st.dataframe lo importa.
    st.markdown("\n".join(lines))

    if match["gaps"]:
        st.markdown("**Qué falta y por qué**")
        for gap in match["gaps"]:
            with st.expander(f"{icons[gap['status']]} · {gap['text']}"):
                st.write(gap["reason"])
                st.caption(gap["action"])
                for evidence in gap["evidence"][:3]:
                    st.markdown(f"- *{evidence['where']}*: {evidence['quote']}")


with tab_in:
    cv_file = st.file_uploader("Sube tu CV", type=["pdf", "docx"])
    jd_text = st.text_area(
        "Pega la oferta",
        height=240,
        placeholder="Pega aquí el anuncio completo (requisitos, responsabilidades, stack)…",
    )
    col_lang, col_role = st.columns(2)
    with col_lang:
        language = st.selectbox("Idioma de salida", ["es", "en"], index=0)
    with col_role:
        role = st.text_input("Puesto al que aplicas (opcional)")

    analyze = st.button("Analizar", type="primary")

    if analyze:
        cv_bytes = cv_file.getvalue() if cv_file else None
        cv_name = cv_file.name if cv_file else None

        bar = st.progress(0.0, text="Preparando…")

        def on_progress(fraction: float, message: str) -> None:
            bar.progress(min(max(fraction, 0.0), 0.99), text=message)

        st.session_state.result = run_pipeline(
            cv_bytes,
            cv_name,
            jd_text,
            language=language,
            role=role,
            on_progress=on_progress,
        )
        bar.progress(1.0, text="Listo")

    result = st.session_state.result
    if result:
        for error in result["errors"]:
            st.error(error)
        for warning in result["warnings"]:
            st.warning(warning)

        if result["cv_text"]:
            with st.expander(f"Texto extraído del CV ({len(result['cv_text'])} caracteres)"):
                st.text(result["cv_text"])
        if result["jd_text"] and not result["errors"]:
            with st.expander("Oferta (texto pegado)"):
                st.text(result["jd_text"])

with tab_an:
    result = st.session_state.result
    if not result or not (result.get("cv") or result.get("jd")):
        st.info("Primero sube un CV y pega una oferta en la pestaña Entrada.")
    else:
        if result.get("match"):
            render_match(result["match"])
            st.divider()
        show_warnings(result)
        col_cv, col_jd = st.columns(2, gap="large")
        with col_cv:
            if result.get("cv"):
                render_cv(result["cv"])
        with col_jd:
            if result.get("jd"):
                render_jd(result["jd"])

with tab_art:
    result = st.session_state.result
    st.subheader("Artefactos")
    if not result or result["stage"] != "matched":
        st.info(
            "Cinco bullets reescritos y una carta, cada frase anclada a una línea del CV. "
            "Se generan cuando exista el score."
        )
    else:
        st.info(
            "Score listo. Siguiente: Writer produce los bullets y la carta, "
            "y el Critic rechaza cualquier frase sin evidencia en el CV."
        )
