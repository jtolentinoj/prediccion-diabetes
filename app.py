from base64 import b64encode
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.preparar_datos import VARIABLES


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "modelos" / "modelo_diabetes.joblib"
RESULTS = ROOT / "resultados"
ASSETS = ROOT / "assets"

st.set_page_config(
    page_title="Predicción de diabetes | UPAO",
    page_icon="🧪",
    layout="wide",
)

ICONS = {
    "flask": '<path d="M8 3h8M9 3v7L4 19a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2l-5-9V3M7 15h10"/><path d="M10 18h.01m4-1h.01"/>',
    "user": '<circle cx="12" cy="7" r="4"/><path d="M5 21v-2a7 7 0 0 1 14 0v2"/>',
    "drop": '<path d="M12 2C9 7 5 11 5 15a7 7 0 0 0 14 0c0-4-4-8-7-13Z"/><path d="M15 16a3 3 0 0 1-2 3"/>',
    "heart": '<path d="M20 5a5 5 0 0 0-8 1 5 5 0 0 0-8-1c-5 5 2 11 8 16 6-5 13-11 8-16Z"/><path d="M3 12h5l2-4 3 8 2-4h6"/>',
    "waves": '<path d="M2 6c3-4 5 4 8 0s5 4 8 0c2-2 3-1 4 0M2 12c3-4 5 4 8 0s5 4 8 0c2-2 3-1 4 0M2 18c3-4 5 4 8 0s5 4 8 0c2-2 3-1 4 0"/>',
    "syringe": '<path d="m17 3 4 4m-2-2-3 3m-4-3 7 7M13 7l-8 8v4h4l8-8M5 19l-3 3m6-10 3 3m0-6 3 3"/>',
    "scale": '<rect x="3" y="3" width="18" height="18" rx="3"/><path d="M7 10a6 6 0 0 1 10 0l-5 4Zm5 4 2-4"/>',
    "users": '<circle cx="9" cy="7" r="3"/><path d="M3 21v-3a6 6 0 0 1 12 0v3m2-17a3 3 0 0 1 0 6m1 3a5 5 0 0 1 3 5v3"/>',
    "calendar": '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M7 3v4m10-4v4M3 10h18m-14 4h1m3 0h1m3 0h1m-9 3h1m3 0h1"/>',
    "chip": '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4"/>',
}


def icon(name):
    return (
        '<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="1.65" stroke-linecap="round" '
        'stroke-linejoin="round">' + ICONS[name] + "</svg>"
    )


reference = ASSETS / "referencia_laboratorio.png"
hero_art = ""
if reference.is_file():
    encoded = b64encode(reference.read_bytes()).decode("ascii")
    hero_art = f' style="background-image:url(data:image/png;base64,{encoded})"'

estilos = (ASSETS / "laboratorio.css").read_text(encoding="utf-8")
st.markdown(f"<style>{estilos}</style>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="lab-hero">
        <div class="lab-art" aria-hidden="true"{hero_art}></div>
        <div class="lab-logo">{icon("flask")}</div>
        <div class="lab-heading">
            <h1>Predicción de diabetes</h1>
            <p><strong>UPAO</strong> · Proyecto de IA</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL.is_file():
    st.error("Ejecuta primero: python src/evaluar_modelo_final.py")
    st.stop()


@st.cache_resource
def load_model(path, version):
    return joblib.load(path)


model = load_model(str(MODEL), MODEL.stat().st_mtime_ns)
metrics = pd.read_csv(RESULTS / "metricas_prueba.csv")
row = metrics.iloc[0]


@st.dialog("Resultado de la clasificación")
def show_result(prediction):
    st.metric("Clase predicha", str(prediction))
    if prediction == 1:
        st.warning("Clase positiva según el modelo SVM.")
    else:
        st.info("Clase negativa según el modelo SVM.")
    st.caption("Prototipo académico. Sin uso diagnóstico.")
    if st.button("Volver al formulario", type="primary"):
        st.rerun()


FIELDS = [
    ("Pregnancies", "Embarazos", "user", 0, 2, "%d"),
    ("Glucose", "Glucosa", "drop", 1, 120, "%d"),
    ("BloodPressure", "Presión diastólica", "heart", 1, 70, "%d"),
    ("SkinThickness", "Pliegue cutáneo", "waves", 1, 25, "%d"),
    ("Insulin", "Insulina", "syringe", 1, 100, "%d"),
    ("BMI", "IMC", "scale", 0.1, 28.5, "%.1f"),
    ("DiabetesPedigreeFunction", "Antecedentes familiares", "users", 0.0, 0.351, "%.3f"),
    ("Age", "Edad", "calendar", 21, 35, "%d"),
]

prediction_tab, evaluation_tab = st.tabs(["▥  Predicción", "▤  Resultados de prueba"])

with prediction_tab:
    with st.form("datos_clinicos"):
        st.markdown(
            f'<h2 class="lab-form-title">{icon("drop")} Ingresar datos del registro</h2>',
            unsafe_allow_html=True,
        )
        st.caption("Clasifica el caso de ejemplo o modifica los valores.")
        values = {}
        for start in (0, 4):
            for col, field in zip(st.columns(4), FIELDS[start:start + 4]):
                key, label, symbol, minimum, default, fmt = field
                with col:
                    st.markdown(
                        f'<div class="lab-field">{icon(symbol)}<span>{label}</span></div>',
                        unsafe_allow_html=True,
                    )
                    values[key] = st.number_input(
                        label, min_value=minimum, value=default, format=fmt,
                        key=key, label_visibility="collapsed",
                    )

        _, columna_central, _ = st.columns([1, 2, 1])

        with columna_central:
            classify = st.form_submit_button(
                "▷  Clasificar registro",
                type="primary",
                use_container_width=True,
            )

    if classify:
        missing = [label for key, label, *_ in FIELDS if values[key] is None]
        if missing:
            st.error("Completa estos campos: " + ", ".join(missing) + ".")
        else:
            entry = pd.DataFrame([values], columns=VARIABLES)
            show_result(int(model.predict(entry)[0]))

    st.markdown(
        f"""
        <div class="lab-summary">
          <div><span class="lab-chip">{icon("chip")} SVM</span></div>
          <div>Exactitud<strong>{row['accuracy']:.2%}</strong></div>
          <div>Sensibilidad<strong>{row['recall']:.2%}</strong></div>
          <div>AUC-ROC<strong>{row['auc_roc']:.3f}</strong></div>
        </div>
        <div class="lab-footer">Prototipo académico. Sin uso diagnóstico.</div>
        """,
        unsafe_allow_html=True,
    )

with evaluation_tab:
    st.subheader("Desempeño del modelo en prueba")
    st.caption("Evaluación del SVM con 231 registros reservados.")
    for col, label, value in zip(
        st.columns(5),
        ["Exactitud", "Precisión", "Sensibilidad", "F1", "AUC-ROC"],
        [
            f"{row['accuracy']:.2%}", f"{row['precision']:.2%}",
            f"{row['recall']:.2%}", f"{row['f1']:.3f}", f"{row['auc_roc']:.3f}",
        ],
    ):
        col.metric(label, value)

    with st.expander("Ver tabla completa de métricas"):
        st.dataframe(metrics.round(4), hide_index=True)

    st.markdown("#### Matriz de confusión y curva ROC")
    left, right = st.columns(2)
    left.image(str(RESULTS / "matriz_confusion.png"))
    right.image(str(RESULTS / "curva_roc.png"))
