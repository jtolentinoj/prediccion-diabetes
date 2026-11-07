from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.preparar_datos import VARIABLES


PROYECTO = Path(__file__).resolve().parent
RUTA_MODELO = PROYECTO / "modelos" / "modelo_diabetes.joblib"
RESULTADOS = PROYECTO / "resultados"

st.set_page_config(
    page_title="Predicción de diabetes | Proyecto de IA",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
            ellipse at top right,
            rgba(14, 116, 144, 0.12),
            transparent 45%
            ),
        linear-gradient(160deg, #f8fafc 0%, #edf4f8 100%);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }
    .portada-ia {
        background: linear-gradient(120deg, #0f172a, #164e63);
        color: #ffffff;
        border-radius: 20px;
        padding: clamp(24px, 4vw, 48px);
        margin-bottom: 24px;
    }

    .portada-ia h1 {
        color: #ffffff;
        font-size: clamp(2rem, 4vw, 3.25rem);
        margin: 8px 0;
        padding: 0;
    }

    .portada-ia p {
        color: #e2e8f0;
        margin: 8px 0 0;
    }

    .portada-ia .etiqueta {
        color: #a5f3fc;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.06em;
    }

    .portada-ia .detalle {
        color: #cbd5e1;
        font-size: 0.9rem;
        margin-top: 16px;
    }
    </style>

    <div class="portada-ia">
        <span class="etiqueta">UPAO · PROYECTO DE INTELIGENCIA ARTIFICIAL</span>
        <h1>Predicción de diabetes</h1>
        <p>Clasificación con aprendizaje automático a partir de ocho variables.</p>
        <p class="detalle">
            Modelo SVM · 231 registros de prueba · Prototipo académico
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not RUTA_MODELO.is_file():
    st.error("Ejecuta primero: python src/evaluar_modelo_final.py")
    st.stop()


@st.cache_resource
def cargar_modelo(ruta, version):
    """Cargar el modelo una vez por versión del archivo."""
    return joblib.load(ruta)


modelo = cargar_modelo(str(RUTA_MODELO), RUTA_MODELO.stat().st_mtime_ns)
@st.dialog("Resultado de la clasificación")
def mostrar_resultado(clase):
    etiqueta = "Positiva" if clase == 1 else "Negativa"

    st.metric("Clase predicha", str(clase))
    st.info(f"Clasificación del modelo SVM: {etiqueta}")
    st.caption("Prototipo académico. Sin uso diagnóstico.")

    if st.button("Volver al formulario", type="primary"):
        st.rerun()

prediccion, evaluacion = st.tabs(["Predicción", "Resultados de prueba"])

with prediccion:
    st.subheader("Clasifica un registro de ejemplo")
    st.caption("Demostración académica. La clasificación no tiene uso diagnóstico.")
    st.write("Los valores iniciales corresponden a un caso de ejemplo.")

    with st.form("datos_clinicos"):
        izquierda, derecha = st.columns(2)

        with izquierda:
            embarazos = st.number_input("Embarazos", min_value=0, value=2)
            glucosa = st.number_input("Glucosa (mg/dL)", min_value=1, value=120)
            presion = st.number_input("Presión diastólica (mmHg)", min_value=1, value=70)
            pliegue = st.number_input("Pliegue cutáneo (mm)", min_value=1, value=25)

        with derecha:
            insulina = st.number_input("Insulina (µU/mL)", min_value=1, value=100)
            imc = st.number_input("IMC (kg/m²)", min_value=0.1, value=28.5)
            antecedentes = st.number_input( "Índice de antecedentes familiares", min_value=0.0,
                value=0.351,
                format="%.3f",
            )
            edad = st.number_input("Edad (años)", min_value=21, value=35)

        calcular = st.form_submit_button("Clasificar registro", type="primary")

        if calcular:
            campos = [
                ("Embarazos", embarazos),
                ("Glucosa", glucosa),
                ("Presión diastólica", presion),
                ("Pliegue cutáneo", pliegue),
                ("Insulina", insulina),
                ("IMC", imc),
                ("Índice de antecedentes familiares", antecedentes),
                ("Edad", edad),
            ]

            faltantes = [
                nombre for nombre, valor in campos if valor is None
            ]

            if faltantes:
                st.error("Completa estos campos: " + ", ".join(faltantes) + ".")
            else:
                entrada = pd.DataFrame(
                    [[valor for _, valor in campos]],
                    columns=VARIABLES,
                )
            clase = int(modelo.predict(entrada)[0])
            mostrar_resultado(clase)

with evaluacion:
    st.subheader("Desempeño del modelo en prueba")
    st.caption("Evaluación del SVM con 231 registros reservados.")

    metricas = pd.read_csv(RESULTADOS / "metricas_prueba.csv")
    fila = metricas.iloc[0]

    exactitud, precision, recall, f1, auc = st.columns(5)

    exactitud.metric("Exactitud", f"{fila['accuracy']:.2%}")
    precision.metric("Precisión", f"{fila['precision']:.2%}")
    recall.metric("Sensibilidad", f"{fila['recall']:.2%}")
    f1.metric("F1", f"{fila['f1']:.3f}")
    auc.metric("AUC-ROC", f"{fila['auc_roc']:.3f}")

    with st.expander("Ver tabla completa de métricas"):
        st.dataframe(metricas.round(4), hide_index=True)

    st.markdown("#### Matriz de confusión y curva ROC")

    izquierda, derecha = st.columns(2)
    izquierda.image(str(RESULTADOS / "matriz_confusion.png"))
    derecha.image(str(RESULTADOS / "curva_roc.png"))