from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.preparar_datos import VARIABLES


PROYECTO = Path(__file__).resolve().parent
RUTA_MODELO = PROYECTO / "modelos" / "modelo_diabetes.joblib"
RESULTADOS = PROYECTO / "resultados"

st.set_page_config(page_title="Predicción de diabetes", layout="wide")
st.title("Predicción de diabetes")
st.caption("Prototipo académico de clasificación. Sin uso diagnóstico.")

if not RUTA_MODELO.is_file():
    st.error("Ejecuta primero: python src/evaluar_modelo_final.py")
    st.stop()


@st.cache_resource
def cargar_modelo(ruta, version):
    """Cargar el modelo una vez por versión del archivo."""
    return joblib.load(ruta)


modelo = cargar_modelo(str(RUTA_MODELO), RUTA_MODELO.stat().st_mtime_ns)

prediccion, evaluacion = st.tabs(["Predicción", "Resultados de prueba"])

with prediccion:
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
            antecedentes = st.number_input(
                "Índice de antecedentes familiares",
                min_value=0.0,
                value=0.351,
                format="%.3f",
            )
            edad = st.number_input("Edad (años)", min_value=21, value=35)

        calcular = st.form_submit_button("Clasificar registro")

    if calcular:
        entrada = pd.DataFrame([{
            "Pregnancies": embarazos,
            "Glucose": glucosa,
            "BloodPressure": presion,
            "SkinThickness": pliegue,
            "Insulin": insulina,
            "BMI": imc,
            "DiabetesPedigreeFunction": antecedentes,
            "Age": edad,
        }], columns=VARIABLES)

        clase = int(modelo.predict(entrada)[0])
        etiqueta = "positiva" if clase == 1 else "negativa"
        st.info(f"Clase predicha: {clase} ({etiqueta} según el modelo)")

with evaluacion:
    metricas = pd.read_csv(RESULTADOS / "metricas_prueba.csv")
    st.dataframe(metricas.round(4), hide_index=True)

    izquierda, derecha = st.columns(2)
    izquierda.image(str(RESULTADOS / "matriz_confusion.png"))
    derecha.image(str(RESULTADOS / "curva_roc.png"))