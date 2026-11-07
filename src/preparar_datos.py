from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


CARPETA_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_DATOS = CARPETA_PROYECTO / "data" / "diabetes.csv"

VARIABLES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

COLUMNAS_CON_CEROS_INVALIDOS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def preparar_datos():
    """Leer, revisar y separar los datos sin aprender de la prueba."""

    # 1. Leer el archivo original.
    datos = pd.read_csv(RUTA_DATOS)

    # 2. Comprobar las columnas necesarias.
    columnas_necesarias = VARIABLES + ["Outcome"]

    faltantes = [
        columna
        for columna in columnas_necesarias
        if columna not in datos.columns
    ]

    if faltantes:
        raise ValueError(f"Faltan estas columnas: {faltantes}")

    # 3. Eliminar filas completamente duplicadas.
    datos = datos.drop_duplicates().copy()

    # 4. Separar entradas y etiqueta.
    X = datos[VARIABLES].copy()
    y = datos["Outcome"].copy()

    # 5. Marcar como faltantes los ceros de ciertas mediciones.
    # No se modifica el archivo CSV original.
    X[COLUMNAS_CON_CEROS_INVALIDOS] = (
        X[COLUMNAS_CON_CEROS_INVALIDOS].replace(0, np.nan)
    )

    # 6. Reservar el 30 % para la evaluación final.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = preparar_datos()

    print("\nDIVISIÓN DEL DATASET")
    print(f"Entrenamiento: {len(X_train)} registros")
    print(f"Prueba: {len(X_test)} registros")

    print("\nCLASES EN ENTRENAMIENTO")
    print(y_train.value_counts().sort_index())

    print("\nCLASES EN PRUEBA")
    print(y_test.value_counts().sort_index())

    print("\nFALTANTES EN LAS ENTRADAS DE ENTRENAMIENTO")
    print(X_train.isna().sum())

    print("\nFALTANTES EN LAS ENTRADAS DE PRUEBA")
    print(X_test.isna().sum())

    # Comprobar que ninguna fila esté en ambos conjuntos.
    filas_compartidas = X_train.index.intersection(X_test.index)

    print("\nFILAS COMPARTIDAS ENTRE ENTRENAMIENTO Y PRUEBA")
    print(len(filas_compartidas))

    print("\nPreparación terminada correctamente.")


if __name__ == "__main__":
    main()