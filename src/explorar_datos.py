from pathlib import Path

import pandas as pd


# 1. Encontrar la carpeta principal del proyecto.
CARPETA_PROYECTO = Path(__file__).resolve().parents[1]

# 2. Construir la dirección del archivo de datos.
RUTA_DATOS = CARPETA_PROYECTO / "data" / "diabetes.csv"


def main():
    """Cargar los datos y mostrar una revisión inicial."""

    # 3. Comprobar que el archivo está en la ubicación esperada.
    if not RUTA_DATOS.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en: {RUTA_DATOS}"
        )

    # 4. Leer el archivo CSV.
    datos = pd.read_csv(RUTA_DATOS)

    # 5. Mostrar una vista previa.
    print("\nPRIMEROS CINCO REGISTROS")
    print(datos.head().to_string(index=False))

    # 6. Mostrar el tamaño del dataset.
    print("\nTAMAÑO DEL DATASET")
    print(f"Registros: {datos.shape[0]}")
    print(f"Columnas: {datos.shape[1]}")

    # 7. Mostrar los nombres y tipos de las columnas.
    print("\nTIPOS DE DATOS")
    print(datos.dtypes)

    # 8. Contar valores faltantes reconocidos por Pandas.
    print("\nVALORES FALTANTES")
    print(datos.isna().sum())

    # 9. Contar filas completamente duplicadas.
    print("\nREGISTROS DUPLICADOS")
    print(datos.duplicated().sum())

    # 10. Revisar cuántos casos hay de cada clase.
    print("\nDISTRIBUCIÓN DE OUTCOME")
    print(datos["Outcome"].value_counts().sort_index())

    # 11. Revisar ceros que podrían representar mediciones faltantes.
    columnas_revisar = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]

    print("\nCEROS EN MEDICIONES CLÍNICAS")
    print((datos[columnas_revisar] == 0).sum())


# Ejecutar main cuando abrimos este archivo como programa.
if __name__ == "__main__":
    main()