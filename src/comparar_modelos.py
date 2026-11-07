from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from entrenar_logistica import crear_modelo
from preparar_datos import preparar_datos


def crear_modelos():
    """Construir los tres modelos con su propio procesamiento."""
    return {
        "Regresion Logistica": crear_modelo(),

        "Random Forest": Pipeline([
            ("imputacion", SimpleImputer(strategy="median")),
            ("clasificador", RandomForestClassifier(
                n_estimators=150,
                max_depth=4,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
            )),
        ]),

        "SVM": Pipeline([
            ("imputacion", SimpleImputer(strategy="median")),
            ("escalado", StandardScaler()),
            ("clasificador", SVC(
                kernel="rbf",
                C=1.0,
                class_weight="balanced",
            )),
        ]),
    }


def main():
    # Usamos solo entrenamiento. La prueba queda reservada.
    X_train, _, y_train, _ = preparar_datos()

    validacion = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )

    metricas = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": "recall",
        "f1": "f1",
        "auc_roc": "roc_auc",
    }

    filas = []

    for nombre, modelo in crear_modelos().items():
        print(f"Validando {nombre}...")

        resultados = cross_validate(
            modelo,
            X_train,
            y_train,
            cv=validacion,
            scoring=metricas,
            n_jobs=1,
            error_score="raise",
        )

        fila = {"modelo": nombre}
        for metrica in metricas:
            fila[metrica] = resultados[f"test_{metrica}"].mean()

        fila["recall_desviacion"] = resultados["test_recall"].std()
        filas.append(fila)

    # Ordenar por Recall. Si hay empate, usar F1 y luego AUC-ROC.
    tabla = pd.DataFrame(filas).sort_values(
        by=["recall", "f1", "auc_roc"],
        ascending=False,
    )

    carpeta = Path(__file__).resolve().parents[1] / "resultados"
    carpeta.mkdir(exist_ok=True)
    tabla.to_csv(carpeta / "comparacion_inicial.csv", index=False)

    print("\nCOMPARACIÓN INICIAL — PROMEDIO DE 5 PARTICIONES")
    print(tabla.round(3).to_string(index=False))
    print(f"\nPrimero por Recall: {tabla.iloc[0]['modelo']}")
    print("Archivo generado: resultados/comparacion_inicial.csv")
    print("El conjunto de prueba permanece reservado.")


if __name__ == "__main__":
    main()