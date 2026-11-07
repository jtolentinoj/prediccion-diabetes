from pathlib import Path

import pandas as pd
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from comparar_modelos import crear_modelos
from preparar_datos import preparar_datos


def main():
    X_train, _, y_train, _ = preparar_datos()

    # Valores que probaremos para cada clasificador.
    parametros = {
        "Regresion Logistica": {
            "clasificador__C": [0.1, 1.0, 10.0],
        },
        "Random Forest": {
            "clasificador__max_depth": [4, 8, None],
            "clasificador__min_samples_leaf": [2, 5],
        },
        "SVM": {
            "clasificador__C": [0.1, 1.0, 10.0],
            "clasificador__gamma": ["scale", 0.01],
        },
    }

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
        print(f"Ajustando {nombre}...")

        busqueda = GridSearchCV(
            estimator=modelo,
            param_grid=parametros[nombre],
            scoring=metricas,
            cv=validacion,
            refit=False,
            n_jobs=1,
            error_score="raise",
        )
        busqueda.fit(X_train, y_train)

        resultados = pd.DataFrame(busqueda.cv_results_)
        mejor = resultados.sort_values(
            by=["mean_test_recall", "mean_test_f1", "mean_test_auc_roc"],
            ascending=False,
        ).iloc[0]

        fila = {"modelo": nombre}
        for metrica in metricas:
            fila[metrica] = mejor[f"mean_test_{metrica}"]

        fila["parametros"] = mejor["params"]
        filas.append(fila)

    tabla = pd.DataFrame(filas).sort_values(
        by=["recall", "f1", "auc_roc"],
        ascending=False,
    )

    carpeta = Path(__file__).resolve().parents[1] / "resultados"
    carpeta.mkdir(exist_ok=True)
    tabla.to_csv(carpeta / "comparacion_ajustada.csv", index=False)

    ganador = tabla.iloc[0]

    print("\nMEJOR CONFIGURACIÓN DE CADA MODELO")
    print(tabla.drop(columns="parametros").round(6).to_string(index=False))
    print(f"\nModelo seleccionado: {ganador['modelo']}")
    print(f"Parámetros: {ganador['parametros']}")
    print("Archivo generado: resultados/comparacion_ajustada.csv")


if __name__ == "__main__":
    main()