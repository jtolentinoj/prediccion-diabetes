import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preparar_datos import preparar_datos


def crear_modelo():
    """Construir el procesamiento y el clasificador."""

    modelo = Pipeline(
        steps=[
            # Completar faltantes con la mediana de cada variable.
            ("imputacion", SimpleImputer(strategy="median")),

            # Estandarizar las variables numéricas.
            ("escalado", StandardScaler()),

            # Clasificador de referencia.
            (
                "clasificador",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    return modelo


def main():
    # La prueba se reserva: no se utiliza en esta comparación inicial.
    X_train, _, y_train, _ = preparar_datos()

    modelo = crear_modelo()

    # Dividir el entrenamiento en cinco particiones.
    validacion = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # Métricas que calcularemos en cada partición.
    metricas = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "auc_roc": "roc_auc",
    }

    print("\nEvaluando Regresión Logística...")

    resultados = cross_validate(
        estimator=modelo,
        X=X_train,
        y=y_train,
        cv=validacion,
        scoring=metricas,
        n_jobs=1,
        error_score="raise",
    )

    # Organizar las cinco evaluaciones en una tabla.
    tabla = pd.DataFrame(
        {
            nombre: resultados[f"test_{nombre}"]
            for nombre in metricas
        }
    )

    tabla.index = range(1, 6)
    tabla.index.name = "Particion"

    print("\nRESULTADOS POR PARTICIÓN")
    print(tabla.round(3).to_string())

    print("\nRESUMEN DE VALIDACIÓN CRUZADA")
    resumen = pd.DataFrame(
        {
            "Promedio": tabla.mean(),
            "Desviacion": tabla.std(ddof=0),
        }
    )

    print(resumen.round(3).to_string())

    print("\nEl conjunto de prueba permanece reservado.")


if __name__ == "__main__":
    main()