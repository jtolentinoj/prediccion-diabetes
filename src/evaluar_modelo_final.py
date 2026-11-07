from ast import literal_eval
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from comparar_modelos import crear_modelos
from preparar_datos import preparar_datos


PROYECTO = Path(__file__).resolve().parents[1]
RESULTADOS = PROYECTO / "resultados"
MODELOS = PROYECTO / "modelos"


def main():
    # 1. Leer la primera fila de la tabla ordenada por Recall, F1 y AUC.
    comparacion = pd.read_csv(RESULTADOS / "comparacion_ajustada.csv")
    ganador = comparacion.iloc[0]
    nombre = ganador["modelo"]
    parametros = literal_eval(ganador["parametros"])

    # 2. Reconstruir el Pipeline con los parámetros seleccionados.
    modelo = crear_modelos()[nombre]
    modelo.set_params(**parametros)

    # 3. Entrenar con todo el conjunto de entrenamiento.
    X_train, X_test, y_train, y_test = preparar_datos()
    modelo.fit(X_train, y_train)

    # 4. Evaluar con el conjunto de prueba reservado.
    predicciones = modelo.predict(X_test)
    if hasattr(modelo, "decision_function"):
        puntuaciones = modelo.decision_function(X_test)
    else:
        puntuaciones = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        "accuracy": accuracy_score(y_test, predicciones),
        "precision": precision_score(y_test, predicciones, zero_division=0),
        "recall": recall_score(y_test, predicciones, zero_division=0),
        "f1": f1_score(y_test, predicciones, zero_division=0),
        "auc_roc": roc_auc_score(y_test, puntuaciones),
    }

    # 5. Guardar métricas y gráficos.
    tabla = pd.DataFrame([{"modelo": nombre, **metricas}])
    tabla.to_csv(RESULTADOS / "metricas_prueba.csv", index=False)

    matriz = confusion_matrix(y_test, predicciones, labels=[0, 1])
    grafico = ConfusionMatrixDisplay(matriz, display_labels=[0, 1])
    grafico.plot(cmap="Blues", colorbar=False)
    grafico.ax_.set(
        title=f"Matriz de confusión — {nombre}",
        xlabel="Clase predicha",
        ylabel="Clase real",
    )
    grafico.figure_.savefig(
        RESULTADOS / "matriz_confusion.png", dpi=160, bbox_inches="tight"
    )
    plt.close(grafico.figure_)

    curva = RocCurveDisplay.from_predictions(
        y_test, puntuaciones, name=nombre
    )
    curva.ax_.plot([0, 1], [0, 1], "--", color="gray")
    curva.ax_.set_title("Curva ROC en el conjunto de prueba")
    curva.figure_.savefig(
        RESULTADOS / "curva_roc.png", dpi=160, bbox_inches="tight"
    )
    plt.close(curva.figure_)

    # 6. Guardar el Pipeline que cargará la aplicación.
    MODELOS.mkdir(exist_ok=True)
    joblib.dump(modelo, MODELOS / "modelo_diabetes.joblib")

    print(f"\nModelo final: {nombre}")
    print(f"Parámetros: {parametros}")
    print(f"Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")
    print("\nMÉTRICAS DE PRUEBA")
    print(tabla.round(4).to_string(index=False))

    print("\nMATRIZ DE CONFUSIÓN")
    print("Filas: clase real | Columnas: clase predicha")
    print(matriz)
    print("\nModelo guardado en modelos/modelo_diabetes.joblib")


if __name__ == "__main__":
    main()