# Métricas de evaluación

El problema del TFG se formula como clasificación binaria:

- `1`: el precio ajustado de cierre de la siguiente sesión es superior al de la sesión actual.
- `0`: el precio ajustado de cierre de la siguiente sesión no es superior al de la sesión actual.

## Métricas principales

- `accuracy`: proporción total de predicciones correctas.
- `precision`: proporción de predicciones alcistas que realmente fueron alcistas.
- `recall`: proporción de sesiones alcistas reales que el modelo detectó.
- `f1`: media armónica de precision y recall.

Estas métricas se calculan siempre para todos los modelos.

## Matriz de confusión

La evaluación guarda los cuatro valores de la matriz de confusión:

- `tn`: clase 0 predicha como 0.
- `fp`: clase 0 predicha como 1.
- `fn`: clase 1 predicha como 0.
- `tp`: clase 1 predicha como 1.

Esto permite revisar si el modelo tiende a generar más falsos positivos o falsos negativos.

## ROC-AUC

Cuando el modelo proporciona probabilidades, se calcula también `roc_auc`. Esta métrica resume la capacidad del modelo para ordenar observaciones positivas por encima de negativas sin depender de un único umbral de clasificación.

## Evaluación por ticker

Además de la evaluación global, las métricas pueden calcularse por `ticker`. Esto permite detectar si el rendimiento del modelo es homogéneo o si depende de una empresa concreta.
