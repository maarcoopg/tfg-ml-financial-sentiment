# Modelos híbridos

Los modelos híbridos se entrenan con las mismas variables financieras del modelo base y añaden variables agregadas de sentimiento financiero diario.

## Datos utilizados

- Entrenamiento: `data/processed/model/hybrid_train.csv`
- Prueba: `data/processed/model/hybrid_test.csv`
- Activos incluidos: AAPL, MSFT, NVDA y TSLA
- Activo excluido: SPY

El universo de activos, fechas y variable objetivo coincide con el del modelo base para que la comparación sea directa.

## Modelos entrenados

- `dummy`: referencia que predice la clase mayoritaria.
- `logistic_regression`: regresión logística con imputación y escalado.
- `random_forest`: bosque aleatorio con profundidad limitada.
- `hist_gradient_boosting`: boosting basado en histogramas.

## Métricas globales en test

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| dummy | 0.5389 | 0.5389 | 1.0000 | 0.7004 | 0.5000 |
| logistic_regression | 0.5077 | 0.5356 | 0.6493 | 0.5870 | 0.5011 |
| random_forest | 0.5136 | 0.5488 | 0.5470 | 0.5479 | 0.5040 |
| hist_gradient_boosting | 0.5059 | 0.5460 | 0.4933 | 0.5183 | 0.5054 |

Las métricas completas, incluyendo matriz de confusión y desglose por ticker, se guardan en `reports/metrics/`.
