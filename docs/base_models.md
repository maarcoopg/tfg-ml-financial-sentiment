# Modelos base

Los modelos base se entrenan con variables financieras únicamente. No incluyen ninguna variable de sentimiento.

## Datos utilizados

- Entrenamiento: `data/processed/model/base_train.csv`
- Prueba: `data/processed/model/base_test.csv`
- Activos incluidos: AAPL, MSFT, NVDA y TSLA
- Activo excluido: SPY

SPY queda fuera del entrenamiento principal para mantener el mismo universo de activos que el modelo híbrido.

## Modelos entrenados

- `dummy`: referencia que predice la clase mayoritaria.
- `logistic_regression`: regresión logística con imputación y escalado.
- `random_forest`: bosque aleatorio con profundidad limitada.
- `hist_gradient_boosting`: boosting basado en histogramas.

## Métricas globales en test

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| dummy | 0.5389 | 0.5389 | 1.0000 | 0.7004 | 0.5000 |
| logistic_regression | 0.5316 | 0.5401 | 0.8817 | 0.6699 | 0.5115 |
| random_forest | 0.5095 | 0.5388 | 0.6233 | 0.5780 | 0.5124 |
| hist_gradient_boosting | 0.4855 | 0.5265 | 0.4497 | 0.4851 | 0.4930 |

Las métricas completas, incluyendo matriz de confusión y desglose por ticker, se guardan en `reports/metrics/`.
