# Evaluación de modelos entrenados

La evaluación consolidada se genera con:

```bash
python src/models/evaluate_models.py
```

## Entradas

El script lee todas las predicciones disponibles en:

```text
reports/predictions/
```

Cada archivo debe contener:

- `dataset_type`
- `model_name`
- `ticker`
- `target`
- `prediction`
- `probability`, si el modelo proporciona probabilidades

## Salidas

La evaluación genera:

- `reports/metrics/all_model_metrics.csv`: métricas globales y por ticker.
- `reports/metrics/global_model_metrics.csv`: métricas globales por modelo.

## Resultado

Se evaluaron ocho combinaciones:

- 4 modelos financieros base.
- 4 modelos híbridos.

Cada combinación se evaluó de forma global y por ticker. Las métricas calculadas son accuracy, precision, recall, F1, ROC-AUC y matriz de confusión.
