# Comparación base vs híbrido

La comparación se genera con:

```bash
python src/models/compare_models.py
```

## Criterio de comparación

Cada modelo híbrido se compara contra el modelo base con el mismo algoritmo. La diferencia se calcula como:

```text
métrica híbrida - métrica base
```

Los dos enfoques utilizan los mismos activos, fechas y variable objetivo. SPY no participa en esta comparación.

## Métricas globales

| Modelo | Accuracy base | Accuracy híbrido | Delta accuracy | F1 base | F1 híbrido | Delta F1 | ROC-AUC base | ROC-AUC híbrido | Delta ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dummy | 0.5389 | 0.5389 | 0.0000 | 0.7004 | 0.7004 | 0.0000 | 0.5000 | 0.5000 | 0.0000 |
| hist_gradient_boosting | 0.4855 | 0.5059 | 0.0203 | 0.4851 | 0.5183 | 0.0332 | 0.4930 | 0.5054 | 0.0124 |
| logistic_regression | 0.5316 | 0.5077 | -0.0240 | 0.6699 | 0.5870 | -0.0828 | 0.5115 | 0.5011 | -0.0103 |
| random_forest | 0.5095 | 0.5136 | 0.0041 | 0.5780 | 0.5479 | -0.0301 | 0.5124 | 0.5040 | -0.0084 |

## Interpretación preliminar

La incorporación de sentimiento mejora las métricas del modelo `hist_gradient_boosting`, especialmente F1 y accuracy. En `random_forest` la mejora de accuracy es leve, pero F1 y ROC-AUC empeoran. En `logistic_regression`, el modelo base obtiene mejores resultados.

Con estos resultados no puede afirmarse que el sentimiento mejore de forma general todos los modelos. La conclusión más prudente es que la señal de sentimiento puede aportar información en determinados algoritmos, pero su efecto depende del modelo utilizado.

Las tablas completas se guardan en:

- `reports/comparison/base_vs_hybrid_global.csv`
- `reports/comparison/base_vs_hybrid_by_ticker.csv`
