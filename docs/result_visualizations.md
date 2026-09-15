# Visualizaciones de resultados

Las figuras se generan con:

```bash
python src/visualization/plot_results.py
```

## Figuras generadas

- `reports/historical/figures/global_metrics_by_model.png`: métricas globales por modelo y tipo de dataset.
- `reports/historical/figures/global_metrics_facets.png`: métricas globales separadas por panel.
- `reports/historical/figures/base_vs_hybrid_metric_deltas.png`: diferencias entre modelo híbrido y modelo base.
- `reports/historical/figures/hybrid_top_feature_importance.png`: variables más importantes en modelos híbridos interpretables.

## Uso en la memoria

Las figuras permiten presentar:

- Rendimiento global de los modelos.
- Comparación directa entre enfoque base e híbrido.
- Peso relativo de variables financieras y de sentimiento.

Las tablas originales se mantienen en `reports/historical/metrics/`, `reports/historical/comparison/` y `reports/historical/feature_importance/`.
