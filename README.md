# tfg-ml-financial-sentiment

Trabajo de Fin de Grado sobre un modelo híbrido de aprendizaje automático para la predicción de tendencias bursátiles mediante análisis técnico y sentimiento financiero.

## Objetivo

El proyecto compara dos enfoques de clasificación binaria:

- Modelo financiero base: usa precios, volumen, retornos e indicadores técnicos.
- Modelo híbrido: usa las mismas variables financieras y añade sentimiento financiero agregado por día.

La variable objetivo vale `1` cuando el precio ajustado de cierre de la siguiente sesión es superior al de la sesión actual, y `0` en caso contrario.

## Activos

Los datos financieros iniciales incluyen AAPL, MSFT, NVDA, TSLA y SPY. Para el entrenamiento y la comparación principal se usan AAPL, MSFT, NVDA y TSLA. SPY queda excluido porque es un ETF y no existe una señal de sentimiento corporativo comparable con la de las empresas.

## Ejecución

La revisión metodológica se ejecuta con `python -m src.experiments.run_review`.
Usa horarios reales del mercado, validación temporal anidada con purga y salidas
aisladas por ejecución. Véase `docs/revision_implementation.md`. Los informes
anteriores a esta revisión son históricos exploratorios.

La guía completa de ejecución está en:

```text
docs/full_pipeline_execution.md
```

El bloque final de modelado se reproduce con:

```bash
python src/data/split_train_test.py
python src/models/train_models.py --dataset base
python src/models/train_models.py --dataset hybrid
python src/models/evaluate_models.py
python src/models/compare_models.py
python src/models/analyze_feature_importance.py
python src/visualization/plot_results.py
```
