# Revisión final de reproducibilidad

## Estado revisado

La revisión final se realizó sobre el bloque ya procesado del proyecto, desde la división temporal hasta modelos, evaluación, comparación, interpretabilidad y visualizaciones.

## Comandos verificados

```bash
python -m compileall src
python src/data/split_train_test.py
python src/models/train_models.py --dataset base
python src/models/train_models.py --dataset hybrid
python src/models/evaluate_models.py
python src/models/compare_models.py
python src/models/analyze_feature_importance.py
python src/visualization/plot_results.py
python -m jupyter nbconvert --to notebook --execute notebooks/01_dataset_exploration.ipynb --output 01_dataset_exploration.executed.ipynb --output-dir notebooks
python -m jupyter nbconvert --to notebook --execute notebooks/02_model_results.ipynb --output 02_model_results.executed.ipynb --output-dir notebooks
```

Todos los comandos finalizaron correctamente.

## Salidas comprobadas

- `data/processed/model/base_train.csv`: 8.844 filas.
- `data/processed/model/base_test.csv`: 2.212 filas.
- `data/processed/model/hybrid_train.csv`: 8.844 filas.
- `data/processed/model/hybrid_test.csv`: 2.212 filas.
- `reports/metrics/all_model_metrics.csv`: 40 filas.
- `reports/metrics/global_model_metrics.csv`: 8 filas.
- `reports/comparison/base_vs_hybrid_global.csv`: 4 filas.
- `reports/comparison/base_vs_hybrid_by_ticker.csv`: 16 filas.
- `reports/feature_importance/feature_importance.csv`: 82 filas.
- `reports/figures/`: 4 figuras PNG.

## Consistencia metodológica

Los datasets de modelado incluyen AAPL, MSFT, NVDA y TSLA. SPY queda excluido del entrenamiento principal y de la comparación base vs híbrido para mantener el mismo universo de activos en ambos enfoques.

La división entrenamiento-prueba es temporal y no aleatoria. El conjunto de entrenamiento termina el 13 de octubre de 2023 y el conjunto de prueba empieza el 16 de octubre de 2023.

## Limpieza

Se añadieron reglas en `.gitignore` para evitar versionar copias temporales de notebooks ejecutados y el directorio local usado para revisar el borrador de issues.

## Advertencias no bloqueantes

Durante el entrenamiento, `joblib` mostró una advertencia de Windows al no encontrar `wmic` para detectar núcleos físicos. El entrenamiento continuó usando núcleos lógicos.

Durante la ejecución de notebooks, Jupyter mostró una advertencia sobre IDs de celdas ausentes y otra de compatibilidad del event loop de Windows. Los notebooks se ejecutaron correctamente.
