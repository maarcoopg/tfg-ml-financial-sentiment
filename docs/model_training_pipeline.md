# Pipeline de entrenamiento

El entrenamiento de modelos se centraliza en:

```text
src/models/train_models.py
```

El script recibe el tipo de dataset:

```bash
python src/models/train_models.py --dataset base
python src/models/train_models.py --dataset hybrid
```

También permite entrenar modelos concretos:

```bash
python src/models/train_models.py --dataset base --models logistic_regression random_forest
```

## Entradas

Los datos se cargan desde `data/processed/model/`:

- `base_train.csv` y `base_test.csv` para el modelo financiero base.
- `hybrid_train.csv` y `hybrid_test.csv` para el modelo híbrido.
- `feature_sets.json` para definir las columnas predictoras de cada enfoque.

## Salidas

El pipeline genera:

- Modelos entrenados en `models/trained/`.
- Predicciones en `reports/predictions/`.
- Métricas en `reports/metrics/`.
- Metadatos de entrenamiento en `reports/model_metadata/`.

## Modelos candidatos

Los modelos disponibles son:

- `dummy`
- `logistic_regression`
- `random_forest`
- `hist_gradient_boosting`

Todos los modelos usan una semilla fija (`42`) cuando el estimador lo permite.
