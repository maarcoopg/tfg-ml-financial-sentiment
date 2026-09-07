# Division entrenamiento-prueba

## Archivo de partida

La division se realiza sobre:

```text
data/processed/hybrid/final_hybrid_dataset.csv
```

## Criterio de division

El dataset se ordena cronologicamente por `Date` y se divide usando el 80% inicial de fechas unicas para entrenamiento y el 20% final para prueba.

La division no es aleatoria. Esto evita que observaciones futuras entren en el conjunto de entrenamiento.

## Archivos generados

Los archivos se guardan en:

```text
data/processed/model/
```

Archivos completos:

- `train.csv`
- `test.csv`

Variables predictoras y objetivo:

- `X_train.csv`
- `X_test.csv`
- `y_train.csv`
- `y_test.csv`

`X_*` contiene todas las columnas salvo `target`. `y_*` contiene `ticker`, `Date` y `target` para mantener trazabilidad.
