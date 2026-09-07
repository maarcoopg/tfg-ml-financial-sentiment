# División entrenamiento-prueba

## Archivo de partida

La división se realiza sobre:

```text
data/processed/hybrid/final_hybrid_dataset.csv
```

## Criterio de división

El dataset se ordena cronológicamente por `Date` y se divide usando el 80% inicial de fechas únicas para entrenamiento y el 20% final para prueba.

La división no es aleatoria. Esto evita que observaciones futuras entren en el conjunto de entrenamiento.

## Activos incluidos

Los entrenamientos principales usan únicamente AAPL, MSFT, NVDA y TSLA. SPY queda excluido porque es un ETF y no dispone de una señal de sentimiento corporativo comparable a la de las empresas. Esta decisión mantiene equivalente el universo de activos entre el modelo financiero base y el modelo híbrido.

## Archivos generados

Los archivos se guardan en:

```text
data/processed/model/
```

Archivos completos:

- `train.csv`
- `test.csv`

Datasets comparables para modelado:

- `base_train.csv`
- `base_test.csv`
- `hybrid_train.csv`
- `hybrid_test.csv`

Variables predictoras y objetivo generales:

- `X_train.csv`
- `X_test.csv`
- `y_train.csv`
- `y_test.csv`

Variables predictoras separadas por enfoque:

- `X_base_train.csv`
- `X_base_test.csv`
- `X_hybrid_train.csv`
- `X_hybrid_test.csv`

`X_*` contiene únicamente variables predictoras. `y_*` contiene `ticker`, `Date` y `target` para mantener trazabilidad.

## Variables por enfoque

El modelo base utiliza solo variables financieras: precios, volumen, retornos e indicadores técnicos.

El modelo híbrido utiliza las mismas variables financieras y añade variables agregadas de sentimiento financiero diario.

La configuración exacta de columnas queda guardada en:

```text
data/processed/model/feature_sets.json
```
