# Variables de sentimiento con retardo temporal

## Objetivo

Este experimento prueba si las noticias financieras tienen un efecto que persiste más allá del día de publicación. Para ello se crea un dataset `lagged_hybrid` que añade retardos y ventanas móviles de sentimiento al dataset híbrido actual.

## Variables generadas

Para cada variable de sentimiento se generan:

- Retardos de 1, 2 y 3 sesiones: `lag_1`, `lag_2`, `lag_3`.
- Medias móviles de 3 y 5 sesiones: `rolling_3`, `rolling_5`.

Las variables se calculan por `ticker`, ordenando cronológicamente cada serie. Las medias móviles usan información disponible hasta la fecha observada. Los valores iniciales sin historial suficiente se rellenan con `0`.

## Archivos generados

- `data/processed/hybrid/final_hybrid_dataset_lagged.csv`
- `data/processed/model/lagged_hybrid_train.csv`
- `data/processed/model/lagged_hybrid_test.csv`
- `data/processed/model/X_lagged_hybrid_train.csv`
- `data/processed/model/X_lagged_hybrid_test.csv`
- `reports/historical/tuning/model_progression_summary.csv`

El dataset mantiene las mismas filas, fechas, tickers y variable objetivo que el híbrido original. Solo se añaden variables explicativas.

## Dimensiones

- Filas totales: 11.056
- Filas train: 8.844
- Filas test: 2.212
- Variables retardadas nuevas: 65
- Variables predictoras totales en `lagged_hybrid`: 92

## Resultados con umbral 0.5

| Modelo | Accuracy híbrido | Accuracy lagged | Delta accuracy | F1 híbrido | F1 lagged | Delta F1 | ROC-AUC híbrido | ROC-AUC lagged | Delta ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 0.5063 | 0.5063 | 0.0000 | 0.5781 | 0.5321 | -0.0459 | 0.5033 | 0.5042 | 0.0009 |
| random_forest | 0.4778 | 0.5086 | 0.0307 | 0.3899 | 0.5365 | 0.1466 | 0.5010 | 0.5147 | 0.0137 |
| hist_gradient_boosting | 0.5145 | 0.4973 | -0.0172 | 0.5487 | 0.5260 | -0.0227 | 0.5087 | 0.4957 | -0.0130 |
| xgboost | 0.5014 | 0.4896 | -0.0118 | 0.5194 | 0.4819 | -0.0375 | 0.5026 | 0.4993 | -0.0032 |
| lightgbm | 0.5045 | 0.5122 | 0.0077 | 0.5116 | 0.5422 | 0.0306 | 0.5119 | 0.5217 | 0.0098 |

## Lectura

El retardo de sentimiento mejora las estimaciones puntuales de `random_forest` y `lightgbm` frente al híbrido sin retardos. El mayor ROC-AUC observado del experimento aparece en `lagged_hybrid` con `lightgbm`, que obtiene ROC-AUC 0.5217 y F1 0.5422 con umbral 0.5. Esto no acredita una mejora estadísticamente concluyente.

El resultado no mejora todos los algoritmos. `hist_gradient_boosting` y `xgboost` empeoran al añadir las variables retardadas, lo que sugiere que el aumento de dimensionalidad no beneficia por igual a todos los modelos.

## Conclusión

La hipótesis de persistencia temporal sigue abierta. Los intervalos exploratorios por bloques temporales de la diferencia entre LightGBM con y sin retardos incluyen cero. El modelo trivial obtiene además accuracy 0.5389 y F1 0.7004. La alineación por fecha no controla la hora de publicación y el test se ha consultado en varios experimentos: estos resultados deben conservarse como exploratorios, no como evidencia final de capacidad predictiva. Véase `docs/revision_objetiva_2026-09-08.md`.
