# Validación cruzada temporal e hiperparámetros

## Objetivo

Se incorporó una validación cruzada temporal dentro del conjunto de entrenamiento para ajustar hiperparámetros sin utilizar el conjunto de test final. El test final se mantiene reservado para una única evaluación posterior.

## Diseño

La validación se realiza sobre fechas únicas del conjunto de entrenamiento mediante tres ventanas tipo walk-forward:

- Cada fold entrena con fechas anteriores.
- Cada fold valida sobre fechas posteriores.
- No se mezclan observaciones futuras en el entrenamiento.

Se ajustaron los siguientes modelos:

- `logistic_regression`
- `random_forest`
- `hist_gradient_boosting`
- `xgboost`
- `lightgbm`

El modelo `dummy` se mantiene como referencia y no se ajusta.

## Criterio de selección

Los hiperparámetros se seleccionan priorizando `ROC-AUC` en validación temporal. Además, se calcula un umbral optimizado para F1, pero se reportan dos variantes:

- `tuned_fixed_0_5`: hiperparámetros ajustados y umbral fijo 0.5.
- `tuned_optimized_threshold`: hiperparámetros ajustados y umbral elegido en validación para maximizar F1.

Esta separación evita confundir una mejora real del modelo con una subida artificial del F1 por predecir una proporción excesiva de observaciones como clase positiva.

## Búsqueda ampliada

La segunda pasada amplió el espacio de hiperparámetros y añadió XGBoost y LightGBM. En total se evaluaron 148 combinaciones de hiperparámetros y 444 entrenamientos de validación temporal.

## Resultados frente a modelos iniciales

| Dataset | Modelo | Variante | Accuracy inicial | Accuracy tuned | Delta accuracy | F1 inicial | F1 tuned | Delta F1 | ROC-AUC inicial | ROC-AUC tuned | Delta ROC-AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | hist_gradient_boosting | tuned_fixed_0_5 | 0.4855 | 0.5050 | 0.0194 | 0.4851 | 0.5409 | 0.0558 | 0.4930 | 0.4963 | 0.0033 |
| base | logistic_regression | tuned_fixed_0_5 | 0.5316 | 0.4914 | -0.0402 | 0.6699 | 0.4088 | -0.2610 | 0.5115 | 0.5113 | -0.0001 |
| base | random_forest | tuned_fixed_0_5 | 0.5095 | 0.5000 | -0.0095 | 0.5780 | 0.4841 | -0.0938 | 0.5124 | 0.5125 | 0.0001 |
| hybrid | hist_gradient_boosting | tuned_fixed_0_5 | 0.5059 | 0.5145 | 0.0086 | 0.5183 | 0.5487 | 0.0304 | 0.5054 | 0.5087 | 0.0033 |
| hybrid | logistic_regression | tuned_fixed_0_5 | 0.5077 | 0.5063 | -0.0014 | 0.5870 | 0.5781 | -0.0090 | 0.5011 | 0.5033 | 0.0021 |
| hybrid | random_forest | tuned_fixed_0_5 | 0.5136 | 0.4778 | -0.0357 | 0.5479 | 0.3899 | -0.1580 | 0.5040 | 0.5010 | -0.0030 |

XGBoost y LightGBM no tienen resultado inicial previo, por lo que se analizan como modelos adicionales ajustados.

## Modelos adicionales ajustados

| Dataset | Modelo | Variante | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| base | xgboost | tuned_fixed_0_5 | 0.4986 | 0.5356 | 0.5235 | 0.5295 | 0.4987 |
| hybrid | xgboost | tuned_fixed_0_5 | 0.5014 | 0.5403 | 0.5000 | 0.5194 | 0.5026 |
| base | lightgbm | tuned_fixed_0_5 | 0.4869 | 0.5271 | 0.4656 | 0.4944 | 0.4962 |
| hybrid | lightgbm | tuned_fixed_0_5 | 0.5045 | 0.5456 | 0.4815 | 0.5116 | 0.5119 |

## Comparación base vs híbrido tras tuning

Con umbral fijo 0.5:

- `logistic_regression`: el modelo base sigue siendo mejor que el híbrido.
- `random_forest`: el modelo híbrido empeora frente al base.
- `hist_gradient_boosting`: el híbrido mejora accuracy, F1 y ROC-AUC frente al base.
- `xgboost`: el híbrido mejora ligeramente accuracy y ROC-AUC, pero empeora F1.
- `lightgbm`: el híbrido mejora accuracy, F1 y ROC-AUC frente al base.

Los mejores resultados híbridos con umbral 0.5 son:

- `lightgbm`: ROC-AUC 0.5119, F1 0.5116.
- `hist_gradient_boosting`: ROC-AUC 0.5087, F1 0.5487.

El mejor ROC-AUC global del bloque tuned es el de `random_forest` base con 0.5125, aunque el mejor resultado híbrido queda muy cerca con LightGBM.

Con umbral optimizado dentro del rango 0.40-0.60, el F1 sube en varios modelos. Aun así, esta variante aumenta mucho el recall y genera más falsos positivos, por lo que se considera un análisis secundario y no el resultado principal.

## Conclusión

La validación cruzada temporal mejora la calidad metodológica del experimento y permite ajustar hiperparámetros sin fuga temporal. No obstante, no produce una mejora fuerte y generalizada del enfoque híbrido.

El ajuste confirma que el siguiente paso más prometedor no es seguir bajando umbrales, sino mejorar la representación temporal del sentimiento. Tiene sentido añadir variables retardadas y ventanas móviles de noticias para capturar que una noticia puede afectar no solo al día de publicación, sino también a sesiones posteriores.
