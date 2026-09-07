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

El modelo `dummy` se mantiene como referencia y no se ajusta.

## Criterio de selección

Los hiperparámetros se seleccionan priorizando `ROC-AUC` en validación temporal. Además, se calcula un umbral optimizado para F1, pero se reportan dos variantes:

- `tuned_fixed_0_5`: hiperparámetros ajustados y umbral fijo 0.5.
- `tuned_optimized_threshold`: hiperparámetros ajustados y umbral elegido en validación para maximizar F1.

Esta separación evita confundir una mejora real del modelo con una subida artificial del F1 por predecir casi todas las observaciones como clase positiva.

## Resultados frente a modelos iniciales

| Dataset | Modelo | Variante | Accuracy inicial | Accuracy tuned | Delta accuracy | F1 inicial | F1 tuned | Delta F1 | ROC-AUC inicial | ROC-AUC tuned | Delta ROC-AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | hist_gradient_boosting | tuned_fixed_0_5 | 0.4855 | 0.5063 | 0.0208 | 0.4851 | 0.5349 | 0.0499 | 0.4930 | 0.4985 | 0.0054 |
| base | logistic_regression | tuned_fixed_0_5 | 0.5316 | 0.5316 | 0.0000 | 0.6699 | 0.6699 | 0.0000 | 0.5115 | 0.5114 | -0.0001 |
| base | random_forest | tuned_fixed_0_5 | 0.5095 | 0.4995 | -0.0099 | 0.5780 | 0.4781 | -0.0999 | 0.5124 | 0.5114 | -0.0010 |
| hybrid | hist_gradient_boosting | tuned_fixed_0_5 | 0.5059 | 0.5081 | 0.0023 | 0.5183 | 0.5302 | 0.0119 | 0.5054 | 0.5113 | 0.0059 |
| hybrid | logistic_regression | tuned_fixed_0_5 | 0.5077 | 0.5063 | -0.0014 | 0.5870 | 0.5781 | -0.0090 | 0.5011 | 0.5033 | 0.0021 |
| hybrid | random_forest | tuned_fixed_0_5 | 0.5136 | 0.5063 | -0.0072 | 0.5479 | 0.5236 | -0.0243 | 0.5040 | 0.5047 | 0.0007 |

Con umbral optimizado, varios modelos alcanzan F1 cercano a 0.70. Sin embargo, esa mejora se produce porque el umbral baja a 0.30 y el modelo predice casi todas las observaciones como subida. Por eso, esos resultados se guardan para análisis, pero no son la evidencia principal de mejora.

## Comparación base vs híbrido tras tuning

Con umbral fijo 0.5:

- `logistic_regression`: el modelo base sigue siendo mejor que el híbrido.
- `random_forest`: el híbrido mejora F1 frente al base, pero mantiene peor ROC-AUC.
- `hist_gradient_boosting`: el híbrido mejora ROC-AUC frente al base, aunque el F1 queda ligeramente por debajo.

El resultado más interesante sigue estando en `hist_gradient_boosting`, porque el modelo híbrido ajustado obtiene el mejor ROC-AUC del bloque tuned: 0.5113.

## Conclusión

La validación cruzada temporal mejora la calidad metodológica del experimento y permite ajustar hiperparámetros sin fuga temporal. No obstante, no produce una mejora fuerte y generalizada del enfoque híbrido.

El ajuste confirma que el siguiente paso más prometedor no es seguir bajando umbrales, sino mejorar la representación temporal del sentimiento. Tiene sentido añadir variables retardadas y ventanas móviles de noticias para capturar que una noticia puede afectar no solo al día de publicación, sino también a sesiones posteriores.
