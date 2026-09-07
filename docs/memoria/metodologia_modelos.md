# Metodología de modelado y evaluación

## Planteamiento del problema

El objetivo del modelado es predecir la dirección de la siguiente sesión bursátil. La tarea se formula como un problema de clasificación binaria, donde la clase positiva representa una subida del precio ajustado de cierre en la siguiente sesión.

## Diseño experimental

Se comparan dos configuraciones:

- Modelo financiero base: entrenado únicamente con precios, volumen, retornos e indicadores técnicos.
- Modelo híbrido: entrenado con las mismas variables financieras y variables de sentimiento financiero agregadas por día.

Ambos enfoques se entrenan y evalúan sobre las mismas fechas, los mismos tickers y la misma variable objetivo. Esto permite atribuir las diferencias de rendimiento a la incorporación de variables de sentimiento, no a cambios en el universo de datos.

SPY queda fuera del entrenamiento y de la comparación principal porque se trata de un ETF y no dispone de una señal de sentimiento corporativo comparable a la de AAPL, MSFT, NVDA y TSLA.

## División temporal

La división entre entrenamiento y prueba se realiza cronológicamente. El 80% inicial de fechas únicas se asigna al entrenamiento y el 20% final a prueba.

Este enfoque evita la fuga temporal de información que podría aparecer con una división aleatoria. En series financieras, mezclar observaciones futuras en el entrenamiento produciría una evaluación demasiado optimista.

## Modelos candidatos

Se entrenan cuatro modelos:

- `dummy`: referencia basada en la clase mayoritaria.
- `logistic_regression`: modelo lineal interpretable para clasificación binaria.
- `random_forest`: ensamblado de árboles de decisión con relaciones no lineales.
- `hist_gradient_boosting`: modelo de boosting para datos tabulares.

El modelo `dummy` se utiliza como referencia mínima. Los modelos restantes permiten comparar un enfoque lineal, un ensamblado basado en bagging y un método de boosting.

## Preprocesamiento del entrenamiento

Los modelos se implementan mediante pipelines de scikit-learn. Todos incluyen imputación por mediana para asegurar robustez ante valores ausentes. La regresión logística incluye además escalado estándar, necesario para estabilizar la estimación de coeficientes.

Cuando el estimador lo permite, se fija una semilla aleatoria igual a `42` para mejorar la reproducibilidad.

## Métricas

La evaluación utiliza:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Matriz de confusión.
- ROC-AUC cuando el modelo proporciona probabilidades.

Además de la evaluación global, se calculan métricas por ticker para comprobar si el rendimiento es homogéneo entre empresas.

## Comparación base vs híbrido

La comparación se realiza emparejando cada modelo base con su equivalente híbrido. Para cada métrica se calcula:

```text
delta = métrica híbrida - métrica base
```

Un delta positivo indica que la incorporación de sentimiento mejora esa métrica para el algoritmo correspondiente. Un delta negativo indica que el modelo financiero base obtiene mejor rendimiento.

## Interpretabilidad

Se analiza la importancia de variables en los modelos que ofrecen una señal interpretable directa:

- Coeficientes absolutos en regresión logística.
- Importancias internas en random forest.

Este análisis permite observar el peso relativo de las variables financieras y de sentimiento, aunque no debe interpretarse como evidencia causal.

## Limitaciones metodológicas

El mercado financiero incorpora ruido, cambios de régimen y eventos no observables en el dataset. Además, el sentimiento agregado puede no capturar correctamente el tono, relevancia o impacto temporal de cada noticia. Por ello, los resultados deben interpretarse como evidencia empírica dentro del diseño experimental definido, no como una garantía de capacidad predictiva generalizable.
