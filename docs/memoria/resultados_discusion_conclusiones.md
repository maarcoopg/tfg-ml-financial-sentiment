# Resultados, discusión y conclusiones

> Estado de revisión: las tablas de este documento corresponden al experimento inicial y se conservan como antecedente. El protocolo corregido y sus limitaciones se describen en `docs/revision_implementation.md`; los resultados nuevos se documentan en `docs/resultados_revision_temporal.md`. El histórico utilizado ya se ha inspeccionado y no constituye un holdout nuevo.

## Resultados globales

La comparación entre el modelo financiero base y el modelo híbrido se realizó sobre el mismo conjunto de activos: AAPL, MSFT, NVDA y TSLA. SPY no se incluyó en los entrenamientos para mantener una comparación homogénea.

| Modelo | Accuracy base | Accuracy híbrido | Delta accuracy | F1 base | F1 híbrido | Delta F1 | ROC-AUC base | ROC-AUC híbrido | Delta ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dummy | 0.5389 | 0.5389 | 0.0000 | 0.7004 | 0.7004 | 0.0000 | 0.5000 | 0.5000 | 0.0000 |
| hist_gradient_boosting | 0.4855 | 0.5059 | 0.0203 | 0.4851 | 0.5183 | 0.0332 | 0.4930 | 0.5054 | 0.0124 |
| logistic_regression | 0.5316 | 0.5077 | -0.0240 | 0.6699 | 0.5870 | -0.0828 | 0.5115 | 0.5011 | -0.0103 |
| random_forest | 0.5095 | 0.5136 | 0.0041 | 0.5780 | 0.5479 | -0.0301 | 0.5124 | 0.5040 | -0.0084 |

El modelo de referencia `dummy` obtiene una accuracy de 0.5389 porque predice siempre la clase mayoritaria. Este resultado muestra que la distribución del objetivo está ligeramente desbalanceada hacia sesiones alcistas, por lo que el F1 de este modelo debe interpretarse con cautela: alcanza 0.7004 a costa de no identificar ninguna observación de la clase negativa.

## Comparación base vs híbrido

La incorporación de sentimiento no mejora de forma uniforme todos los modelos. El resultado más favorable aparece en `hist_gradient_boosting`, donde el enfoque híbrido mejora la accuracy en 0.0203, el F1 en 0.0332 y el ROC-AUC en 0.0124.

En `random_forest`, el modelo híbrido mejora ligeramente la accuracy, pero empeora F1 y ROC-AUC. Esto sugiere que el aumento de aciertos globales no se traduce necesariamente en una mejor separación entre clases.

En `logistic_regression`, el modelo financiero base obtiene mejores resultados que el híbrido. La caída del recall y del F1 indica que la incorporación de variables de sentimiento no mejora el comportamiento del modelo lineal en este experimento.

## Importancia de variables

El análisis de importancia muestra que las variables financieras siguen siendo dominantes. En `random_forest` híbrido, las variables financieras concentran aproximadamente el 75% de la importancia agregada y las variables de sentimiento el 25%. En regresión logística híbrida, las variables de sentimiento también tienen peso, aunque inferior al bloque financiero.

Entre las variables financieras destacan `daily_return`, `RSI`, `volatility_20`, `MACD`, `MACD_signal` y variables de precio. Entre las variables de sentimiento aparecen variables como `sentiment_mean`, `sentiment_median` y `sentiment_max`.

## Discusión

Los resultados muestran diferencias puntuales entre algoritmos, pero no acreditan por sí solos que el sentimiento contenga una señal predictiva robusta. La capacidad de los métodos no lineales para representar interacciones es una motivación del experimento, no una explicación demostrada de las diferencias observadas. Deben considerarse la incertidumbre, la cobertura de noticias y la disponibilidad temporal de las variables.

También debe tenerse en cuenta que la predicción diaria de dirección bursátil es una tarea ruidosa. Una parte importante de los movimientos de precio puede depender de factores no incluidos en el dataset, como resultados empresariales, cambios macroeconómicos, política monetaria, eventos geopolíticos o expectativas de mercado no capturadas por las noticias descargadas.

El uso de sentimiento diario agregado simplifica además la información textual. Esta agregación facilita el modelado, pero puede perder matices relevantes como la hora exacta de publicación, la relevancia de cada fuente, la novedad de la noticia o el retraso con el que el mercado incorpora la información.

## Conclusiones

El proyecto permite construir un pipeline completo y reproducible que integra datos financieros, indicadores técnicos y sentimiento financiero para un problema de clasificación binaria de tendencia bursátil.

La comparación empírica no permite afirmar que el sentimiento mejore siempre al modelo financiero base. La mejora más clara aparece en `hist_gradient_boosting`, mientras que otros modelos muestran mejoras parciales o empeoramientos.

La comparación inicial no demuestra una mejora robusta atribuible al sentimiento. La utilidad de representaciones alternativas debe tratarse como una hipótesis experimental. Ni la importancia interna de una variable ni un AUC puntual ligeramente superior a 0,5 prueban capacidad predictiva independiente o rentabilidad.

## Limitaciones y líneas futuras

Como limitaciones principales destacan:

- Uso de sentimiento agregado diario en lugar de embeddings o análisis textual más profundo.
- En el experimento inicial faltaba validación temporal con múltiples ventanas; la revisión posterior incorpora validación anidada y purga. Sigue pendiente una confirmación en un periodo realmente no utilizado.
- Evaluación limitada a cuatro empresas tecnológicas de gran capitalización.
- Exclusión de SPY del entrenamiento principal por falta de sentimiento corporativo comparable.

Como líneas futuras, se podrían incorporar modelos de lenguaje especializados en finanzas, validación walk-forward, nuevas variables macroeconómicas y análisis del impacto horario de las noticias respecto a la apertura y cierre del mercado.
