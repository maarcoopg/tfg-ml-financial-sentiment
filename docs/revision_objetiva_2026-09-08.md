# Revisión objetiva del proyecto

Fecha: 8 de septiembre de 2026. Rama revisada: `feature/lagged-sentiment-features`, incluidos los cambios locales de la issue 45. Último commit: `b56d503`.

## Valoración

El proyecto tiene una base funcional y comprensible para un TFG experimental. La generación de datos, el entrenamiento y el cálculo de métricas presentan varias decisiones correctas y las comprobaciones sobre los artefactos actuales han pasado. Sin embargo, todavía no demuestra una mejora predictiva robusta atribuible al sentimiento. Tampoco demuestra utilidad económica.

El principal trabajo pendiente es reforzar el diseño experimental, la calidad temporal de las noticias y la reproducibilidad. Ampliar la búsqueda de hiperparámetros antes de resolver esto puede mejorar cifras observadas sin mejorar la evidencia.

## Hallazgos prioritarios

### 1. Alta: no existe un instante de predicción coherente y verificable

En `src/data/align_news_to_trading_days.py:41`, la unión utiliza `published_date`, no `published_at`. Una noticia publicada después del cierre se asigna a la misma fecha que otra publicada antes. No hay tratamiento explícito de zonas horarias ni horas de cierre.

Si se pretende predecir al cierre de la sesión t, se están incorporando noticias que aún no estarían disponibles. Si se pretende predecir después de recibir todas las noticias de ese día, el objetivo cierre(t)-cierre(t+1) incluye un tramo que ya ha transcurrido al emitir la predicción. Esto impide interpretar directamente el experimento como una estrategia ejecutable.

Acción: definir primero la hora de decisión, normalizar timestamps y construir ventanas de información coherentes con ella y con el objetivo. Incluir festivos y cierres anticipados. No basta con desplazar todas las noticias un día indiscriminadamente.

La CV tampoco purga etiquetas en sus fronteras: `src/models/tune_temporal_cv.py:139` termina el entrenamiento justo antes del primer día de validación, pero la última etiqueta de entrenamiento depende del cierre de ese primer día. Esto sería información futura si el ajuste se realiza antes de comenzar la validación; si se realiza después del cierre de ese día, puede estar disponible. Debe documentarse ese instante y, para un protocolo de separación estricta, excluir las etiquetas cuyo horizonte alcance la validación. La misma consideración aplica al corte final de entrenamiento y prueba.

### 2. Alta: las mejoras no están demostradas estadísticamente

`src/models/evaluation.py:29` calcula métricas puntuales correctamente, pero no su incertidumbre ni la incertidumbre de las diferencias entre modelos. Afirmar que existe señal por observar AUC > 0,5 es demasiado fuerte.

Se ha realizado en esta revisión un bootstrap exploratorio por bloques consecutivos de fechas. Cada fecha conserva juntas las cuatro empresas; los mismos índices se aplican a ambos modelos. Se utilizan 1.000 réplicas por longitud de bloque, semilla 42 e intervalos percentiles del 95 %.

| Bloque, sesiones | IC AUC LightGBM con retardos | IC diferencia AUC frente a LightGBM híbrido sin retardos |
| --- | --- | --- |
| 5 | [0,4981; 0,5474] | [-0,0173; 0,0414] |
| 20 | [0,5022; 0,5448] | [-0,0185; 0,0390] |
| 60 | [0,4949; 0,5436] | [-0,0226; 0,0402] |

La diferencia observada es +0,0098, pero todos los intervalos de la diferencia incluyen cero. La evidencia frente a AUC 0,5 depende de la longitud de bloque. Estos resultados no prueban ausencia de señal; muestran que la mejora no es concluyente con este análisis. Además, los intervalos son exploratorios y no corrigen la selección del mejor modelo entre múltiples experimentos ni la reutilización del test.

El ajuste de parámetros del código usa únicamente train, lo cual es correcto. Sin embargo, las decisiones sucesivas de ampliar modelos, parámetros y retardos se han tomado después de observar resultados del mismo test. Ese periodo ya sirve como referencia de desarrollo y pierde fuerza como evaluación final independiente. Hace falta una evaluación externa temporal o un nuevo periodo realmente no utilizado. Una validación anidada ayuda a estructurar futuras comparaciones, pero no borra el conocimiento adquirido sobre el test actual.

### 3. Alta: cambio importante de cobertura de noticias

La tabla siguiente se obtiene de los datasets de modelado actuales; cada fila representa una empresa y una sesión.

| Año | Filas | Filas con noticias | Noticias |
| --- | ---: | ---: | ---: |
| 2015 | 1.008 | 329 | 414 |
| 2021 | 1.008 | 912 | 3.306 |
| 2024 | 1.008 | 953 | 3.675 |
| 2025 | 992 | 972 | 27.381 |

El entrenamiento termina el 13/10/2023 y el test empieza el 16/10/2023. Por tanto, el gran salto de volumen de 2025 aparece en prueba. No se ha demostrado si se debe a cobertura del proveedor, cambios de fuentes, mayor actividad informativa o una combinación. No puede asumirse que la falta de registros antiguos equivale a ausencia real de noticias.

`src/data/handle_days_without_news.py:40` distingue días con registros mediante `has_news`, lo cual es útil, pero no distingue ausencia real de noticias de falta de cobertura de la fuente. Las variables de volumen pueden aprender cambios del archivo disponible.

Acción: auditar cobertura por empresa, año y fuente; contrastar periodos comparables; definir criterios de cobertura antes de medir mejoras. Probar volumen relativo a su historia y separar cobertura desconocida de ausencia de noticias cuando exista información para hacerlo. No eliminar años solo porque empeoren la métrica.

### 4. Media: F1 y accuracy no acreditan una ventaja sobre el baseline

| Modelo | Accuracy | F1 positivo | ROC-AUC |
| --- | ---: | ---: | ---: |
| Siempre predice sube, baseline existente | 0,5389 | 0,7004 | 0,5000 |
| Regresión logística base inicial | 0,5316 | 0,6699 | 0,5115 |
| LightGBM híbrido ajustado | 0,5045 | 0,5116 | 0,5119 |
| LightGBM híbrido con retardos ajustado | 0,5122 | 0,5422 | 0,5217 |

El test tiene 1.192 positivos y 1.020 negativos. El F1 positivo recompensa mucho predecir subidas en esta distribución; el baseline obtiene 0,7004 sin detectar ninguna bajada. El umbral se optimiza para ese F1 en `src/models/tune_temporal_cv.py:148`; en LightGBM con retardos, el umbral 0,4 produce 1.906 predicciones positivas de 2.212.

Acción: conservar el baseline en todas las comparaciones, informar balanced accuracy, MCC o F1 macro junto con AUC y matrices de confusión, y fijar de antemano el criterio de selección. La balanced accuracy del LightGBM con retardos a umbral 0,5 es 0,5102.

La variación entre empresas y periodos también importa: el AUC de ese modelo es 0,4831 en AAPL, 0,5682 en MSFT, 0,5040 en NVDA y 0,5278 en TSLA. Por año resulta 0,4820 en el tramo de 2023, 0,4933 en 2024 y 0,5469 en 2025. No es una mejora uniforme y estas particiones tampoco demuestran por sí solas significación.

### 5. Media: hay fallos concretos de reproducibilidad y presentación

- `src/models/tune_temporal_cv.py:450`: una ejecución parcial sobrescribe los CSV globales con solo lo ejecutado, mientras que permanecen modelos y predicciones de otras ejecuciones. Se pueden mezclar artefactos antiguos con resúmenes nuevos. Repetir solo `lagged_hybrid` puede perder la referencia híbrida necesaria para comparaciones posteriores. Usar directorios por ejecución y un manifiesto con datos, configuración, versiones y commit.
- `src/data/split_train_test.py:124`: reconstruye `feature_sets.json` sin las claves de retardos. Volver a ejecutar el split después de generar retardos deja sus CSV en disco pero hace que su carga falle hasta repetir la generación de retardos. Separar configuración de características y resultados generados, o reconstruirlos de forma consistente.
- `src/data/download_financial_news.py:150`: un resultado vacío genera un CSV sin columnas. Su posterior lectura provoca `EmptyDataError`. Se ha reproducido en memoria; no hay chunks de ese tipo en los archivos actuales. Guardar siempre el esquema y probar descargas vacías y reanudaciones. Tampoco hay reintentos con espera progresiva; un límite de resultados alcanzado en un único día no se resuelve subdividiendo por horas, por lo que el descargador no garantiza exhaustividad en ese caso.
- `src/models/train_models.py:127`: LightGBM configura `subsample=0.8`, pero conserva `subsample_freq=0`, que según la documentación instalada desactiva el muestreo de filas. Ese 80 % no está teniendo el efecto esperado. Activarlo explícitamente si se quiere evaluar ese mecanismo y repetir la comparación correspondiente.
- `src/visualization/plot_results.py:40`: el primer gráfico pasa accuracy, precision, recall, F1 y AUC a una única barra por modelo y dataset. El agregador calcula su media y representa una mezcla sin interpretación definida. La figura posterior por facetas sí separa métricas. Eliminar o corregir el primer gráfico.
- `src/visualization/plot_results.py:10` y `src/models/analyze_feature_importance.py:10`: siguen leyendo las salidas iniciales, no las ajustadas ni las de retardos. La memoria también mantiene resultados iniciales y afirma ausencia de validación temporal en `docs/memoria/resultados_discusion_conclusiones.md:51`, pese a estar ya implementada.

### 6. Media: faltan controles automáticos que protejan el experimento

No se ha encontrado una suite de pruebas ni un workflow de CI. Varias comprobaciones se imprimen por consola sin detener la ejecución si se incumplen. Por ejemplo, el merge financiero-sentimiento no valida la unicidad de su relación y distintos scripts descartan o rellenan valores inválidos sin un informe formal de incidencias.

Acción: añadir pruebas de alineación horaria, fronteras de etiquetas, independencia entre tickers al calcular retardos, equivalencia de filas entre enfoques, duplicados en uniones, rangos de scores y reproducibilidad de métricas. Centralizar tickers, fechas, horizonte, rutas y esquema. Un comando de pipeline con dependencias explícitas reduciría errores de orden entre scripts que sobrescriben archivos intermedios.

## Puntos buenos verificados

- Separación comprensible entre datos, modelos, evaluación y visualización. Las funciones de construcción de modelos y cálculo de métricas se reutilizan.
- El objetivo corresponde a la siguiente sesión y `next_adj_close` queda fuera de las variables predictoras. No se han encontrado discrepancias entre objetivos guardados y precios brutos actuales.
- Train/test y folds se dividen por fechas completas, manteniendo juntas las empresas de cada sesión. No se utiliza una partición aleatoria de filas.
- La imputación y el escalado están dentro de pipelines que se ajustan en cada entrenamiento. No se calcula el escalado sobre todo el dataset antes de la CV.
- Base, híbrido y retardos utilizan exactamente las mismas filas y objetivos; SPY está excluido en los artefactos de modelado actuales.
- Los retardos se calculan por empresa y coinciden al recalcularlos. Las ventanas móviles incluyen la sesión actual: esto es correcto si sus datos están disponibles al emitir la predicción; depende de resolver el punto temporal anterior.
- Hay cinco algoritmos predictivos y un baseline trivial. La diversidad es suficiente para una comparación inicial; la falta de algoritmos no es el principal problema identificado.
- Las métricas incluyen matriz de confusión, AUC y resultados por empresa. Las 30 tablas de métricas ajustadas coinciden con sus predicciones al recalcular accuracy, F1 y AUC.
- Dependencias fijadas, semilla explícita, modelos y predicciones guardados, documentación por etapas y un historial reciente de ramas de funcionalidad y merges identificables. Son bases útiles, aunque no equivalen a reproducibilidad completa desde una descarga nueva.

## Mejoras experimentales después de corregir la evaluación

1. Evaluar hipótesis pequeñas mediante validación temporal externa: baseline, solo precios, solo volumen de noticias, solo sentimiento y combinación. Esto separa el efecto de tener noticias del tono de las noticias.
2. Comparar retardos y ventanas por separado con un conjunto reducido. Actualmente se añaden 65 variables de una vez, de 27 a 92; así no se puede atribuir el cambio a un retardo concreto. No hay evidencia suficiente para afirmar que ese aumento produzca sobreajuste, pero sí aumenta la complejidad y dificulta la interpretación.
3. Probar relevancia por empresa y novedad. Se descarga `relevance_score`, pero la agregación de `src/data/aggregate_daily_sentiment.py:48` usa medias sin ponderarla. Evitar que múltiples versiones de un mismo evento dominen el día.
4. Probar representaciones financieras relativas: distancia porcentual a medias, MACD relativo al precio y volumen relativo a su historia. Los niveles absolutos de precios y medias cambian con el tiempo y entre empresas; el modelo conjunto no incorpora ticker como característica. Comparar explícitamente modelo conjunto y por empresa, sin asumir cuál será mejor.
5. El sentimiento actual procede de Alpha Vantage: `src/data/extract_news_sentiment.py:27` copia el score del proveedor. Es una integración legítima, pero no un clasificador textual entrenado en este proyecto. Una muestra anotada permitiría evaluar su calidad; un modelo financiero de texto alternativo sería un experimento posterior, no una mejora garantizada.
6. Horizontes de varias sesiones o predicción de movimientos relevantes son hipótesis posibles, pero cambian el problema. Exigen nuevas etiquetas, purga acorde al horizonte y comparaciones separadas; no deben elegirse por el resultado del test actual.
7. Si se pretende afirmar utilidad económica, añadir una evaluación de ejecución con precios disponibles al operar, costes, rotación, exposición y drawdown. Las métricas de clasificación actuales no permiten concluir rentabilidad.

## Alcance de la revisión

Se revisaron scripts de datos, entrenamiento, CV, evaluación, importancia de variables, visualización, documentación, dependencias, inventario de pruebas y el historial reciente de Git. Se comprobó la sintaxis de todos los Python bajo `src` sin modificar código.

Se verificaron 8.844 filas de entrenamiento, 2.212 de prueba, equivalencia entre enfoques, objetivos frente a precios, ausencia de duplicados ticker-fecha y de nulos en el dataset híbrido, valores finitos del test con retardos, recálculo de retardos y consistencia de 30 salidas de métricas ajustadas. Se calcularon los intervalos exploratorios descritos anteriormente a partir de predicciones guardadas.

No se descargaron datos nuevos, no se reentrenaron modelos, no se corrigió código y no se realizaron commits, merges ni cambios en GitHub. La revisión valida los artefactos disponibles, no la completitud del histórico del proveedor ni una ejecución de extremo a extremo desde cero.
