# Modelo híbrido de aprendizaje automático para la predicción de tendencias bursátiles mediante análisis técnico y sentimiento financiero

**Borrador de la memoria del Trabajo de Fin de Grado**  
Grado en Ingeniería del Software · Universidad de Sevilla  
**Autor:** Marco Padilla Gómez  
**Tutor:** Jose Antonio Troyano Jimenez  
**Revisión documental:** 19 de septiembre de 2026, según las orientaciones del tutor.

> Este documento describe la implementación y los experimentos existentes. No presenta como realizadas las propuestas futuras. La revisión principal corresponde a `review-full-20260908` y se amplía con `per-company-20260916`, `ticker-aware-full-20260916`, `first-round-full-20260916` y `variable-ablation-full-20260917`. Se distinguen sus métricas y todos los resultados se consideran exploratorios, al reutilizar un histórico ya examinado.

## Índice

1. [Introducción y objetivos](#1-introducción-y-objetivos)
2. [Planificación](#2-planificación)
3. [Estado del arte y fundamentos teóricos](#3-estado-del-arte-y-fundamentos-teóricos)
4. [Fuentes de datos y procesamiento](#4-fuentes-de-datos-y-procesamiento)
5. [Diseño experimental y resultados](#5-diseño-experimental-y-resultados)
6. [Especificación de requisitos](#6-especificación-de-requisitos)
7. [Análisis del sistema](#7-análisis-del-sistema)
8. [Conclusiones](#8-conclusiones)
9. [Bibliografía](#9-bibliografía)
10. [Anexos](#10-anexos)

## 1. Introducción y objetivos

### 1.1 Resumen

Este trabajo estudia si incorporar sentimiento de noticias financieras aporta información predictiva adicional a un modelo basado en precios, volumen e indicadores técnicos. La tarea consiste en clasificar la dirección del precio ajustado de cierre de la siguiente sesión, no en estimar un precio exacto ni en desarrollar un sistema de negociación real.

Se construye un flujo de procesamiento modular en Python para integrar datos de Yahoo Finance, obtenidos mediante `yfinance`, y noticias con sentimiento por empresa de Alpha Vantage. El universo de modelado comprende Apple, Microsoft, NVIDIA y Tesla durante el histórico de 2015 a 2025. SPY se descargó inicialmente como referencia, pero no participa en los entrenamientos.

El desarrollo evolucionó desde una partición temporal simple hacia una evaluación anidada con entrenamiento expansivo, purga del horizonte de las etiquetas y asignación de noticias según el cierre real del mercado. Se compararon cinco algoritmos predictivos y una referencia de clase mayoritaria. El experimento corregido comprende doce variantes de variables, tres bloques externos y tres particiones internas: 1.080 ajustes internos y 183 entrenamientos externos.

El mayor ROC-AUC agrupado observado fue 0,5108, con HistGradientBoosting y variables relativas, con un intervalo exploratorio del 95 % de [0,4855; 0,5406]. La comparación principal entre LightGBM híbrido con y sin retardo de una sesión presentó una diferencia de AUC de aproximadamente −0,0008, cuyo intervalo incluyó cero. Por tanto, no se demuestra una ventaja predictiva robusta del sentimiento con los datos, representaciones y protocolo utilizados.

La contribución del proyecto es tanto experimental como de ingeniería: integración de fuentes heterogéneas, control temporal explícito, evaluación comparable y conservación verificable de resultados. El histórico ya había sido consultado, por lo que los resultados deben considerarse exploratorios y no una confirmación independiente.

Dos experimentos adicionales estudian si conviene especializar el aprendizaje por empresa o conservar el entrenamiento conjunto incorporando identidad, variables relativas e interacciones. Entrenar por separado no mejora el promedio general. Las variables relativas producen pequeñas mejoras descriptivas, pero ninguno de los 81 intervalos de diferencias de AUC macro del segundo experimento excluye cero. Destaca el bosque aleatorio híbrido en NVDA: pasa de 0,5180 a 0,5717 de AUC al utilizar relativas e identificador, aunque sin identificador ya obtiene 0,5711. Esta señal no se generaliza a todas las empresas ni demuestra rentabilidad.

Una ronda posterior separa deduplicación, ventanas de entrenamiento, nuevas variables de sentimiento, criterio de selección y búsqueda ampliada. La combinación completa no mejora el AUC macro general. Una ablación individual identifica resultados puntuales: la sorpresa del volumen mejora descriptivamente los tres híbridos sin retardo, y la intensidad absoluta eleva el AUC macro del bosque híbrido de 0,5045 a 0,5202. No se acredita una mejora robusta tras considerar la incertidumbre, las comparaciones múltiples y la reutilización de las fechas evaluadas.

**Palabras clave:** aprendizaje automático, sentimiento financiero, series temporales, clasificación binaria, validación temporal, reproducibilidad.

### 1.2 Motivación

Los precios y las noticias representan fuentes de información diferentes. Los primeros resumen movimientos ya observados; las segundas contienen mensajes sobre empresas, expectativas y acontecimientos. La motivación del proyecto consiste en estudiar si una representación numérica de esas noticias ayuda a clasificar movimientos posteriores cuando se añade a un conjunto financiero común.

La existencia de noticias relacionadas con un movimiento no implica que su sentimiento permita anticiparlo. Una noticia puede describir un cambio de precio que ya ocurrió, publicarse cuando parte de la información ya se ha incorporado al mercado o resultar relevante a un horizonte distinto del estudiado. Por ello, la disponibilidad temporal y la comparación con referencias sencillas forman parte del problema, no son detalles secundarios.

Desde Ingeniería del Software, el trabajo requiere resolver adquisición de datos, validación, transformación, contratos entre módulos, almacenamiento de experimentos y comunicación de resultados. Una mejora aparente obtenida con una evaluación incorrecta no cumpliría el objetivo académico, aunque mostrase una métrica atractiva.

### 1.3 Objetivos, hipótesis y alcance

#### 1.3.1 Pregunta de investigación

¿Mejora la incorporación de sentimiento financiero la capacidad de discriminar subidas y no subidas de la siguiente sesión respecto a utilizar exclusivamente información financiera histórica?

#### 1.3.2 Objetivos específicos

1. Construir conjuntos de datos financieros y de noticias trazables por empresa y fecha.
2. Definir una etiqueta y variables coherentes con el instante de predicción.
3. Comparar configuraciones base e híbrida sobre las mismas observaciones.
4. Seleccionar hiperparámetros sin utilizar el bloque externo correspondiente.
5. Investigar si retardos, ventanas, relevancia, representaciones relativas y especialización por empresa modifican los resultados.
6. Cuantificar incertidumbre y documentar los límites de las conclusiones.
7. Proporcionar programas, pruebas y cuadernos de análisis que permitan inspeccionar y reproducir el experimento.

La hipótesis principal plantea una posible contribución adicional del sentimiento. Una hipótesis secundaria plantea persistencia durante varias sesiones. También se estudia si entrenar un modelo independiente por empresa mejora la adaptación a sus características, y si un modelo conjunto con identidad e interacciones puede conservar información compartida sin ignorar esas diferencias. Las comparaciones realizadas no establecen una ventaja general confirmada; el resultado favorable en NVDA con variables relativas requiere validación independiente.

#### 1.3.3 Alcance y exclusiones

La unidad de observación es una pareja empresa-sesión. La revisión inicial utiliza modelos conjuntos para las cuatro empresas sin introducir el identificador bursátil como predictor. Los experimentos adicionales comparan modelos independientes y modelos conjuntos con indicadores binarios de empresa, variables relativas e interacciones. La clasificación es diaria y binaria. No se desarrolla predicción intradía, una política de inversión, ejecución de órdenes ni una simulación económica retrospectiva. Tampoco se entrena un modelo propio de lenguaje: el sentimiento empleado procede del proveedor.

## 2. Planificación

### 2.1 Organización y estimación del esfuerzo

El trabajo se organiza en diez paquetes que combinan investigación, implementación, experimentación y comunicación. La tabla propone una distribución global de **330 horas**, dentro del intervalo de 300–360 indicado por el tutor. Es una estimación de planificación reconstruida para este borrador, **no un registro certificado de horas efectivamente invertidas**. El historial de Git acredita entregas, pero no mide dedicación. Marco deberá contrastar y corregir las estimaciones antes de entregar la memoria; el tiempo de cálculo desatendido no se suma automáticamente como trabajo personal.

| Paquete | Actividades y entregable | Horas estimadas | Situación |
| --- | --- | ---: | --- |
| P1. Definición del problema | Objetivos, alcance, preguntas y requisitos | 25 | Base elaborada; revisión con tutor |
| P2. Estado del arte | Búsqueda, lectura crítica y comparación de publicaciones | 35 | Fundamentación inicial; ampliación pendiente |
| P3. Adquisición y auditoría | Precios, noticias, cobertura, duplicados y trazabilidad | 40 | Implementado y evaluado |
| P4. Preparación y variables | Calendario, etiqueta, indicadores, agregación y retardos | 25 | Implementado y evaluado |
| P5. Modelado inicial | Referencia mayoritaria, cinco algoritmos y flujo de entrenamiento | 35 | Implementado y evaluado |
| P6. Evaluación temporal | Validación interna, purga, ajuste e incertidumbre | 45 | Implementado y evaluado |
| P7. Experimentos adicionales | Empresas, representación, ventanas y ablaciones | 35 | Resultados documentados |
| P8. Ingeniería y pruebas | Contratos, pruebas, artefactos, diagramas y reproducibilidad | 25 | Implementado; documentación ampliada |
| P9. Análisis y memoria | Figuras, interpretación, redacción y revisiones | 50 | En curso |
| P10. Preparación de la defensa | Selección de evidencias, presentación y ensayo | 15 | Pendiente |
| **Total previsto** | **Trabajo realizado y pendiente, no horas acreditadas** | **330** | **Estimación pendiente de validación del autor** |

### 2.2 Dependencias y seguimiento

La dependencia principal es P1 → P2/P3 → P4 → P5 → P6 → P7. P8 y P9 acompañan a las demás tareas y P10 se realiza al consolidar la memoria. La revisión bibliográfica no se considera cerrada antes de programar: también sirve para interpretar resultados y revisar hipótesis durante el desarrollo.

Los hitos verificables son la disponibilidad del corpus, el primer modelo comparable, la corrección temporal, los experimentos por empresa, la ronda controlada y la memoria revisada. Se comprueban mediante código, pruebas y manifiestos, no mediante fechas de ejecución convertidas artificialmente en horas. Las issues y ramas registran decisiones; los informes mantienen resultados favorables y desfavorables. La aceptación de un experimento exige trazabilidad y comparación válida, no una mejora del AUC.

### 2.3 Riesgos y alcance pendiente

| Riesgo | Efecto | Medida aplicada o pendiente |
| --- | --- | --- |
| Cobertura incompleta o cambiante | Comparaciones sesgadas por disponibilidad | Auditoría temporal, conservación de entradas y límites explícitos |
| Fuga temporal | Resultados optimistas | Asignación al cierre, purga y preprocesamiento dentro del entrenamiento |
| Búsqueda reiterada sobre las mismas fechas | Selección de máximos espurios | Protocolos acotados y resultados rotulados como exploratorios |
| Coste de cómputo | Rejillas inviables | Candidatos prefijados, subconjunto de algoritmos en ampliaciones y artefactos reutilizables |
| Crecimiento del alcance | Memoria y desarrollo inconexos | Una hipótesis por ampliación y aprobación de su alcance antes de implementarla |
| Pérdida de datos locales | Reproducción incompleta | Manifiestos y conservación externa de entradas; los hashes no sustituyen una copia |

FinBERT, SHAP, otros horizontes, simulación económica y aplicación interactiva siguen siendo propuestas. No se incluyen como trabajo ejecutado ni se les asignan resultados ficticios. Si se aprueba una ampliación, deberá revisarse la planificación y la estimación de esfuerzo en lugar de mantener artificialmente el total anterior.

## 3. Estado del arte y fundamentos teóricos

### 3.1 Análisis técnico como representación

El análisis técnico se utiliza aquí para transformar el historial en variables tabulares: retornos, tendencia suavizada, oscilación y variabilidad. No se presupone que un indicador implique rentabilidad. La comparación experimental determina si estas representaciones resultan útiles en la tarea definida.

### 3.2 Sentimiento financiero e integración temprana

Una noticia puede recibir una puntuación de tono distinta para cada empresa mencionada. Por ello se utiliza la puntuación por empresa en lugar de asumir que el tono global del artículo es adecuado para todas sus empresas. El servicio de consulta de Alpha Vantage proporciona noticias y metadatos de sentimiento; su documentación describe los filtros temporales y por activos. [Alpha Vantage](https://www.alphavantage.co/documentation/#news-sentiment).

En este trabajo, «híbrido» significa concatenar variables financieras y variables de sentimiento antes del clasificador. No significa combinar dos redes neuronales ni construir un sistema multimodal entrenado de extremo a extremo. Se valoraron alternativas como diccionarios o FinBERT, pero no se implementaron como fuente operativa de los resultados presentados.

### 3.3 Aprendizaje supervisado tabular

La regresión logística constituye una referencia lineal; el bosque aleatorio (Random Forest) combina árboles; HistGradientBoosting, XGBoost y LightGBM representan enfoques de potenciación por gradiente, que combina modelos débiles de forma secuencial. XGBoost y LightGBM se incorporaron para ampliar la comparación de modelos tabulares, no porque su superioridad estuviera garantizada. Véanse los artículos originales de [XGBoost](https://arxiv.org/abs/1603.02754) y [LightGBM](https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html).

### 3.4 Evaluación temporal

La validación temporal mantiene el entrenamiento antes de la evaluación. La documentación de [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) ofrece una referencia sobre particiones ordenadas. La implementación del proyecto es propia: divide fechas completas y añade purga según `target_end`, manteniendo juntas las empresas de una sesión. No debe confundirse con aplicar directamente validación cruzada aleatoria con k particiones ni con una llamada literal a TimeSeriesSplit.

### 3.5 Hipótesis del mercado eficiente

La eficiencia informativa vincula los precios con la información disponible. Se distinguen información histórica de mercado, información pública e información también privada, asociadas a las formas débil, semifuerte y fuerte. El proyecto combina historial de precios y noticias públicas, pero no constituye por sí solo una prueba de ninguna de esas formas. La formulación depende además del modelo de rendimiento esperado y riesgo. [Lo, síntesis de la hipótesis del mercado eficiente](https://web.mit.edu/~alo/www/Papers/EMH_Final.pdf).

Como notación para este trabajo, sea $\mathcal{I}_t$ la información disponible y $\mu^{eq}_{t+1}$ el rendimiento esperado bajo un modelo de equilibrio. Una restricción de ausencia de rendimiento anormal predecible se representa por:

$$
\mathbb{E}[r_{t+1}-\mu^{eq}_{t+1}\mid\mathcal{I}_t]=0.
$$

La expresión no impone rendimiento esperado nulo ni probabilidad de subida igual a 0,5. Un AUC cercano a 0,5 en cuatro empresas no demuestra eficiencia; un AUC superior tampoco la refuta sin estudiar riesgo, costes y validez del protocolo. Son alcances diferentes: aquí se estima discriminación estadística sobre etiquetas diarias, no rendimiento anormal de una estrategia.

### 3.6 Hipótesis del mercado adaptativo

Lo propone interpretar los mercados mediante competencia, adaptación y selección, conciliando mecanismos de eficiencia y comportamiento. La eficacia de una regla puede depender del entorno y de los participantes, en lugar de ser estable en todo el histórico. [Lo (2004)](https://web.mit.edu/Alo/www/Papers/JPM2004.html).

Una representación conceptual útil, no una ecuación universal atribuida al autor ni un modelo estimado en este TFG, es:

$$
\Pr(y_{t+1}=1\mid X_t,Z_t)=f_{\theta_t}(X_t,Z_t),
$$

donde $Z_t$ resume el entorno y $\theta_t$ puede variar. Motiva estudiar bloques temporales y ventanas retrospectivas. Que una ventana corta empeore no refuta esta hipótesis: puede perder información útil o aumentar la variabilidad del ajuste. En los experimentos actuales no se estima $Z_t$ ni se han etiquetado regímenes alcistas o bajistas con un protocolo específico.

### 3.7 Antecedentes organizados por fuente y modelo

La selección siguiente introduce líneas representativas, no una revisión sistemática exhaustiva. Se distingue predecir sentimiento textual de predecir movimientos de mercado. No se trasladan cifras de un artículo a este proyecto cuando cambian universo, periodo, etiqueta o partición.

| Fuente | Estudio | Enfoque | Relación con este trabajo |
| --- | --- | --- | --- |
| Precios históricos | Fischer y Krauss (2018) | Redes LSTM para predicción financiera | Alternativa secuencial a los modelos tabulares; no implementada aquí |
| Redes sociales | Bollen, Mao y Zeng (2011) | Indicadores de estado de ánimo, análisis de precedencia y red neuronal difusa para DJIA | Distingue dimensiones del texto; su índice agregado no equivale a noticias por empresa |
| Noticias | Hu et al. (2018) | Redes de atención sobre secuencias de noticias | Modela influencia desigual y contexto; la agregación diaria de este proyecto es más simple |
| Tweets y precios | Xu y Cohen (2018) | Modelo generativo que combina lenguaje y precios | Antecedente de fusión de fuentes; distinto de concatenar indicadores agregados |
| Texto financiero | Araci (2019), FinBERT | Adaptación de BERT al dominio y clasificación de sentimiento | Alternativa al sentimiento del proveedor; clasificar bien el texto no garantiza predecir retornos |

Fuentes primarias: [Fischer y Krauss](https://cris.fau.de/publications/208534319/), [Bollen et al.](https://arxiv.org/abs/1010.3003), [Hu et al.](https://arxiv.org/abs/1712.02136), [Xu y Cohen](https://aclanthology.org/P18-1183/) y [Araci](https://arxiv.org/abs/1908.10063).

La comparación sugiere tres decisiones para la investigación: distinguir la calidad del lenguaje de su utilidad predictiva, controlar el instante de disponibilidad y comparar cada fuente contra una referencia financiera común. El resultado de una arquitectura compleja en otro corpus no justifica adoptarla sin esas comprobaciones. Las noticias también difieren de las redes sociales en selección editorial, repetición y unidad de análisis; no se presupone que una puntuación tenga el mismo significado en ambas fuentes.

Para ampliar esta revisión se registrarán por estudio periodo, activos, tamaño de muestra, fuente, unidad temporal, objetivo, referencias, forma de separar entrenamiento y prueba, costes y disponibilidad de código. Se priorizarán artículos con protocolo verificable y se documentarán también resultados negativos. Queda pendiente una lectura crítica completa y una cobertura bibliográfica más amplia antes de presentar este capítulo como estado del arte definitivo.

### 3.8 Herramientas y sistemas relacionados

| Recurso | Función | Diferencia respecto al sistema desarrollado |
| --- | --- | --- |
| Alpha Vantage | API de datos y noticias con sentimiento | Proveedor de entrada, no evaluador independiente de la utilidad de sus puntuaciones |
| yfinance | Acceso programático a datos de Yahoo Finance | Adquisición de precios, no protocolo experimental |
| FinBERT | Clasificación de sentimiento financiero | Procesamiento textual potencial, aún no conectado al flujo |
| VectorBT | Simulación de carteras a partir de órdenes o señales | Extensión económica posible; las métricas actuales no son un backtest |
| SHAP | Atribuciones de variables a predicciones | Extensión explicativa; no sustituye evaluación externa ni demuestra causalidad |

La documentación de [Alpha Vantage](https://www.alphavantage.co/documentation/#news-sentiment), [yfinance](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html), [FinBERT](https://huggingface.co/ProsusAI/finbert), [VectorBT](https://vectorbt.dev/api/portfolio/base/) y [SHAP](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) permite delimitar esas funciones. El producto construido es un flujo local reproducible con informes y cuadernos; no se presenta como una aplicación móvil, un servicio de señales ni un sistema en tiempo real.

### 3.9 Formulación del aprendizaje y la comparación

Cada ejemplo es $(X_{i,t},y_{i,t})$, con $X_{i,t}$ disponible al cierre. El modelo devuelve $p_{i,t}$ y la clase $\hat y_{i,t}=\mathbb{1}(p_{i,t}\geq0,5)$. La regresión logística utiliza:

$$
p_{i,t}=\sigma(\beta_0+\beta^\top X_{i,t}),\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

La pérdida logística binaria básica es $-\sum_j[y_j\log p_j+(1-y_j)\log(1-p_j)]$, acompañada de regularización en el estimador. El parámetro `C` controla inversamente su intensidad. Un bosque agrega predicciones de árboles construidos con aleatorización; la potenciación construye una suma secuencial de árboles $F_M(x)=F_0(x)+\eta\sum_{m=1}^{M}h_m(x)$. La profundidad, tamaño de hojas y regularización limitan complejidad, pero no garantizan generalización temporal.

Para un bloque externo $k$, la selección interna es $\hat\lambda_k=\arg\max_{\lambda\in\Lambda}K^{-1}\sum_{j=1}^{K}\operatorname{AUC}_{k,j}(\lambda)$. Solo después se reentrena con el pasado admisible y se evalúa el bloque externo. Las diferencias pareadas mantienen iguales fechas, activos y objetivos. El código de [construcción de modelos](../src/models/train_models.py) y [selección de la ronda](../src/experiments/round_selection.py) concreta los valores y convenciones; la notación anterior no sustituye esa especificación.

## 4. Fuentes de datos y procesamiento

### 4.1 Datos y construcción del objetivo

#### 4.1.1 Universo y periodo

Se descargaron inicialmente AAPL, MSFT, NVDA, TSLA y SPY. El modelado utiliza las cuatro empresas, con SPY excluido por ser un ETF y no disponer en este diseño de una señal corporativa comparable. No se evalúa, por tanto, una hipótesis de sentimiento sobre el índice.

El periodo solicitado fue 2015–2025. Los precios disponibles comienzan el 2 de enero de 2015 y llegan al 30 de diciembre de 2025. Al requerir la sesión siguiente para la etiqueta, la última fecha modelada es el 29 de diciembre de 2025. El panel de las cuatro empresas contiene 11.056 observaciones etiquetadas; no son 11.056 fechas independientes, sino filas empresa-sesión.

`yfinance` permite descargar campos de mercado como precios y volumen. La configuración y los CSV guardados son la evidencia efectiva de este proyecto; una descarga posterior puede no reproducir exactamente los mismos valores. [Documentación de yfinance](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).

#### 4.1.2 Etiqueta

Para el cierre ajustado $P^{adj}_{i,t}$:

$$
y_{i,t}=\mathbb{1}\left(P^{adj}_{i,t+1}>P^{adj}_{i,t}\right).
$$

La clase cero incluye bajadas e igualdad; no equivale exclusivamente a «baja». El horizonte es la siguiente sesión disponible, no necesariamente el siguiente día natural. La última observación sin precio posterior se elimina. `next_adj_close` sirve para construir la etiqueta, pero no entra en las variables predictoras; tampoco entran `target`, `target_end` o `Date`. La columna textual `ticker` identifica y agrupa filas: solo el experimento de identidad incorpora indicadores binarios derivados de ella, nunca del objetivo.

Se utiliza `Adj Close` para objetivo, retorno e indicadores basados en cierre. La revisión comprueba el objetivo frente al precio bruto guardado y obtiene `target_end` a partir de la fecha de la siguiente observación. Esto permite expresar de forma explícita cuándo se conoce cada etiqueta.

#### 4.1.3 Noticias y disponibilidad

La descarga usa Alpha Vantage, filtros por empresa y rangos temporales. El código parte de rangos anuales, divide rangos con al menos 950 resultados y establece un límite de 1.000. Si una sola jornada permanece saturada se produce un error explícito. Los fragmentos se guardan para reanudar descargas; se incorporan reintentos y validación de respuestas.

El acceso de pago se utilizó para ampliar la capacidad operativa de adquisición durante el proyecto. No se dispone aquí de una contabilidad validada de cuotas, facturas, horas o coste total, por lo que no se inventan importes. Una suscripción tampoco demuestra que el histórico del proveedor sea exhaustivo.

Los registros conservan empresa, título, resumen, URL, fuente, marca temporal, puntuación y relevancia. Una misma noticia puede aparecer asociada a varias empresas; los recuentos deben interpretarse como registros noticia-empresa, no necesariamente como artículos únicos en todo el corpus.

### 4.2 Variables financieras y sentimiento

#### 4.2.1 Bloque financiero

El modelo base contiene 14 variables: `Adj Close`, `Close`, `High`, `Low`, `Open`, `Volume`, `daily_return`, `sma_5`, `sma_20`, `sma_50`, `RSI`, `MACD`, `MACD_signal` y `volatility_20`.

| Transformación | Definición operativa | Tratamiento inicial |
| --- | --- | --- |
| Retorno diario | Variación relativa de `Adj Close` | Primer retorno sin histórico: cero |
| SMA 5, 20 y 50 | Media móvil de `Adj Close` | `min_periods=1` |
| RSI 14 | Suavizado exponencial de ganancias y pérdidas, alfa 1/14 | Relleno inicial con 50 y rango [0,100] |
| MACD | EMA de 12 menos EMA de 26 sesiones | `adjust=False` |
| Señal MACD | EMA de 9 sesiones del MACD | Historial disponible |
| Volatilidad 20 | Desviación típica móvil del retorno diario | Mínimo dos observaciones; inicial cero |

Estas convenciones conservan las primeras filas y son decisiones de implementación. No equivalen a disponer de una ventana completa desde la primera sesión. Mezclar precios ajustados con niveles OHLC no ajustados también merece cautela al interpretar variables o comparar periodos con operaciones corporativas.

#### 4.2.2 Bloque de sentimiento

La puntuación procede de `ticker_sentiment_score`; no es una predicción textual generada por un modelo entrenado en este TFG. Las etiquetas alcistas (`Bullish` y `Somewhat-Bullish`) se agrupan como positivas; las bajistas (`Bearish` y `Somewhat-Bearish`), como negativas; y la etiqueta `Neutral`, como neutral. El clasificador de etiquetas actual asigna también las etiquetas desconocidas a neutral: esta tolerancia constituye una limitación que debería sustituirse por auditoría explícita en una evolución del sistema.

Por empresa y sesión se calculan media, mediana, mínimo y máximo de tono; número de noticias; número de noticias en día no bursátil; conteos y proporciones positivos, negativos y neutrales; y `has_news`. Son 13 variables de sentimiento y, junto a las 14 financieras, 27 variables híbridas.

Las sesiones sin noticias registradas se conservan con puntuaciones, conteos y proporciones cero y `has_news=False`. No se interpreta «sin noticias registradas» como «no ocurrió nada» ni se confunde con una noticia de tono neutral. El indicador de presencia permite distinguir parcialmente ambos casos, pero no resuelve una cobertura incompleta.

#### 4.2.3 Retardos y variantes controladas

El primer ensayo de persistencia añadió retardos de 1, 2 y 3 sesiones y medias móviles de 3 y 5 a las 13 variables de sentimiento: 65 variables nuevas, hasta 92 predictoras. Las operaciones se agrupan por empresa. Los retardos usan sesiones anteriores; las medias móviles incluyen la actual porque el instante de decisión se sitúa después de conocer su información.

La revisión posterior separó hipótesis para evitar modificar 65 variables a la vez:

| Variante | Variables | Propósito |
| --- | ---: | --- |
| `base` | 14 | Referencia financiera |
| `news_volume` | 16 | Añadir cantidad y presencia de noticias |
| `sentiment_only` | 8 | Tono, proporciones y presencia, sin precios ni conteos |
| `financial_sentiment` | 22 | Variables financieras más el bloque anterior |
| `hybrid` | 27 | Combinación completa de referencia |
| `lag_1`, `lag_2`, `lag_3` | 29 cada una | Híbrido más retardo de tono medio y cantidad |
| `rolling_3`, `rolling_5` | 29 cada una | Híbrido más media móvil de tono y cantidad |
| `weighted_sentiment` | 27 | Sustituir la media simple por media ponderada por relevancia |
| `relative_features` | 22 | Sustituir niveles por representaciones relativas y mantener sentimiento |

`sentiment_only` no es tono puro, porque incorpora proporciones y presencia. Las ventanas de `news_count` son medias, no sumas. La media ponderada es $\sum_j r_j s_j/\sum_j r_j$, con cero cuando no existe denominador válido.

Las variables relativas incluyen distancia a SMA, MACD dividido por precio y volumen o cantidad de noticias divididos por su media de las veinte sesiones anteriores. Esta última media desplaza primero una sesión para excluir el valor actual del denominador. La variante cambia también la representación financiera; una mejora no podría atribuirse únicamente al sentimiento.

### 4.3 Disponibilidad temporal y calidad de datos

#### 4.3.1 Instante de predicción

La predicción se plantea después de conocer el cierre de la sesión t y sus variables financieras. Una noticia se asigna al primer cierre igual o posterior a su marca temporal. Así, una noticia posterior al cierre del viernes se asigna a una sesión posterior, no al viernes. Si coincide exactamente con el cierre se considera disponible, una hipótesis que debe hacerse explícita.

El calendario NASDAQ incorpora sesiones, festivos y cierres anticipados mediante [pandas_market_calendars](https://pandas-market-calendars.readthedocs.io/en/latest/). Las marcas temporales sin zona se interpretan como UTC por defecto; para interpretar su fecha local se utiliza America/New_York. La revisión incluye pruebas de horario de verano y cierres anticipados.

La marca temporal de publicación solo aproxima disponibilidad. No se dispone de tiempos de recepción, latencia del cálculo de sentimiento, revisiones del artículo ni una base que garantice íntegramente la información disponible en cada instante. Además, usar el cierre como característica no implica poder negociar a ese mismo precio una vez conocido. El objetivo es predictivo, no una simulación de ejecución.

#### 4.3.2 Auditoría observada

La ejecución completa contiene 46.014 registros de noticias alineados con el calendario generado. De ellos, 17.308 pasan a una fecha posterior a su fecha local de publicación, incluyendo fines de semana y mensajes posteriores al cierre. No son necesariamente 17.308 errores del flujo de procesamiento anterior.

No se detectan duplicados exactos bajo la clave empresa, URL, título y marca temporal. Sí existen 3.124 repeticiones de título dentro de la misma empresa y sesión; no se eliminan automáticamente porque pueden representar actualizaciones distintas. Los registros sin cierre posterior disponible se conservan separados.

El conjunto de datos modelado registra 3.675 noticias en 2024 y 27.336 en 2025. El cambio de volumen puede alterar la distribución de las variables y no debe atribuirse automáticamente a actividad informativa real: la cobertura no está certificada. El manifiesto conserva `coverage_verified=False`.

### 4.4 Definición matemática de las transformaciones

Las fórmulas siguientes describen el código existente, incluidas sus convenciones de arranque. Sea $P_t$ el cierre ajustado de una empresa; todas las operaciones se calculan dentro de esa empresa y en orden temporal.

**Retorno y medias móviles.** El retorno simple es $r_t=P_t/P_{t-1}-1$. La primera fila se completa con cero por falta de observación anterior. Para una ventana $w$ y $m_t=\min(w,t+1)$ observaciones disponibles:

$$
\operatorname{SMA}_{w,t}=\frac{1}{m_t}\sum_{j=0}^{m_t-1}P_{t-j},\qquad w\in\{5,20,50\}.
$$

La EMA implementada con `adjust=False` sigue $E_t=\alpha P_t+(1-\alpha)E_{t-1}$, con $E_0=P_0$ y $\alpha=2/(w+1)$. A diferencia de la SMA, pondera más las observaciones recientes sin imponer un corte abrupto. Son descriptores del pasado, no reglas de compraventa incorporadas al sistema.

**MACD y señal.** Se define $\operatorname{MACD}_t=\operatorname{EMA}_{12,t}-\operatorname{EMA}_{26,t}$ y $\operatorname{signal}_t=\operatorname{EMA}_{9}(\operatorname{MACD})_t$. El MACD conserva unidades de precio; dividirlo por $P_t$ en la representación relativa reduce diferencias de escala entre activos.

**RSI.** Para $\Delta_t=P_t-P_{t-1}$, se calculan $g_t=\max(\Delta_t,0)$ y $\ell_t=\max(-\Delta_t,0)$. Sus medias exponenciales usan $\alpha=1/14$, `adjust=False` y un mínimo de 14 observaciones válidas. Para pérdidas medias positivas:

$$
\operatorname{RS}_t=\frac{\bar g_t}{\bar\ell_t},\qquad
\operatorname{RSI}_t=100-\frac{100}{1+\operatorname{RS}_t}.
$$

Con ganancias y pérdidas medias nulas se usa 50; con ganancias positivas y pérdidas nulas, 100. Los valores iniciales no disponibles se rellenan con 50. Es un suavizado con coeficiente de Wilder; la inicialización de `ewm` no debe confundirse con otra implementación que inicialice la recurrencia mediante una media simple de las primeras 14 variaciones. Se conserva esta convención en todas las comparaciones.

**Volatilidad.** Para $m_t=\min(20,t+1)$ y $m_t\geq2$:

$$
s_t=\sqrt{\frac{1}{m_t-1}\sum_{j=0}^{m_t-1}(r_{t-j}-\bar r_t)^2}.
$$

La implementación utiliza desviación muestral (`ddof=1`), no volatilidad anualizada. Se rellena con cero cuando no hay dos observaciones. La interpretación debe considerar que el primer retorno fue rellenado, no observado.

**Representación relativa.** La distancia a una media es $d_{w,t}=P_t/\operatorname{SMA}_{w,t}-1$. El volumen relativo utiliza $V_t/\bar V_{t-1}^{(20)}$, donde el denominador excluye la sesión actual; se aplica el mismo principio a la cantidad de noticias. El código de [variantes](../src/experiments/ablations.py) establece los tratamientos de denominadores nulos y valores iniciales. No se ajusta una normalización utilizando fechas futuras.

**Agregación textual.** Para las $n_{i,t}$ noticias registradas, con puntuaciones $s_j$ y relevancias $q_j$:

$$
\bar s_{i,t}=\frac{1}{n_{i,t}}\sum_j s_j,\qquad
\bar s^{(q)}_{i,t}=\frac{\sum_jq_js_j}{\sum_jq_j},\qquad
a_{i,t}=\frac{1}{n_{i,t}}\sum_j|s_j|.
$$

La dispersión de las nuevas variables usa denominador $n_{i,t}$ (`ddof=0`), no el de la volatilidad financiera. La sorpresa del tono resta su media histórica, excluyendo sesiones sin noticias de esa media. La sorpresa del volumen estandariza $\log(1+n_{i,t})$ con media y desviación de las veinte sesiones anteriores. Se utilizan ceros para casos sin información suficiente y un indicador explícito de disponibilidad histórica. La fórmula de tono no convierte esos ceros en observaciones neutrales reales.

**Indicadores no utilizados.** Las bandas de Bollinger pueden expresarse como $\operatorname{SMA}_{w,t}\pm k s^{(P)}_{w,t}$, con desviación de precios y parámetros prefijados. Se mencionan como ejemplo de indicador relacionado, no como variable entrenada: no aparecen en los conjuntos de este TFG. Incorporarlas exigiría un experimento nuevo; tampoco se presupone normalidad ni cobertura probabilística del 95 % por elegir $k=2$.

La correspondencia con la implementación puede revisarse en [medias móviles](../src/data/calculate_moving_averages.py), [MACD](../src/data/calculate_macd.py), [RSI](../src/data/calculate_rsi.py), [volatilidad](../src/data/calculate_volatility.py) y [representación enriquecida](../src/experiments/sentiment_representation.py).

### 4.5 Contratos y esquemas de datos

| Entidad lógica | Clave o identidad | Campos principales | Restricciones relevantes |
| --- | --- | --- | --- |
| Precio diario | Empresa y `Date` | OHLC, `Adj Close`, `Volume` | Orden temporal; una observación por sesión y empresa |
| Noticia-empresa | Empresa, URL, título y publicación para duplicados exactos | `title`, `summary`, fuente, sentimiento, relevancia | Puntuación en [−1,1], relevancia en [0,1]; publicación interpretable |
| Calendario | `trading_date` | `market_close` con zona horaria | Sesiones únicas y cierre válido |
| Panel modelado | `ticker`, `Date` | Variables, `target`, `target_end` | Objetivo binario; variables finitas; horizonte posterior |
| Predicción externa | Ejecución, enfoque, variante, modelo, bloque, empresa y fecha | `target`, `probability`, `prediction` | Probabilidad en [0,1]; mismas claves para contrastes |
| Manifiesto | Identificador de ejecución | Configuración, versiones, Git, hashes y estado | Identificador nuevo; finalización explícita; procedencia localizable |

Los CSV no imponen tipos por sí mismos. Los lectores convierten fechas y números y validan contratos antes de entrenar. Las claves de noticia representan registros, no garantizan que dos textos distintos describan acontecimientos distintos. `next_adj_close` y `target_end` se usan para etiqueta y validación, nunca como predictores. El [constructor temporal](../src/data/point_in_time.py) comprueba etiquetas contra los precios guardados, une tablas con validación de cardinalidad y rechaza valores no finitos.

### 4.6 Análisis exploratorio visual

Las figuras de este apartado se calculan a partir del panel conservado de `first-round-full-20260916`, sin descargas ni entrenamiento. Se representan las cuatro empresas por separado para evitar que sus niveles oculten diferencias. La normalización a 100 facilita comparar trayectorias, pero no transforma el gráfico en una simulación de inversión con costes.

![Evolución del cierre ajustado, normalizado al primer valor](figures/memoria/precios.png)

**Figura 4.1.** Cierre ajustado normalizado a 100 al inicio de cada serie; eje vertical logarítmico. El gráfico contextualiza cambios de escala y periodos de movimientos intensos. No se utiliza para escoger retrospectivamente un periodo favorable de prueba.

![Volatilidad móvil por empresa](figures/memoria/volatilidad.png)

**Figura 4.2.** Desviación típica móvil de veinte sesiones, expresada como porcentaje diario. Los cambios de volatilidad motivan el desglose temporal, pero no constituyen una evaluación por regímenes económicos formalmente definidos.

![Volumen diario negociado por empresa](figures/memoria/volumen.png)

**Figura 4.3.** Volumen diario en millones de acciones. Se mantienen escalas propias por empresa; comparar números absolutos entre activos requiere considerar sus características y ajustes históricos.

![Correlaciones entre variables financieras y de sentimiento](figures/memoria/correlaciones.png)

**Figura 4.4.** Correlación de Spearman por empresa entre seis variables representativas, calculada sobre el panel completo con los ceros de las sesiones sin noticias. Es un diagnóstico contemporáneo y descriptivo, no una prueba de causalidad ni una selección de variables basada en capacidad predictiva. La ausencia de correlación monotónica no excluye relaciones no lineales.

![Densidad mensual y cobertura informativa](figures/memoria/cobertura.png)

**Figura 4.5.** Noticias por sesión y proporción de sesiones con noticias, antes y después de deduplicar. Cada registro se cuenta por empresa. El fuerte cambio de cobertura exige cautela al interpretar aumentos de intensidad: no se identifica automáticamente con un cambio de actividad económica real. La asignación de noticias de fines de semana y festivos se describe en 4.3; no se ha contrastado todavía con un calendario independiente de anuncios de resultados empresariales.

## 5. Diseño experimental y resultados

### 5.1 Modelos y selección de hiperparámetros

#### 5.1.1 Algoritmos

Se utilizan cinco algoritmos predictivos: regresión logística, Random Forest, HistGradientBoosting, XGBoost y LightGBM. Se añade DummyClassifier como referencia de clase mayoritaria. Por tanto, son cinco algoritmos más una referencia, no seis modelos de sentimiento diferentes.

Todos los estimadores incorporan imputación por mediana dentro de una cadena de preprocesamiento y modelado. La regresión logística añade StandardScaler. La imputación y el escalado se ajustan exclusivamente con el entrenamiento de cada partición. La semilla habitual es 42 y se limitan hilos en las ejecuciones para controlar el uso de recursos.

#### 5.1.2 Evolución del ajuste

Tras los ensayos iniciales se incorporó validación temporal y se amplió la búsqueda. El ajuste tradicional utiliza para la regresión logística valores de C entre 0,01 y 100 y ponderación de clases opcional. En árboles se prueban profundidad, hojas, regularización, número de iteraciones, tasa de aprendizaje y muestreo según el algoritmo. Los detalles completos permanecen en `src/models/tune_temporal_cv.py` y en los CSV históricos de ajuste de hiperparámetros.

También se exploraron umbrales entre 0,40 y 0,60, en pasos de 0,025, seleccionados mediante F1 de la clase positiva. Este criterio puede favorecer predecir muchas subidas. Por ello, la revisión principal fija el umbral en 0,5 y separa selección de parámetros y evaluación externa.

#### 5.1.3 Búsqueda del experimento corregido

| Algoritmo | Dos configuraciones comparadas | Otros valores fijados en la revisión |
| --- | --- | --- |
| Regresión logística | C = 0,1 o 10 | Máximo 1.000 iteraciones |
| Random Forest | Profundidad = 4 u 8 | 100 árboles; mínimo 20 muestras por hoja |
| HistGradientBoosting | Máximo 7 o 15 hojas | 100 iteraciones; tasa 0,05; parada temprana desactivada |
| XGBoost | Profundidad = 2 o 3 | 100 árboles; tasa 0,05; muestreo de filas y columnas 0,8 |
| LightGBM | 7 o 15 hojas | 100 árboles; tasa 0,05; muestreo 0,8; frecuencia de muestreo 1 |

Los valores no enumerados se heredan del constructor y las versiones fijadas. El manifiesto registra la cuadrícula y la versión del entorno; el código registra la construcción completa. Se corrigió `subsample_freq` de LightGBM porque, con frecuencia cero, el parámetro de muestreo de filas no activaba el comportamiento que se pretendía estudiar.

La búsqueda reducida permite repetir el protocolo en todas las variantes con un presupuesto acotado. No constituye una optimización exhaustiva y sus resultados no se comparan con los históricos como si solo hubiese cambiado una variable.

### 5.2 Diseño experimental y evaluación

#### 5.2.1 De la partición temporal simple al protocolo anidado

El primer diseño dividió el 80 % inicial de fechas únicas para entrenamiento y el 20 % final para prueba: 8.844 y 2.212 filas. La revisión incorpora purga: una fila de entrenamiento solo es admisible si su fecha y el fin de su etiqueta son anteriores al comienzo de validación. Por ello el primer entrenamiento externo corregido contiene 8.840 filas.

El protocolo actual divide el periodo externo en tres bloques cronológicos. Para cada bloque, se utiliza únicamente el pasado disponible. Dentro de ese pasado se realizan tres particiones expansivas y se escoge la configuración con mayor media de AUC interno. El modelo seleccionado se ajusta de nuevo y predice el bloque externo una sola vez dentro de esa ejecución.

| Bloque | Fin de entrenamiento | Fin máximo de etiqueta | Inicio externo | Fin externo | Filas de entrenamiento | Filas externas |
| --- | --- | --- | --- | --- | ---: | ---: |
| 1 | 2023-10-12 | 2023-10-13 | 2023-10-16 | 2024-07-11 | 8.840 | 740 |
| 2 | 2024-07-10 | 2024-07-11 | 2024-07-12 | 2025-04-04 | 9.580 | 736 |
| 3 | 2025-04-03 | 2025-04-04 | 2025-04-07 | 2025-12-29 | 10.316 | 736 |

Los bloques externos anteriores pueden incorporarse al entrenamiento de bloques posteriores cuando sus etiquetas ya son conocidas. Eso es coherente con el diseño expansivo, pero hace que los resultados de los bloques no constituyan experimentos completamente independientes.

Doce variantes por cinco algoritmos, dos configuraciones, tres particiones internas y tres bloques externos producen 1.080 ajustes internos. Los sesenta pares variante-algoritmo más el modelo de referencia, en tres bloques, producen 183 modelos externos. Se almacenan 61 grupos de predicciones con 2.212 filas cada uno, es decir, 134.932 predicciones; no son 134.932 observaciones independientes.

#### 5.2.2 Métricas

Se denotan los verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos mediante TP, TN, FP y FN, respectivamente, para mantener la correspondencia con la notación habitual de las métricas:

- **Exactitud (`accuracy`):** $(TP+TN)/N$.
- **Precisión (`precision`):** $TP/(TP+FP)$.
- **Sensibilidad (`recall`):** $TP/(TP+FN)$.
- **F1 positivo:** media armónica de precisión y sensibilidad de la clase subida.
- **Exactitud equilibrada (`balanced_accuracy`):** media de la sensibilidad de ambas clases.
- **F1 macro:** media de los F1 de ambas clases.
- **MCC:** correlación entre clases observadas y predichas, con referencia cero.
- **ROC-AUC:** evalúa la ordenación de las puntuaciones, sin fijar un umbral de clasificación; 0,5 representa falta de discriminación en esa medida.

Estas definiciones y sus implementaciones pueden consultarse en [scikit-learn 1.7](https://scikit-learn.org/1.7/modules/model_evaluation.html). Un AUC de 0,51 no significa acertar el 51 % ni equivale a una rentabilidad del 1 %. El proyecto prioriza AUC para selección interna y lo complementa con métricas equilibradas y un modelo de referencia.

Se informa tanto AUC agrupado de todas las predicciones externas como media de AUC por bloque. No coinciden necesariamente: el primero incluye comparaciones entre puntuaciones de modelos entrenados en distintos momentos, mientras que la segunda resume discriminación dentro de cada bloque.

#### 5.2.3 Incertidumbre

Se emplea remuestreo pareado de bloques móviles: las fechas se remuestrean en bloques contiguos y las cuatro empresas de cada fecha permanecen juntas. Los dos modelos comparados utilizan exactamente las mismas filas remuestreadas. Se calculan 1.000 réplicas e intervalos percentiles del 95 %.

El bloque principal tiene veinte sesiones. La comparación LightGBM con retardo de una sesión frente a híbrido también se evalúa con bloques de cinco y sesenta. Los intervalos son exploratorios: no incorporan todo el proceso de reentrenamiento y selección, ni corrigen las múltiples comparaciones. La comparación principal se fijó para esta ejecución tras haber observado resultados históricos anteriores; no equivale a un prerregistro independiente de todo el desarrollo.

#### 5.2.4 Especialización por empresa y representación compartida

El experimento `per-company-20260916` surge de una limitación posible del entrenamiento conjunto: las cuatro empresas pueden responder de manera distinta a indicadores y noticias. Se entrenan estimadores independientes para AAPL, MSFT, NVDA y TSLA, comparando base, híbrido y retardo de una sesión. Se mantienen los cinco algoritmos, las rejillas de la sección 5.1.3, tres particiones internas, tres bloques externos y umbral 0,5. No se utiliza el antiguo diseño de 92 variables, sino 14, 27 y 29 variables, respectivamente.

Cada empresa dispone de 2.764 observaciones y 553 sesiones externas entre el 16 de octubre de 2023 y el 29 de diciembre de 2025. Tras la purga, los entrenamientos individuales contienen 2.210, 2.395 y 2.579 filas. La especialización reduce a una cuarta parte los ejemplos de cada estimador respecto al panel conjunto. Puede favorecer adaptación, pero también aumentar la variabilidad del ajuste; el experimento no identifica una causa única para los cambios observados.

Las noticias se filtran por empresa antes de calcular sus retardos. Un artículo que menciona varias empresas puede participar en cada una con su puntuación específica; no se impone exclusividad artificial ni se mezcla el sentimiento entre activos. Imputación, escalado, selección de parámetros y entrenamiento utilizan exclusivamente el pasado de la empresa correspondiente. La referencia conjunta se reentrena y sus 35.392 probabilidades coinciden exactamente con las variantes equivalentes de la revisión principal.

El resultado mixto motiva `ticker-aware-full-20260916`: conservar el panel completo, permitiendo identificar la empresa y reduciendo diferencias de escala entre activos. Se separan cambios para distinguir su contribución:

| Enfoque | Cambio respecto a la referencia conjunta | Algoritmos |
| --- | --- | --- |
| Original + empresa | Variables originales y tres indicadores binarios de empresa | Los cinco |
| Relativas | Sustituir niveles absolutos por representaciones relativas, sin identidad | Los cinco |
| Relativas + empresa | Representación relativa más identidad | Los cinco |
| Interacciones originales | Originales, identidad y productos empresa-variable | Regresión logística |
| Interacciones relativas | Relativas, identidad y productos empresa-variable | Regresión logística |

Se codifican MSFT, NVDA y TSLA, con AAPL como referencia. En regresión logística los indicadores permiten distintos niveles de partida, mientras que sus productos con las variables permiten pendientes diferentes. Los árboles pueden representar interacciones mediante sus divisiones. No se implementa un modelo jerárquico bayesiano ni se garantiza que compartir información sea óptimo.

Las relativas incluyen distancias del cierre ajustado a sus medias móviles, MACD dividido por precio y volúmenes frente a su media de veinte sesiones anteriores. La base relativa tiene 9 variables financieras; el híbrido, 22; y el híbrido con retardo, 24. La base no incorpora noticias. El retardo relativo añade el tono medio y el volumen relativo de noticias de la sesión anterior de la misma empresa. Añadir identidad incorpora tres columnas; las interacciones relativas alcanzan 39, 91 y 99 variables. La rejilla de regularización permanece igual, por lo que no se optimiza exhaustivamente esta mayor dimensionalidad.

Se reutilizan las referencias conjunta e individual solo tras verificar entradas, panel reconstruido, código principal, versiones, rejillas, umbral y fronteras temporales. No se descargan datos nuevos. La selección interna del modelo conjunto mantiene el AUC agrupado del panel; la evaluación externa prioriza el promedio de AUC dentro de cada empresa. Esta diferencia entre criterio de ajuste y resumen externo se conserva para mantener la comparación, no se presenta como una selección optimizada para AUC macro.

#### 5.2.5 Métricas comparables en las ampliaciones

Para evitar comparar puntuaciones de empresas con calibraciones distintas, se calcula el AUC de cada empresa sobre sus 553 sesiones externas y después su media con igual peso:

$$
\operatorname{AUC}_{macro}=\frac{1}{4}\sum_{i=1}^{4}\operatorname{AUC}_i.
$$

Este AUC macro no es el AUC agrupado de todas las filas de la revisión inicial. Tampoco coincide necesariamente con la media por bloque temporal. Las tablas que promedian además los cinco algoritmos son resúmenes descriptivos, no un modelo adicional ni una combinación de predicciones. Los contrastes mantienen fijos algoritmo, variante y observaciones.

En el experimento individual se calculan 60 diferencias individual menos conjunto y otras 40 comparaciones de sentimiento y retardo dentro del enfoque individual. En el segundo experimento, la comparación principal es relativas con identidad menos conjunto original para cada algoritmo y variante; los demás contrastes separan identidad, representación e interacciones. Sus 405 intervalos se distribuyen en 81 macro y 324 por empresa. Se mantienen 1.000 réplicas de bloques de veinte sesiones; en los contrastes macro se remuestrean las mismas fechas simultáneamente para las cuatro empresas y ambos enfoques. No se reentrena dentro del remuestreo ni se corrige por comparaciones múltiples.

#### 5.2.6 Ronda controlada de mejoras

El experimento `first-round-full-20260916` investiga si la calidad de las noticias, la cantidad de historia y el procedimiento de selección limitaban la representación relativa. Su referencia es el modelo conjunto con variables relativas y sin identidad. Se prefijan tres algoritmos: regresión logística, bosque aleatorio y potenciación por histogramas. Los promedios entre estos tres no deben compararse directamente con los promedios de cinco algoritmos anteriores.

Se conservan las cuatro empresas, las 553 sesiones externas por empresa, tres bloques externos, tres particiones internas, purga y umbral 0,5. Las siete etapas son referencia, deduplicación, ventana de tres años, ventana de cinco años, sentimiento enriquecido, selección por AUC macro y búsqueda ampliada. Las ventanas se calculan respecto al inicio de cada partición, también durante la selección interna. El contraste principal es ajuste ampliado menos referencia para el híbrido de cada algoritmo.

La deduplicación conserva la primera publicación de cada título normalizado idéntico por empresa en 24 horas, sin utilizar noticias futuras. El enriquecimiento añade dispersión del sentimiento, intensidad absoluta, sorpresa del tono, disponibilidad de historia y sorpresa del volumen. Las referencias históricas usan las veinte sesiones anteriores; el tono requiere al menos cinco observaciones con noticias, y la sorpresa del volumen estandariza `log1p(news_count)` y se limita a [−5; 5]. El híbrido pasa de 22 a 27 variables y el de retardo de 24 a 34; la base permanece en nueve.

La selección macro alinea el criterio interno con el promedio de AUC por empresa. La búsqueda ampliada compara veinte candidatos totales por algoritmo de árboles y seis en regresión logística, incluyendo la ventana de entrenamiento. No son veinte candidatos por ventana. Los candidatos y semillas se fijan antes de ejecutar la ronda. Se generan 210 modelos externos y 2.214 evaluaciones internas; los 315 intervalos pareados se distribuyen en 63 macro y 252 por empresa. El [protocolo de la ronda](first_round_protocol.md) conserva los espacios de búsqueda y controles.

#### 5.2.7 Ablación individual de las nuevas variables

Como añadir las cinco variables simultáneamente no produjo una mejora general, `variable-ablation-full-20260917` estudia qué aporta cada variable añadida a la referencia sin extras y qué sucede al retirarla del bloque completo. Se mantienen noticias deduplicadas, historia completa, fechas, etiquetas, algoritmos y purga. En la variante con retardo se añade o retira conjuntamente la variable actual y su retardo de una sesión; no se aísla el efecto de cada componente.

Los hiperparámetros se heredan, por algoritmo, variante y bloque, de la referencia correspondiente: `deduplicated` al añadir y `enhanced` al retirar. Se habían seleccionado mediante validación temporal interna, pero no se vuelven a optimizar para cada subconjunto. Las dos referencias pueden tener parámetros diferentes. Esta decisión estudia el cambio de representación manteniendo fijo el ajuste dentro de cada contraste, no busca el mejor modelo posible para cada variable.

Se verifican 36 modelos de referencia y se entrenan 180 nuevos: cinco variables, dos operaciones, dos variantes, tres algoritmos y tres bloques. Se calculan 60 contrastes macro y 240 por empresa. Las 15 adiciones al híbrido sin retardo son primarias; las demás, secundarias. Los intervalos del 95 % utilizan 1.000 réplicas pareadas de bloques de veinte sesiones, sin reajustar modelos ni corregir por multiplicidad. El [protocolo de ablación](variable_ablation_protocol.md) se fija antes de consultar estos resultados.

### 5.3 Resultados

#### 5.3.1 Antecedentes experimentales

Los primeros ensayos emplearon Dummy, regresión logística, Random Forest y HistGradientBoosting. Después se incorporaron XGBoost y LightGBM, ajuste temporal y retardos. La siguiente tabla conserva la evolución histórica del AUC a umbral 0,5; el umbral afecta a las clases, no al cálculo del AUC.

| Algoritmo | Base inicial | Híbrido inicial | Híbrido ajustado | Híbrido ajustado con 65 retardos/ventanas |
| --- | ---: | ---: | ---: | ---: |
| Regresión logística | 0,5115 | 0,5011 | 0,5033 | 0,5042 |
| Random Forest | 0,5124 | 0,5040 | 0,5010 | 0,5147 |
| HistGradientBoosting | 0,4930 | 0,5054 | 0,5087 | 0,4957 |
| XGBoost | No ejecutado | No ejecutado | 0,5026 | 0,4993 |
| LightGBM | No ejecutado | No ejecutado | 0,5119 | 0,5217 |

Fuente: `reports/historical/tuning/model_progression_summary.csv`. Las celdas ausentes no son ceros. El AUC 0,5217 de LightGBM motivó seguir estudiando el retardo, pero no constituye evidencia final: se habían consultado repetidamente las mismas fechas externas y existían limitaciones en la alineación horaria y la purga.

#### 5.3.2 Comparación corregida base e híbrido

| Algoritmo | AUC base | AUC híbrido | Diferencia híbrido − base |
| --- | ---: | ---: | ---: |
| Regresión logística | 0,5040 | 0,4976 | −0,0064 |
| Random Forest | 0,5049 | 0,5018 | −0,0031 |
| HistGradientBoosting | 0,4868 | 0,4932 | +0,0064 |
| XGBoost | 0,4942 | 0,4970 | +0,0028 |
| LightGBM | 0,4920 | 0,4954 | +0,0034 |

Fuente: `reports/experiments/review-full-20260908/metrics/global_metrics.csv`; diferencias calculadas antes de redondear. No existe una mejora uniforme. Las diferencias positivas puntuales no bastan para sostener una ventaja robusta.

#### 5.3.3 Resultados destacados y modelo de referencia

| Configuración | Exactitud | Exactitud equilibrada | F1 macro | MCC | AUC agrupado | Media AUC externa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy | 0,5389 | 0,5000 | 0,3502 | 0,0000 | 0,5000 | 0,5000 |
| HistGradientBoosting relativo | 0,5163 | 0,5024 | 0,4924 | 0,0052 | 0,5108 | 0,5099 |
| Random Forest ponderado | 0,5140 | 0,5051 | 0,5022 | 0,0104 | 0,5075 | 0,5181 |
| LightGBM híbrido | 0,5041 | 0,5016 | 0,5016 | 0,0032 | 0,4954 | 0,5004 |
| LightGBM retardo 1 | 0,4959 | 0,4938 | 0,4938 | −0,0123 | 0,4946 | 0,5005 |

El modelo de referencia predice subida en todas las filas externas: 1.192 positivos y 1.020 negativos. Obtiene la mayor exactitud de este experimento y F1 positivo 0,7004, a pesar de no identificar ninguna clase negativa. Este ejemplo explica por qué el F1 positivo aislado produjo inicialmente una impresión excesivamente favorable.

HistGradientBoosting relativo presenta el mayor AUC agrupado; Random Forest ponderado, la mayor media externa. Son máximos descriptivos entre alternativas observadas, no un modelo final validado de forma independiente.

![AUC de las variantes y algoritmos del experimento corregido](../reports/final/figures/roc_auc.png)

**Figura 1.** AUC agrupado de las predicciones externas, sin el modelo de referencia. La referencia de discriminación es 0,5; el mapa no sustituye los intervalos de incertidumbre. Fuente: selección trazada en `reports/final/provenance.json`.

#### 5.3.4 Intervalos y persistencia

| Comparación | Diferencia AUC | Intervalo exploratorio del 95 % |
| --- | ---: | --- |
| HistGradientBoosting relativo − híbrido | +0,0176 | [−0,0096; 0,0477] |
| Random Forest ponderado − híbrido | +0,0056 | [−0,0041; 0,0156] |
| LightGBM retardo 1 − híbrido | −0,0008 | [−0,0193; 0,0132] |

Fuente: `metrics/auc_intervals.csv` de la misma ejecución, bloques de veinte sesiones. El intervalo del AUC de HistGradientBoosting relativo es [0,4855; 0,5406]. Los intervalos del retardo LightGBM incluyen cero también con bloques de cinco y sesenta sesiones.

De 62 intervalos de diferencias, 61 incluyen cero. El único que no lo incluye corresponde a un empeoramiento de LightGBM con solo sentimiento frente al modelo de referencia. Ninguna mejora positiva excluye cero bajo este análisis. Esto limita la evidencia disponible, pero no demuestra equivalencia exacta entre modelos ni ausencia universal de efecto de las noticias.

#### 5.3.5 Modelos independientes por empresa

Promediando el AUC dentro de cada empresa y después entre los cinco algoritmos:

| Variante | Conjunto original | Individual | Diferencia individual − conjunto |
| --- | ---: | ---: | ---: |
| Base | 0,4947 | 0,4930 | −0,0017 |
| Híbrido | 0,4986 | 0,4964 | −0,0021 |
| Con retardo | 0,4986 | 0,4969 | −0,0017 |

Separar empresas no mejora el promedio general. El desglose del híbrido, como media de los cinco algoritmos, muestra la heterogeneidad:

| Empresa | Híbrido conjunto | Híbrido individual |
| --- | ---: | ---: |
| AAPL | 0,4799 | 0,4489 |
| MSFT | 0,4934 | 0,5021 |
| NVDA | 0,5013 | 0,5120 |
| TSLA | 0,5197 | 0,5227 |

De las 60 diferencias individual menos conjunto, 28 son positivas. Solo tres intervalos quedan totalmente por encima de cero, todos en MSFT; otros tres quedan por debajo, todos en AAPL; los 54 restantes incluyen cero. Las mejoras relativas en MSFT no prueban discriminación absoluta: el AUC de esas configuraciones sigue próximo a 0,5. En las 40 comparaciones internas de híbrido frente a base y retardo frente a híbrido no hay intervalos completamente positivos.

Los máximos individuales observados son 0,4709 en AAPL, 0,5141 en MSFT, 0,5304 en NVDA y 0,5324 en TSLA. Son máximos seleccionados después de observar la prueba; ninguno supera en exactitud a su referencia mayoritaria. No se usan para elegir automáticamente un modelo final. Los resultados completos y sus intervalos están en el [análisis por empresa](../reports/experiments/per-company-20260916/analysis.md) y en el [cuaderno 03](../notebooks/03_per_company_models.ipynb).

#### 5.3.6 Identidad, variables relativas e interacciones

Se utiliza el mismo resumen de AUC macro promediado entre cinco algoritmos, sin incorporar las interacciones exclusivas de regresión logística a esa media:

| Variante | Conjunto original | Original + empresa | Relativas | Relativas + empresa |
| --- | ---: | ---: | ---: | ---: |
| Base | 0,4947 | 0,4947 | 0,5031 | 0,5010 |
| Híbrido | 0,4986 | 0,4984 | 0,5064 | 0,5069 |
| Con retardo | 0,4986 | 0,4989 | 0,5058 | 0,5037 |

Las relativas sin identidad mejoran descriptivamente las 15 combinaciones algoritmo-variante frente al conjunto original; relativas con identidad mejoran 12. Añadir identidad a las relativas solo mejora 6 de 15. La evidencia apunta a la representación relativa como cambio más relevante que el identificador, pero **ninguno de los 81 intervalos de diferencias macro excluye cero**. No se demuestra una mejora global ni equivalencia exacta entre enfoques.

El mayor AUC macro nuevo es 0,5157, con bosque aleatorio, relativas, identidad y retardo. Su intervalo es [0,4863; 0,5386]. Frente al mismo algoritmo conjunto con retardo, la diferencia es +0,0066, con intervalo [−0,0163; 0,0294]. Su exactitud macro es 52,35 %, exactitud equilibrada 50,13 % y MCC medio 0,0019; la referencia mayoritaria alcanza 53,89 % de exactitud. Este máximo descriptivo no acredita una ventaja global en aciertos.

En regresión logística, las interacciones relativas elevan el AUC macro híbrido de 0,4980 a 0,5059 y el de retardo de 0,4981 a 0,5076, pero reducen el de la base de 0,5010 a 0,4978. Ninguno de los seis intervalos macro que comparan interacciones con su versión de solo identidad excluye cero.

#### 5.3.7 Resultado exploratorio en NVDA

La comparación del mismo bosque aleatorio híbrido en las mismas fechas permite separar los cambios:

| Enfoque | AUC en NVDA |
| --- | ---: |
| Conjunto original | 0,5180 |
| Individual | 0,5026 |
| Original + empresa | 0,5132 |
| Relativas | 0,5711 |
| Relativas + empresa | 0,5717 |

Relativas con identidad obtiene un intervalo de AUC [0,5314; 0,6158]. La diferencia frente al conjunto original es +0,0537, con intervalo [0,0147; 0,0980]. Sin embargo, añadir identidad a las relativas aporta solo +0,00065, con intervalo [−0,0307; 0,0322]. La mejora observada no puede atribuirse principalmente al identificador ni, por esta comparación, al sentimiento: cambia la representación y ambos modelos comparados ya incorporan noticias.

El AUC de la configuración con relativas e identidad es 0,5783, 0,5429 y 0,5936 en los tres bloques. Su exactitud es 55,88 %, frente a 54,97 % de la clase mayoritaria, equivalente a cinco aciertos adicionales en 553 sesiones. La exactitud equilibrada es 54,75 % y MCC 0,0972. Un AUC de 0,5717 no significa acertar el 57,17 %.

El patrón no es general: AAPL empeora en los promedios de las tres variantes; MSFT y TSLA presentan cambios pequeños o mixtos. En el contraste principal por empresa hay ocho intervalos positivos, todos en NVDA, y uno negativo en AAPL. Son comparaciones correlacionadas y sin corrección por multiplicidad, no ocho confirmaciones independientes. El caso de NVDA se destaca después de explorar muchas combinaciones sobre fechas conocidas y requiere confirmación en fechas nuevas.

Fuente: [análisis de identidad y relativas](../reports/experiments/ticker-aware-full-20260916/analysis.md), sus tablas `metrics/` y el [cuaderno 04](../notebooks/04_ticker_aware_models.ipynb). Estos resultados amplían la memoria sin reemplazar la selección histórica de `reports/final/`.

#### 5.3.8 Primera ronda de mejoras: resultado agregado

La deduplicación reduce el corpus alineado de 46.014 a 42.734 registros: elimina 3.280, de los cuales 3.103 corresponden a 2025. Esto reduce repeticiones literales, pero no certifica cobertura exhaustiva ni identifica todos los eventos repetidos semánticamente. Las entradas originales se conservan.

Media de AUC macro entre los tres algoritmos de esta ronda:

| Variante | Referencia | Deduplicación | Tres años | Cinco años | Enriquecido | Selección macro | Ajuste ampliado |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 0,5039 | 0,5039 | 0,4970 | 0,5023 | 0,5039 | 0,5039 | 0,4995 |
| Híbrido | 0,5080 | 0,5056 | 0,4951 | 0,5039 | 0,5046 | 0,5046 | 0,5015 |
| Con retardo | 0,5040 | 0,5079 | 0,4979 | 0,5070 | 0,5007 | 0,5007 | 0,4996 |

Las ventanas de tres años empeoran las nueve combinaciones algoritmo-variante frente a la deduplicación con historia completa. El bloque enriquecido solo mejora una de las seis combinaciones con noticias, el bosque híbrido. La selección macro elige los mismos parámetros y produce las mismas predicciones que el criterio agrupado con la rejilla pequeña. La búsqueda ampliada mejora la puntuación interna en 23 de 27 ajustes predictivos, pero solo mejora dos de las nueve combinaciones externas agregadas respecto a la etapa de selección macro.

Los tres contrastes principales del híbrido muestran diferencias negativas cuyos intervalos incluyen cero. El único intervalo macro nominalmente positivo entre 63 contrastes corresponde a deduplicación con potenciación por histogramas y retardo: AUC 0,517173, diferencia +0,014758 e intervalo [0,000080; 0,029403]. El intervalo del propio AUC [0,495506; 0,546076] contiene 0,5. El límite inferior de la diferencia está casi en cero y no hay corrección por multiplicidad; no se considera una mejora robusta. La ronda completa no justifica sustituir automáticamente la configuración anterior.

Fuente: [informe de la ronda](../reports/experiments/first-round-full-20260916/analysis.md) y [cuaderno 05](../notebooks/05_controlled_improvement_round.ipynb).

#### 5.3.9 Aporte individual de las variables de sentimiento

La tabla muestra diferencias de AUC macro, promediadas entre los tres algoritmos únicamente como resumen descriptivo. No representa una combinación de modelos ni dispone de un intervalo propio. Un valor positivo al añadir sugiere utilidad en ese contexto; un valor positivo al retirar sugiere perjuicio dentro del bloque completo.

| Variable | Añadir: híbrido | Añadir: retardo | Retirar: híbrido | Retirar: retardo |
| --- | ---: | ---: | ---: | ---: |
| Dispersión del sentimiento | +0,002037 | −0,004899 | +0,000015 | +0,002836 |
| Intensidad absoluta | +0,003609 | −0,002369 | +0,002242 | +0,001187 |
| Sorpresa del sentimiento | +0,001186 | −0,003151 | −0,000233 | +0,001029 |
| Historia disponible | +0,001750 | −0,000591 | −0,000290 | −0,000543 |
| Sorpresa del volumen de noticias | +0,006484 | −0,001424 | +0,001661 | +0,002466 |

La sorpresa del volumen es la adición con mayor mejora media sin retardo y mejora los tres algoritmos, aunque sus tres intervalos incluyen cero. La intensidad absoluta destaca en el bosque híbrido: AUC 0,520192 frente a 0,504481, diferencia +0,015711 e intervalo [0,000408; 0,026242]. Es el único contraste macro entre los 60 cuyo intervalo no incluye cero. El intervalo del propio AUC [0,498521; 0,540210] todavía contiene 0,5. Mejora en AAPL, NVDA y TSLA, pero empeora en MSFT; no es un beneficio uniforme.

Las cinco adiciones con retardo empeoran en promedio entre algoritmos, especialmente la dispersión. Esto no demuestra que todo retardo sea perjudicial: se comparan familias concretas contra una referencia que ya incorpora dos variables retardadas. Al retirar variables del bloque completo aparecen mejoras pequeñas, pero ninguno de los 30 intervalos macro de retirada excluye cero.

Que una variable ayude sola y no dentro del bloque completo es compatible con redundancia, interacciones o diferencias entre los ajustes de referencia. Cambiar el número de columnas también afecta al muestreo de variables del bosque. La ablación no identifica mecanismos causales. No se elimina ninguna variable automáticamente ni se construye una combinación ganadora seleccionando sobre estas mismas fechas.

Fuente: [informe de ablación](../reports/experiments/variable-ablation-full-20260917/analysis.md) y [cuaderno 06](../notebooks/06_sentiment_variable_ablation.ipynb), con resultados por algoritmo, empresa y bloque.

### 5.4 Discusión

#### 5.4.1 Respuesta a la pregunta principal

No se ha demostrado que añadir el sentimiento disponible mejore de forma robusta la predicción de la siguiente sesión. El resultado depende de algoritmo y representación, con AUC próximos a 0,5 y diferencias inciertas. La conclusión se refiere a estas empresas, fuente, periodo y horizonte.

La hipótesis de persistencia tampoco queda respaldada de forma concluyente. El ensayo con muchas variables retardadas produjo una mejora puntual en algunos modelos, pero la comparación controlada principal no confirmó una ventaja. Ambos ensayos difieren en calendario, purga, parametrización, conjunto de variables y forma de reentrenamiento. No es válido atribuir la caída desde 0,5217 a una única corrección.

#### 5.4.2 Explicaciones posibles, no demostradas

La señal puede ser débil a un día; el sentimiento agregado puede perder matices; las noticias pueden describir hechos ya reflejados en precios; y la cobertura puede introducir cambios de distribución. También pueden influir el modelado conjunto de empresas y una búsqueda de parámetros limitada. Estos mecanismos son hipótesis compatibles con los resultados, no conclusiones causales del experimento.

Las importancias de árboles o los coeficientes absolutos del modelo logístico describen el uso interno de variables. No prueban que una noticia cause un movimiento ni que una variable generalice fuera de muestra. La interpretación debe considerar correlación y escalas, especialmente cuando se comparan niveles de precio y sentimiento.

#### 5.4.3 Contribución de ingeniería

La revisión sustituyó una evaluación más frágil por un procedimiento verificable: horarios explícitos, fronteras purgadas, selección interna, comparaciones pareadas, artefactos aislados y pruebas. La calidad del software y la trazabilidad mejoraron sin que mejoraran necesariamente las métricas predictivas. Diferenciar esas dos dimensiones es una aportación del trabajo.

#### 5.4.4 Qué aportan los nuevos experimentos

Separar empresas prueba una hipótesis distinta de añadir noticias. Su resultado muestra que una mayor especialización no garantiza una mejora: se elimina información de otros activos y se reduce el tamaño de entrenamiento. Esa pérdida de datos es una explicación posible, no un mecanismo demostrado. El segundo experimento conserva el panel para estudiar otra vía de adaptación.

Las relativas describen posiciones y cambios comparables entre activos con niveles de precio y volumen distintos. Pueden facilitar patrones compartidos, mientras que un indicador binario por sí solo no transforma esas escalas. Los resultados son compatibles con esa interpretación, pero no permiten atribuir causalmente toda la mejora a un único mecanismo: se sustituyen varias variables simultáneamente y la selección interna sigue optimizando AUC agrupado.

La decisión es conservar ambos experimentos como evidencia, no reemplazar automáticamente el modelo conjunto ni presentar NVDA como un ganador validado. Su valor académico reside en acotar hipótesis y documentar resultados favorables y desfavorables con el mismo protocolo. Las conclusiones sobre sentimiento, especialización y representación se mantienen separadas.

#### 5.4.5 Qué cambia tras la ronda y las ablaciones

Las nuevas pruebas acotan explicaciones que antes eran solo propuestas. Alinear el criterio interno no cambia las predicciones con los candidatos pequeños, ampliar la búsqueda mejora principalmente la puntuación interna y reducir la historia no aporta una ventaja general. Enriquecer el sentimiento como bloque tampoco ayuda de forma uniforme; separar sus variables permite localizar señales más concretas sin convertirlas en conclusiones robustas.

Se conservan la auditoría, los controles y los resultados como aportaciones metodológicas. La sorpresa del volumen y la intensidad absoluta son hipótesis para una evaluación posterior prefijada, no ganadores confirmados. Incorporar los experimentos al código principal no implica adoptar sus transformaciones como configuración operativa ni modificar `reports/final/`.

### 5.5 Diagnósticos de las predicciones conservadas

La selección de figuras es explícitamente descriptiva: se compara el bosque híbrido sin extras con el mismo enfoque al añadir intensidad absoluta, por ser el contraste destacado en la ablación ya observada. No es una nueva validación ni una selección de umbral. Se utilizan las mismas 553 sesiones por empresa y el umbral fijo 0,5.

![Curvas ROC por empresa del contraste de intensidad absoluta](figures/memoria/roc.png)

**Figura 5.1.** Curvas ROC calculadas con las predicciones externas guardadas. La diagonal representa ausencia de discriminación en AUC, no una estrategia económica. Los AUC de cada panel se calculan dentro de una empresa; no deben confundirse con AUC agrupado entre empresas.

![Curvas precisión-sensibilidad por empresa](figures/memoria/precision_recall.png)

**Figura 5.2.** Curvas de precisión frente a sensibilidad. La referencia horizontal es la prevalencia de subidas de cada empresa. Se muestra precisión media (`average precision`, AP), que no equivale necesariamente al área trapezoidal interpolada. Las curvas no se utilizan para reajustar el umbral sobre la evaluación externa.

![Matrices de confusión antes y después de añadir intensidad absoluta](figures/memoria/confusion.png)

**Figura 5.3.** Recuentos de clase real frente a predicha; filas 0 y 1, columnas 0 y 1. Cada matriz contiene 553 observaciones. La clase 0 incluye igualdad y bajada. Los cambios de aciertos complementan el AUC: mejorar ordenación de probabilidades no implica mejorar todos los errores al umbral elegido.

### 5.6 Monitorización de candidatos e interpretación

![Puntuación interna de candidatos en la búsqueda ampliada](figures/memoria/candidatos.png)

**Figura 5.4.** AUC macro interno medio por candidato y bloque externo, para el híbrido en la etapa de ajuste ampliado. Los identificadores corresponden a configuraciones completas, incluidas ventanas, registradas en el manifiesto; no son valores de un único hiperparámetro. Cada columna es la media de tres particiones internas. El gráfico permite detectar sensibilidad a candidatos, no atribuir causalmente cambios a un parámetro aislado. Se muestran todos los candidatos de los tres algoritmos de esa ronda, no solo el elegido.

No se presentan estos mapas como curvas de aprendizaje. Una curva que relacione tamaño de entrenamiento y error requeriría reajustar modelos sobre tamaños comparables y registrar resultados internos y externos; eso todavía no se ha ejecutado. Los mapas de los dos candidatos de los cinco algoritmos de la revisión inicial pueden reconstruirse a partir de sus trazas, pero no equivalen a una búsqueda densa de cada parámetro.

Los desgloses por activo y bloque de 5.3, junto con los cuadernos 03–06, permiten estudiar heterogeneidad sin elegir únicamente la empresa más favorable. Los informes completos conservan todas las configuraciones; las figuras anteriores son una selección de lectura. El [generador documental](figures/memoria/generar_figuras.py) conserva las fuentes y comprobaciones de estas figuras y no modifica los artefactos experimentales.

## 6. Especificación de requisitos

### 6.1 Requisitos verificables

| Requisito | Solución implementada | Evidencia |
| --- | --- | --- |
| Relacionar precios y noticias por empresa | Claves `ticker`, `Date` y asignación bursátil | `src/data/point_in_time.py` |
| Evitar datos posteriores al instante de decisión | Primer cierre igual o posterior a la marca temporal | Pruebas de calendario y asignación |
| Evitar etiquetas solapadas en validación | Purga con `target_end < inicio_validación` | `src/models/temporal_validation.py` |
| Comparar sobre el mismo universo | Predicciones pareadas por empresa y fecha | Validación de claves y objetivo |
| Aislar imputación y escalado | Cadena de preprocesamiento y modelado ajustada en cada entrenamiento | `src/models/train_models.py` |
| No sobrescribir experimentos anteriores | Identificador nuevo y rechazo de colisiones | `src/experiments/artifacts.py` |
| Conservar resultados auditables | CSV, manifiestos y hashes SHA-256 | `reports/experiments/` |
| Facilitar la explicación | Memoria, guías y seis cuadernos de análisis | `docs/` y `notebooks/` |

### 6.2 Actores, alcance y casos de uso

El actor principal es el investigador que configura y ejecuta experimentos. El tutor o revisor consulta memoria, cuadernos e informes. Los proveedores externos suministran datos, pero no deciden qué modelo se selecciona. No existe un actor de operador bursátil ni integración de órdenes con un intermediario.

```mermaid
flowchart LR
    I[Investigador] --> U1([Adquirir y validar datos])
    I --> U2([Preparar panel temporal])
    I --> U3([Ejecutar comparación])
    I --> U4([Verificar artefactos])
    R[Revisor] --> U5([Consultar resultados y memoria])
    P[Proveedores externos] --> U1
    U1 --> U2
    U2 --> U3
    U3 --> U4
    U4 --> U5
```

**Figura 6.1.** Vista de actores y casos de uso, representada con nodos y relaciones en Mermaid; no es una interfaz gráfica implementada ni un diagrama UML normativo de casos de uso.

| Caso | Precondición | Flujo principal | Resultado o excepción |
| --- | --- | --- | --- |
| CU1. Preparar datos | Entradas disponibles y configuración de zona horaria | Validar, asignar al cierre, agregar y unir por empresa-sesión | Panel y registros no alineados; rechazo ante inconsistencia |
| CU2. Ejecutar experimento | Panel válido e identificador libre | Separar fechas, purgar, seleccionar internamente y predecir | Modelos y predicciones externas; ejecución fallida identificada |
| CU3. Comparar enfoques | Mismas claves, etiquetas y bloques | Calcular métricas y remuestreo pareado | Diferencias e intervalos; error si faltan observaciones |
| CU4. Revisar resultados | Informes y procedencia accesibles | Abrir cuadernos, contrastar hashes y consultar desgloses | Evidencia interpretable, sin llamadas obligatorias a proveedores |

### 6.3 Requisitos no funcionales y aceptación

| ID | Requisito | Criterio de aceptación | Evidencia o límite |
| --- | --- | --- | --- |
| RNF1 | Trazabilidad | Cada ejecución conserva configuración, entradas, versiones y huellas | Manifiestos; no recuperan datos ausentes |
| RNF2 | Reproducibilidad local | Modelos guardados reproducen probabilidades dentro de la tolerancia registrada | Verificaciones por ejecución; dependiente del entorno |
| RNF3 | Integridad temporal | Ninguna etiqueta de entrenamiento alcanza el bloque validado | Pruebas de purga y fronteras |
| RNF4 | Conservación histórica | Una ejecución no sobrescribe otra con el mismo identificador | Rechazo de colisiones en `create_run` |
| RNF5 | Protección de credenciales | Claves fuera del repositorio y de los informes | Variables de entorno; no sustituye auditoría de seguridad completa |
| RNF6 | Mantenibilidad | Separación entre adquisición, modelos, experimentos y presentación | Módulos y pruebas focalizadas |
| RNF7 | Legibilidad | Figuras con unidades, referencias y procedencia; texto en español | Memoria y seis cuadernos; revisión humana pendiente |

No se declara disponibilidad continua, latencia garantizada ni escalabilidad de servicio web: no existen pruebas de carga ni despliegue de ese tipo. El requisito de calidad experimental se cumple documentando resultados válidos, aunque no mejoren el modelo de referencia.

## 7. Análisis del sistema

### 7.1 Arquitectura

```text
Precios descargados + noticias con sentimiento por empresa
                         |
          Validación, indicadores y etiqueta
                         |
     Alineación con el cierre y agregación diaria
                         |
           Variantes de variables comparables
                         |
   Selección interna -> entrenamiento -> bloque externo
                         |
       Predicciones, métricas e incertidumbre
                         |
              Informes y cuadernos
```

`src/data/` contiene adquisición y preparación; `src/models/`, construcción de estimadores, validación y evaluación; `src/experiments/`, orquestación de la revisión y gestión de artefactos; `src/visualization/`, figuras. Los cuadernos de análisis leen resultados o reconstruyen datos mediante funciones existentes; no duplican el entrenamiento completo.

Las dependencias se fijan en `requirements.txt`. El entorno de trabajo utilizado en las verificaciones es Python 3.13. Pandas y NumPy soportan transformaciones tabulares; scikit-learn, XGBoost y LightGBM, modelado; Matplotlib y Seaborn, gráficos; las librerías de calendario proporcionan sesiones y cierres.

```mermaid
flowchart TB
    P[Precios y noticias guardados] --> D[Componente de datos: src/data]
    C[Calendario bursátil] --> D
    D --> E[Orquestación: src/experiments]
    E --> M[Validación y modelos: src/models]
    M --> E
    E --> A[Gestión de artefactos: artifacts.py]
    A --> R[Informes, manifiestos y predicciones]
    A --> L[Datos y modelos locales]
    R --> V[Visualización: src/visualization]
    R --> N[Cuadernos y memoria]
    V --> N
```

**Figura 7.1.** Vista de componentes y dependencias. Son módulos de un proceso local, no microservicios desplegados. La orquestación controla el experimento y la gestión de artefactos conserva su procedencia; los cuadernos leen resultados sin constituir una segunda implementación del entrenamiento.

### 7.2 Persistencia

Los informes se separan en `reports/historical/`, `reports/experiments/<id>/` y `reports/final/`. Esta última carpeta contiene una selección para la memoria, no una nueva evaluación. Los modelos se guardan en `models/experiments/<id>/`; los conjuntos de datos de ejecución, en `data/experiments/<id>/`; y las instantáneas de código, en `artifacts/snapshots/<id>/`. Las verificaciones técnicas se distinguen en `artifacts/verification/`.

Los `.joblib` y los artefactos locales pesados están excluidos del seguimiento ordinario de Git. Se conservan físicamente en el equipo. Esta decisión reduce ruido y tamaño en el repositorio, pero exige preservar los datos originales y el entorno para una reproducción externa; un manifiesto no sustituye a los archivos que identifica.

### 7.3 Verificación y reproducibilidad

El conjunto local de pruebas contiene 58 pruebas superadas, frente a las 28 de la revisión inicial. Cubre calendario regular, fines de semana, cierres anticipados, cambios horarios, marcas temporales, duplicados, objetivo, purga, retardos por empresa, emparejamiento de predicciones, modelo de referencia, LightGBM, descargas y organización de artefactos. Las ampliaciones comprueban aislamiento por empresa, noticias compartidas con puntuaciones distintas, identidad, interacciones, transformaciones causales, ausencia de noticias en la base, referencias incompatibles y remuestreo ponderado. La última ronda añade deduplicación, ventanas, nuevas representaciones, selección macro y aislamiento de las familias de variables. Se ejecuta mediante `python -m unittest discover -s tests -v`.

En la revisión completa se verificó que los 183 modelos guardados reproducían las probabilidades y clases almacenadas, y que las fronteras de los 1.080 ajustes internos y los 183 externos respetaban la condición de purga. Se verificaron ejecuciones reducidas independientes del protocolo y del flujo tradicional.

El experimento por empresa verificó 240 modelos guardados, incluidos los de referencia conjunta, y 1.350 evaluaciones internas. El de identidad añadió 168 modelos y 918 evaluaciones internas; sus 194.656 predicciones incluyen las referencias reutilizadas, no otras tantas observaciones independientes. Los modelos nuevos reproducen las probabilidades guardadas con tolerancia absoluta de 10⁻¹². La recarga de `features.csv` utiliza `float_precision="round_trip"`: un redondeo al leer puede alterar la rama elegida por un árbol cuando una variable está junto a un umbral.

El remuestreo optimizado convierte fechas repetidas en pesos y reproduce los 60 intervalos individual-conjunto de la implementación previa, con diferencia máxima aproximada de 2,6 × 10⁻¹⁶. No modifica el procedimiento estadístico para mejorar resultados. Los cuadernos 03 y 04 están ejecutados y verifican claves, métricas y huellas de informes. `.gitattributes` conserva los bytes de estos informes para evitar invalidar sus hashes al cambiar finales de línea entre sistemas.

La primera ronda verifica 210 modelos guardados, 2.214 evaluaciones internas y 154.840 predicciones pareadas. La ablación verifica otros 180 modelos, 36 referencias reutilizadas y 159.264 predicciones. Estas filas repiten las mismas sesiones para múltiples configuraciones, no amplían el tamaño de la muestra independiente. Se comprueban parámetros heredados, columnas, fronteras purgadas y AUC recalculados. Los cuadernos 05 y 06 se ejecutan desde la raíz y desde su directorio; sus tablas, figuras, tildes y enlaces se revisan. Las huellas de informes también se contrastan con los bytes preparados para Git.

La reorganización conservó 486 archivos entre resultados y modelos históricos, comprobando su contenido mediante hashes. Los manifiestos nuevos registran rutas relativas a la raíz, versiones, parámetros, estado, entradas y salidas. Los anteriores se conservaron en instantáneas. Las instantáneas de fuente no estaban disponibles cuando comenzó el primer experimento completo: este conserva hashes del código original, mientras que la copia de código se comprobó en ejecuciones posteriores.

Existe una configuración de GitHub Actions para las pruebas. Las consultas realizadas al integrar la rama no devolvieron ejecuciones remotas; por tanto, esta memoria acredita pruebas locales, no una ejecución remota de integración continua superada.

La reproducibilidad tiene tres niveles distintos: inspeccionar informes guardados, repetir cálculos con los mismos datos y reconstruir todo desde los proveedores. Los dos primeros tienen controles locales; el tercero sigue condicionado por disponibilidad, revisiones, licencias y cobertura del proveedor. Los hashes permiten detectar cambios, pero no recuperar archivos ausentes.

### 7.4 Proceso de desarrollo y decisiones

El desarrollo fue incremental mediante tareas, ramas de funcionalidad, registros de cambios identificables y fusiones de ramas. El historial conserva, entre otras etapas, indicadores, conjuntos de datos, modelos base e híbridos, evaluación, ajuste temporal y retardos. El trabajo acumulado de retardos, revisión y organización se integró en `main` mediante la fusión `2f1111d`, con implementación en `1ffdc1b`. La remodelación documental se realizó en `docs` y se integró mediante `3efa444`. Los experimentos adicionales de la tarea #47 se desarrollaron en `feature/per-company-temporal-evaluation`: `abdcce5` incorpora la evaluación independiente y `06cbd00` la identidad y las variables relativas; se integraron mediante `00c0146`.

La tarea #48 se desarrolló en `feature/controlled-improvement-round` con commits separados para auditoría (`05d346c`), ventanas (`bbf9836`), representación (`d136944`), selección (`037e741`), resultados (`c23198c`), ejecutor de ablación (`623d8a6`) e informe individual (`c57240e`). La actualización de memoria `2fb2568` precedió a la integración mediante `3ca4e46`, autorizada por el autor. Se conservan las ejecuciones históricas y no se altera la selección de `reports/final/`. La presente reestructuración documental se realiza posteriormente en la rama `docs`.

| Decisión | Motivo | Consecuencia o compromiso |
| --- | --- | --- |
| Clasificar dirección a una sesión | Problema acotado y comparación clara | No se estima magnitud ni rentabilidad |
| Usar cierre ajustado | Coherencia de retorno y etiqueta con ajustes | El histórico puede revisarse retrospectivamente |
| Excluir SPY del modelado | Mantener señal corporativa comparable | Menor diversidad del universo |
| Sentimiento del proveedor | Viabilidad y trazabilidad de datos | Calidad del procesamiento del lenguaje natural no auditada independientemente |
| Conservar días sin noticias | No perder observaciones financieras | Ausencia y falta de cobertura pueden confundirse |
| Elegir AUC y métricas equilibradas | Evitar interpretar solo F1 positivo | Se necesitan varios resúmenes complementarios |
| Purga y validación anidada | Controlar disponibilidad de etiquetas y selección | Mayor coste computacional |
| Retardos separados en la revisión | Aislar representaciones concretas | No reproduce exactamente el ensayo de 92 variables |
| Mantener resultados históricos | Explicar la evolución sin borrar evidencia | Deben rotularse para evitar mezclas |
| Probar estimadores independientes | Estudiar especialización por activo | Menos datos por ajuste; sin mejora media general |
| Separar identidad y representación relativa | Distinguir adaptación y diferencias de escala | Más comparaciones exploratorias; no se declara un ganador global |
| Conservar cuadernos 03 y 04 separados | Comparar hipótesis sin sobrescribir resultados | AUC macro y agrupado deben distinguirse explícitamente |
| Separar deduplicación, ventanas, representación y ajuste | Localizar cambios frente a referencias comunes | Más selección interna no implica mejora externa |
| Heredar parámetros en la ablación individual | Aislar cambios de columnas dentro de cada contraste | No optimiza cada subconjunto; depende de su contexto |
| Conservar cuadernos 05 y 06 separados | Distinguir cambios conjuntos y aportes individuales | No se selecciona automáticamente una combinación ganadora |
| Excluir modelos de Git ordinario | Reducir binarios y ruido experimental | Requiere gestión externa de artefactos |

El asistente de programación se utilizó para implementación, revisión, experimentos y documentación bajo decisiones del estudiante. La memoria debe reflejar ese uso con arreglo a las indicaciones académicas aplicables; este borrador no atribuye al estudiante verificaciones personales que no estén documentadas. Las afirmaciones científicas y la versión entregada requieren su revisión y defensa.

### 7.5 Secuencia de una ejecución temporal

```mermaid
sequenceDiagram
    actor I as Investigador
    participant E as Ejecutor del experimento
    participant A as Gestión de artefactos
    participant D as Constructor de datos
    participant V as Validación temporal
    participant M as Cadena de modelado
    participant R as Evaluación
    I->>E: Configuración e identificador nuevo
    E->>A: create_run
    A-->>E: Directorio y manifiesto inicial
    E->>D: build_dataset o referencia verificada
    D-->>E: Panel y procedencia
    loop Cada bloque externo
        E->>V: Seleccionar pasado y purgar etiquetas
        V-->>E: Entrenamiento y fechas de validación
        opt Experimento con selección interna
            E->>V: Particiones internas y candidatos
            loop Cada candidato y partición
                E->>M: Ajustar preprocesamiento y estimador
                M-->>E: Probabilidades internas
            end
        end
        E->>M: Reajustar en pasado admisible
        M-->>E: Predicciones del bloque externo
        E->>A: Guardar modelo, parámetros y predicciones
    end
    E->>R: Comparar filas pareadas y métricas
    R-->>E: Tablas e intervalos
    E->>A: finish_run y huellas de salidas
    E-->>I: Informes de ejecución completa
```

**Figura 7.2.** Secuencia del flujo experimental. La ablación omite la búsqueda interna y hereda parámetros de referencias verificadas. Una excepción impide presentar la ejecución como completa; no equivale a un resultado negativo del modelo. El remuestreo se aplica a predicciones guardadas, no vuelve a entrenar todos los estimadores.

### 7.6 Modelo conceptual de clases y datos

```mermaid
classDiagram
    class Empresa {
        ticker: string
    }
    class Sesion {
        fecha: date
        cierre_utc: datetime
    }
    class RegistroNoticiaEmpresa {
        url: string
        titulo: string
        publicacion_utc: datetime
        sentimiento: float
        relevancia: float
    }
    class Observacion {
        fecha: date
        target: int
        target_end: date
        variables: vector
    }
    class Ejecucion {
        identificador: string
        configuracion: dict
        estado: string
        hashes: dict
    }
    class Prediccion {
        enfoque: string
        variante: string
        algoritmo: string
        bloque: int
        probabilidad: float
        clase: int
    }
    Empresa "1" --> "0..*" RegistroNoticiaEmpresa : mencionada en
    Empresa "1" --> "0..*" Observacion : identifica
    Sesion "0..1" <-- "0..*" RegistroNoticiaEmpresa : asignada a
    Sesion "1" --> "0..*" Observacion : fecha
    Observacion "1" <-- "0..*" Prediccion : evaluada mediante
    Ejecucion "1" --> "0..*" Prediccion : conserva
```

**Figura 7.3.** Diagrama de clases conceptual del dominio. Estas entidades se materializan principalmente como filas de `DataFrame`, CSV y documentos JSON, no como seis clases Python implementadas. La asociación opcional con sesión representa noticias sin cierre posterior disponible. Una predicción también queda identificada por enfoque, variante y algoritmo dentro de la ejecución; una observación puede recibir muchas predicciones comparables.

En el código sí existe `TrainResult`, una estructura inmutable con nombres y rutas de salida del flujo tradicional. El preprocesamiento utiliza `Pipeline`, `SimpleImputer`, `StandardScaler` y estimadores de las bibliotecas. No se inventa una jerarquía orientada a objetos para describir un proyecto predominantemente funcional. La [construcción real](../src/models/train_models.py) y los [contratos del panel](../src/models/temporal_validation.py) son la referencia para implementación.

### 7.7 Trazabilidad de verificación y fallos

| Comportamiento | Verificación existente | Qué no demuestra |
| --- | --- | --- |
| Calendario y marcas temporales | `tests/test_temporal_pipeline.py` | Disponibilidad histórica real del proveedor |
| Aislamiento de noticias y empresas | `tests/test_per_company.py`, `tests/test_ticker_aware.py` | Generalización a otras empresas |
| Deduplicación conservadora | `tests/test_news_quality.py` | Equivalencia semántica de textos |
| Ventanas y selección temporal | `tests/test_training_windows.py`, `tests/test_round_selection.py` | Que una ventana sea económicamente óptima |
| Variables y ablaciones | `tests/test_sentiment_representation.py`, `tests/test_variable_ablation.py` | Causalidad o utilidad fuera de las fechas observadas |
| Rutas y preservación de artefactos | `tests/test_artifact_layout.py` | Recuperación automática de archivos perdidos |

La verificación combina pruebas automatizadas, reproducción de probabilidades de modelos guardados, cotejo de tablas y revisión de figuras. Un error de entrada o una referencia incompatible debe detener la comparación; un AUC bajo con entradas válidas se conserva como resultado. Son estados distintos y su separación evita confundir calidad del software con éxito predictivo.

## 8. Conclusiones

### 8.1 Conclusiones del trabajo

Se ha construido un sistema modular capaz de integrar precios e información de noticias, generar variables comparables y evaluar modelos de clasificación temporal. La pregunta inicial se ha estudiado mediante cinco algoritmos, una referencia trivial, ajuste de hiperparámetros y variantes de sentimiento contemporáneo y retardado.

La evidencia obtenida no acredita una ventaja predictiva robusta y general del sentimiento para la siguiente sesión en este histórico. Entrenar por empresa no mejora el promedio global. Las variables relativas producen mejoras descriptivas mayores que añadir solo identidad, pero ninguno de los 81 intervalos de diferencias macro de la ampliación excluye cero. El modelo de referencia explica por qué una exactitud o un F1 positivo aparentemente altos pueden resultar poco informativos.

NVDA presenta una señal exploratoria más favorable: el bosque aleatorio híbrido relativo con identidad alcanza 0,5717 de AUC, frente a 0,5180 del conjunto original, con un intervalo de diferencia positivo. Las relativas sin identidad ya alcanzan 0,5711, y AAPL empeora. Por tanto, se conserva este resultado como hipótesis para fechas nuevas, no como evidencia general del sentimiento, confirmación independiente ni demostración de rentabilidad.

La primera ronda no mejora el promedio general al combinar depuración, enriquecimiento y búsqueda ampliada. La ablación individual sugiere utilidad descriptiva de la sorpresa del volumen en híbridos sin retardo y de la intensidad absoluta en el bosque híbrido, que alcanza 0,5202 de AUC macro. Sin embargo, el intervalo del propio AUC contiene 0,5, las diferencias no son uniformes y hay múltiples comparaciones. Las cinco adiciones con retardo empeoran en promedio. Se conservan como resultados exploratorios, sin eliminar variables ni cambiar automáticamente el modelo seleccionado.

El resultado no invalida el TFG ni prueba que las noticias no afecten a los mercados. Delimita lo que puede sostenerse con el experimento realizado. La principal contribución es un procedimiento de comparación más controlado, verificable y documentado, junto con un análisis explícito de sus límites.

### 8.2 Limitaciones y trabajo futuro

#### 8.2.1 Validez interna

El histórico externo se consultó durante el desarrollo. La validación anidada posterior reduce contaminación dentro de una ejecución, pero no deshace decisiones motivadas por resultados anteriores. Tampoco los intervalos de predicciones fijas incorporan toda la incertidumbre del entrenamiento. La comparación entre etapas modifica varios elementos simultáneamente.

Los experimentos por empresa e identidad reutilizan esas fechas. La selección posterior del caso de NVDA y la abundancia de contrastes impiden tratar sus intervalos nominales como confirmación independiente. La coherencia entre bloques no elimina este sesgo. En esas ejecuciones, el criterio interno de AUC agrupado no coincide exactamente con el AUC macro externo y la búsqueda de dos configuraciones por algoritmo limita la adaptación de las variantes con interacciones. La ronda posterior estudia selección macro y candidatos ampliados sobre la representación enriquecida, no vuelve a optimizar todas las interacciones anteriores.

Las ablaciones vuelven a utilizar las mismas fechas. El único contraste macro positivo de intensidad absoluta es nominal, está cerca del límite y forma parte de 60 comparaciones sin corrección. Los hiperparámetros heredados controlan cada comparación, pero no garantizan el ajuste óptimo de los subconjuntos. Añadir una variable y retirarla del bloque completo no son experimentos simétricos ni permiten atribuir causalidad.

#### 8.2.2 Datos y generalización

Solo se estudian cuatro empresas de gran capitalización y elevada presencia mediática. La selección retrospectiva no representa todo el mercado ni incorpora empresas desaparecidas. La cobertura informativa no está certificada; títulos repetidos pueden sobreponderar eventos; las puntuaciones de sentimiento no se contrastaron con una muestra anotada independiente. La deduplicación experimental reduce repeticiones literales, pero no elimina todas las duplicaciones semánticas ni se adopta automáticamente en las ejecuciones anteriores.

El cierre ajustado descargado retrospectivamente, las revisiones del proveedor y la ausencia de marcas temporales de recepción impiden afirmar una reconstrucción perfecta de la información disponible en cada instante. El calendario y la purga corrigen riesgos concretos, no todos los sesgos posibles.

#### 8.2.3 Utilidad económica y uso responsable

No se han calculado comisiones, deslizamiento, rotación, exposición, caída máxima desde un máximo previo ni reglas de ejecución. AUC y exactitud no permiten deducir rentabilidad. Las salidas son académicas y no constituyen recomendaciones de inversión. Las claves de API permanecen fuera del repositorio; antes de redistribuir noticias o textos debe revisarse la autorización correspondiente del proveedor.


#### 8.2.4 Validación futura propuesta

Antes de ampliar otra búsqueda, se fijaría una regla de selección y una comparación principal para evaluarlas en fechas nuevas no utilizadas en estas decisiones. Las variables relativas constituyen una hipótesis prioritaria, no una garantía de mejora. Debe comprobarse si el patrón de NVDA persiste y si es específico de la empresa, del periodo o de la cobertura informativa.

La alineación con AUC macro, la búsqueda ampliada y las ventanas de tres y cinco años ya se han probado en la primera ronda, sin mejora general. Siguen pendientes una evaluación específicamente diseñada para periodos de cobertura comparable, otras empresas o un horizonte diferente. La sorpresa del volumen y la intensidad absoluta pueden motivar hipótesis prefijadas, no otra selección retrospectiva del máximo. Cualquier prueba adicional requerirá definir su protocolo antes de evaluar; no se presenta como ejecutada ni se anticipan sus resultados.

## 9. Bibliografía

La bibliografía técnica inicial se consultó el 15 de septiembre de 2026 y los nuevos antecedentes teóricos y empíricos se contrastaron el 19 de septiembre de 2026. Debe homogeneizarse al estilo exigido por la titulación. La selección no se presenta como una revisión sistemática completa ni como evidencia de haber reproducido los artículos citados.

1. Alpha Vantage. *API Documentation: News & Sentiments*. [Documentación oficial](https://www.alphavantage.co/documentation/#news-sentiment).
2. yfinance. *yfinance.download*. [Documentación del proyecto](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).
3. scikit-learn. *Metrics and scoring: quantifying the quality of predictions*, versión 1.7. [Documentación](https://scikit-learn.org/1.7/modules/model_evaluation.html).
4. scikit-learn. *TimeSeriesSplit*. [Documentación](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html). Se usa como referencia conceptual; el proyecto implementa particiones por fecha con purga propia.
5. pandas_market_calendars. *Documentación del proyecto*. [Calendarios bursátiles](https://pandas-market-calendars.readthedocs.io/en/latest/).
6. Chen, T. y Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. [Artículo](https://arxiv.org/abs/1603.02754).
7. Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. [Artículo](https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html).
8. Lo, A. W. (2007, versión de autor). *Efficient Markets Hypothesis*. Preparado para The New Palgrave: A Dictionary of Economics, segunda edición. [Texto del autor en MIT](https://web.mit.edu/~alo/www/Papers/EMH_Final.pdf).
9. Lo, A. W. (2004). *The Adaptive Markets Hypothesis: Market Efficiency from an Evolutionary Perspective*. Journal of Portfolio Management, 30, 15–29. [Resumen del autor](https://web.mit.edu/Alo/www/Papers/JPM2004.html).
10. Bollen, J., Mao, H. y Zeng, X. (2011). *Twitter mood predicts the stock market*. Journal of Computational Science, 2(1), 1–8. [Artículo y metadatos](https://arxiv.org/abs/1010.3003).
11. Hu, Z., Liu, W., Bian, J., Liu, X. y Liu, T.-Y. (2018). *Listening to Chaotic Whispers: A Deep Learning Framework for News-oriented Stock Trend Prediction*. WSDM 2018. [Versión de los autores, revisada en 2019](https://arxiv.org/abs/1712.02136).
12. Xu, Y. y Cohen, S. B. (2018). *Stock Movement Prediction from Tweets and Historical Prices*. ACL, 1970–1979. [Publicación](https://aclanthology.org/P18-1183/).
13. Fischer, T. y Krauss, C. (2018). *Deep learning with long short-term memory networks for financial market predictions*. European Journal of Operational Research, 270(2), 654–669. [Registro institucional](https://cris.fau.de/publications/208534319/).
14. Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*. [Prepublicación](https://arxiv.org/abs/1908.10063) y [modelo publicado por ProsusAI](https://huggingface.co/ProsusAI/finbert).
15. VectorBT. *Portfolio simulation: base*. [Documentación oficial](https://vectorbt.dev/api/portfolio/base/). Herramienta relacionada, no utilizada en los resultados actuales.
16. SHAP. *TreeExplainer*. [Documentación oficial](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html). Técnica relacionada, todavía no incorporada.

## 10. Anexos

### 10.1 Evidencias locales

| Evidencia | Ruta desde la raíz del repositorio |
| --- | --- |
| Configuración y variables del experimento | `reports/experiments/review-full-20260908/manifest.json` |
| Métricas globales y desgloses | `reports/experiments/review-full-20260908/metrics/` |
| Predicciones externas pareadas | `reports/experiments/review-full-20260908/predictions/predictions.csv` |
| Configuraciones seleccionadas y fronteras | `reports/experiments/review-full-20260908/tuning/selected_params.csv` |
| Búsqueda interna | `reports/experiments/review-full-20260908/tuning/inner_cv.csv` |
| Auditoría de cobertura | `reports/experiments/review-full-20260908/coverage/` |
| Evolución histórica de resultados | `reports/historical/tuning/model_progression_summary.csv` |
| Selección para la memoria | `reports/final/provenance.json` |
| Experimento independiente por empresa | `reports/experiments/per-company-20260916/analysis.md` y `manifest.json` |
| Identidad, relativas e interacciones | `reports/experiments/ticker-aware-full-20260916/analysis.md` y `manifest.json` |
| Métricas macro, por empresa e intervalos adicionales | `reports/experiments/ticker-aware-full-20260916/metrics/` |
| Cuadernos de las ampliaciones | `notebooks/03_per_company_models.ipynb`, `notebooks/04_ticker_aware_models.ipynb` |
| Primera ronda controlada | `reports/experiments/first-round-full-20260916/analysis.md` y `manifest.json` |
| Ablación individual de variables | `reports/experiments/variable-ablation-full-20260917/analysis.md` y `manifest.json` |
| Protocolos de las últimas comparaciones | `docs/first_round_protocol.md`, `docs/variable_ablation_protocol.md` |
| Cuadernos de ronda y ablación | `notebooks/05_controlled_improvement_round.ipynb`, `notebooks/06_sentiment_variable_ablation.ipynb` |
| Pruebas automatizadas | `tests/`, incluidas las pruebas temporales, de artefactos, por empresa y de identidad |

Las tablas redondean resultados guardados; las cifras completas están en los CSV. Los seis cuadernos de análisis permiten consultar dimensiones, calidad, cobertura, comparaciones e incertidumbre sin entrenar modelos ni consumir API. Los informes de las ejecuciones se conservan como documentos históricos: sus notas sobre el estado de la rama o propuestas futuras describen el momento de elaboración; esta memoria incorpora los experimentos posteriores.

### 10.2 Glosario

**Sesión:** día de negociación del calendario utilizado. **Identificador bursátil (`ticker`):** símbolo de un activo. **Variable predictora:** característica utilizada como entrada del modelo. **Objetivo (`target`):** etiqueta que se intenta predecir. **Partición:** división de entrenamiento y evaluación. **Purga:** exclusión de filas cuyo horizonte de etiqueta alcanza la evaluación. **Conjunto de prueba reservado:** observaciones separadas para evaluar. **Ablación:** comparación que modifica un bloque de variables. **Modelo de referencia:** método sencillo con el que se comparan los demás. **Remuestreo:** generación de muestras a partir de los datos para explorar incertidumbre. **Fuga de información:** utilización indebida de información futura o de evaluación.

Los nombres de algoritmos, bibliotecas, columnas, rutas y títulos bibliográficos se conservan en su forma original para facilitar su identificación en el código y en las fuentes.

### 10.3 Reproducción de las figuras de esta revisión

Las nueve figuras añadidas a la memoria se guardan en `docs/figures/memoria/`. El archivo `procedencia.json` identifica las entradas y salidas mediante SHA-256 y conserva los AUC y precisiones medias del diagnóstico. Se regeneran desde la raíz con:

```powershell
.venv\Scripts\python.exe docs/figures/memoria/generar_figuras.py
```

El generador exige los informes de las dos últimas ejecuciones y el panel local conservado de la primera ronda. Verifica sus huellas antes de leerlos. No descarga noticias, no entrena, no modifica informes anteriores ni sustituye la selección de `reports/final/`. Sin ese panel local pueden inspeccionarse las imágenes conservadas, pero no reconstruir las figuras de datos únicamente a partir de sus hashes.

Los cuatro diagramas se mantienen como bloques Mermaid editables dentro del borrador. La exportación final a Word o PDF deberá renderizarlos y revisar su paginación. Las figuras y ecuaciones deberán numerarse de forma automática al preparar el documento definitivo.

### 10.4 Revisión pendiente antes de la entrega

- Validar con el autor las 330 horas propuestas y separar dedicación acreditable de trabajo aún pendiente.
- Ampliar y revisar críticamente el estado del arte; completar las fichas comparables y homogeneizar bibliografía.
- Confirmar con el tutor la ampliación de desarrollo que se realizará. FinBERT, SHAP, otros horizontes, backtesting y aplicación siguen sin ejecutarse.
- Añadir curvas de aprendizaje solo tras realizar los entrenamientos necesarios; no reutilizar curvas de ajuste como si fueran equivalentes.
- Decidir si se necesita un calendario de resultados empresariales para analizar cobertura alrededor de esos eventos.
- Revisar diagramas, ecuaciones, unidades, referencias cruzadas y legibilidad en el formato de entrega.

Las orientaciones de extensión del tutor (20–25 páginas para estado del arte y monitorización; 15–20 para ingeniería, fundamentos matemáticos y análisis exploratorio) son objetivos editoriales, no páginas ya producidas. Esta revisión amplía el contenido y lo organiza en los diez capítulos, pero no acredita esos rangos sin maquetación ni sustituye las lecturas y pruebas pendientes por texto de relleno.
