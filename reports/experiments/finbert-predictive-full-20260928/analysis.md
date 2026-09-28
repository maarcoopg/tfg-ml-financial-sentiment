# Utilidad predictiva de FinBERT

**Issue #51. Ejecución del 28/09/2026. Comparación exploratoria, sin fusión automática.**

## Conclusión

FinBERT cambia el sentimiento y mejora descriptivamente cuatro de las seis
variantes frente a Alpha Vantage sobre exactamente las mismas noticias. **No
demuestra una mejora general y robusta frente a utilizar solo datos financieros.**
El mejor AUC macro de FinBERT es 0,5141, con boosting y retardo. No es una
probabilidad de acierto del 51,41 % ni una estimación de rentabilidad.

El contraste favorable de boosting con retardo frente a Alpha emparejado tiene
un intervalo nominal positivo. Sin embargo, su propio intervalo de AUC incluye
0,5 y su diferencia frente a la base incluye cero. Además, hay seis contrastes
principales, treinta contrastes macro totales y fechas históricas ya consultadas.
No corresponde declarar un modelo ganador confirmado.

## Qué se ha comparado

Se conservan AAPL, MSFT, NVDA y TSLA; SPY no interviene. La base tiene 9 variables
financieras relativas, los híbridos 22 y el retardo 24. Se usan tres algoritmos:
regresión logística, bosque aleatorio y boosting por histogramas.

El [protocolo](../../../docs/finbert_prediccion_protocolo.md) quedó registrado
antes del cálculo financiero, en el commit `e5d8f6a`. El ejecutor corresponde a
`eaee42f`. Solo se permite elegir entre dos configuraciones por algoritmo, usando
AUC macro en tres bloques internos temporales. Se mantienen las tres particiones
externas, con purga, preprocesamiento dentro del entrenamiento y umbral 0,5.

Todos los enfoques entrenan desde enero de 2021, después de la publicación del
checkpoint. Por tanto, las bases se han vuelto a entrenar: **no debe compararse
esta tabla con las anteriores atribuyendo todo el cambio a FinBERT**. Hay 553
sesiones externas por empresa, del 16/10/2023 al 29/12/2025: 2.212 filas por
combinación. Se realizaron 63 ajustes externos, 378 internos y tres referencias
mayoritarias. La base se ajustó una vez por algoritmo y bloque; sus copias para
el remuestreo no se cuentan como modelos nuevos.

| Bloque | Entrenamiento hasta | Final máximo de etiqueta | Evaluación | Filas de entrenamiento / evaluación |
| --- | --- | --- | --- | ---: |
| 1 | 12/10/2023 | 13/10/2023 | 16/10/2023 a 11/07/2024 | 2.796 / 740 |
| 2 | 10/07/2024 | 11/07/2024 | 12/07/2024 a 04/04/2025 | 3.536 / 736 |
| 3 | 03/04/2025 | 04/04/2025 | 07/04/2025 a 29/12/2025 | 4.272 / 736 |

## Noticias y controles

De 46.014 parejas noticia-empresa alineadas al calendario quedan 37.524
deduplicadas desde 2021. El subconjunto emparejado contiene **15.937 parejas**,
el 42,47 % de ese total, y 14.699 textos únicos. El filtro exige alias explícito
de la empresa y excluye fichas dinámicas reconocibles por título o URL. No usa
resultados bursátiles ni las anotaciones de IA. Conserva la primera noticia por
titular normalizado y empresa dentro de 24 horas, sin deduplicar retrospectivamente.

Estos recuentos describen el corpus elegible alineado, no solo las sesiones con
objetivo disponible. En el panel financiero entran 15.844 parejas emparejadas y
37.297 del conjunto completo: las restantes no tienen una fila empresa-sesión
en el panel evaluable. Sus puntuaciones quedan trazadas, pero no añaden filas
ni intervienen en los clasificadores. La cobertura mensual cuenta únicamente
las que sí pertenecen al panel.

FinBERT procesa el campo original de titular y resumen, sin prefijos ni selección
de frases. Ninguno de los textos inferidos supera el límite de 512 tokens. Se usa
`ProsusAI/finbert`, revisión `4556d13015211d73dccd3fdd39d39232506f3e43`, congelado,
FP32, atención explícita y TF32 desactivado. La GPU y CPU coinciden en las 16
etiquetas prefijadas; el máximo error de probabilidad es 0,000001967.

La puntuación es P(positivo) − P(negativo). Las variables diarias se reconstruyen
por empresa y cierre, conservando sesiones sin noticias y la relevancia del
proveedor como peso. Un mismo texto compartido recibe la misma puntuación de
FinBERT: esto **no soluciona el sentimiento dirigido a cada empresa**.

| Empresa | Parejas emparejadas | Acuerdo de etiquetas con Alpha | Correlación de Pearson |
| --- | ---: | ---: | ---: |
| AAPL | 3.311 | 55,96 % | 0,5597 |
| MSFT | 4.641 | 59,30 % | 0,5616 |
| NVDA | 5.268 | 61,96 % | 0,4515 |
| TSLA | 2.717 | 56,05 % | 0,5896 |

Son acuerdos entre sistemas, no aciertos frente a una verdad humana. El filtro
es heurístico y los resúmenes pueden haberse actualizado después de su fecha
registrada; ni la purga ni excluir fichas reconocibles certifican una reconstrucción
histórica perfecta. El umbral de 2021 evita usar el checkpoint antes de publicarse,
pero no resuelve todas las incertidumbres de sus datos de preentrenamiento.

## AUC macro: todas las variantes

Media no ponderada del AUC de las cuatro empresas sobre las predicciones externas.
No debe confundirse con el AUC agrupando empresas ni con el promedio de bloques.

| Enfoque | Regresión logística | Bosque aleatorio | Boosting |
| --- | ---: | ---: | ---: |
| Base financiera | 0,4915 | 0,5131 | 0,4991 |
| Alpha completo, sin retardo | 0,4896 | 0,5105 | 0,5189 |
| Alpha completo, con retardo | 0,4903 | 0,5174 | 0,5153 |
| Alpha emparejado, sin retardo | 0,4875 | 0,5070 | 0,5085 |
| Alpha emparejado, con retardo | 0,4864 | 0,5097 | 0,4917 |
| FinBERT emparejado, sin retardo | 0,4967 | 0,5102 | 0,5061 |
| FinBERT emparejado, con retardo | 0,4970 | 0,5037 | 0,5141 |

El filtro reduce descriptivamente los seis AUC de Alpha, aunque sus intervalos
de diferencia incluyen cero. No basta con que las noticias parezcan más relevantes
para que el clasificador mejore: se pierde volumen y cambia la cobertura. Este
control evita atribuir a FinBERT el efecto conjunto del filtrado y el sentimiento.

## Diferencia sobre las mismas noticias

Intervalos pareados nominales del 95 %, con 1.000 remuestreos de bloques de 20
sesiones compartidos entre empresas. No incluyen reentrenamiento ni corrección
por comparaciones múltiples.

| Algoritmo | Variante | FinBERT menos Alpha | Intervalo de la diferencia |
| --- | --- | ---: | --- |
| Regresión logística | Sin retardo | +0,0092 | [−0,0037; +0,0217] |
| Bosque aleatorio | Sin retardo | +0,0032 | [−0,0144; +0,0202] |
| Boosting | Sin retardo | −0,0024 | [−0,0177; +0,0178] |
| Regresión logística | Con retardo | +0,0106 | [−0,0031; +0,0228] |
| Bosque aleatorio | Con retardo | −0,0060 | [−0,0242; +0,0167] |
| Boosting | Con retardo | +0,0224 | [+0,0019; +0,0463] |

Solo uno de los seis intervalos principales excluye cero. También es el único
de los treinta contrastes macro previstos. El boosting con FinBERT y retardo
tiene intervalo de AUC [0,4927; 0,5360]; frente a su base financiera, la diferencia
es +0,0150 con intervalo [−0,0044; +0,0369]. Ninguno de los seis contrastes
FinBERT-base excluye cero. La mejora respecto a Alpha emparejado no equivale a
superar claramente el azar, los precios solos o el Alpha completo.

## Exactitud y empresas

| FinBERT | Variante | Exactitud | Exactitud equilibrada | F1 macro | MCC |
| --- | --- | ---: | ---: | ---: | ---: |
| Regresión logística | Sin retardo | 51,49 % | 50,33 % | 0,4849 | 0,0030 |
| Bosque aleatorio | Sin retardo | 51,72 % | 50,76 % | 0,5002 | 0,0147 |
| Boosting | Sin retardo | 50,81 % | 49,98 % | 0,4938 | −0,0012 |
| Regresión logística | Con retardo | 50,59 % | 49,42 % | 0,4770 | −0,0154 |
| Bosque aleatorio | Con retardo | 51,72 % | 50,15 % | 0,4840 | 0,0026 |
| Boosting | Con retardo | 51,40 % | 50,71 % | 0,5033 | 0,0144 |

La referencia de clase mayoritaria alcanza 53,89 % de exactitud, AUC 0,5 y
exactitud equilibrada 50 %. Ningún modelo de esta ronda la supera en exactitud
con el umbral prefijado. Esto no convierte a la referencia en una estrategia
rentable, pero impide presentar un 51 % de aciertos como ventaja suficiente.

| FinBERT | Variante | AAPL | MSFT | NVDA | TSLA |
| --- | --- | ---: | ---: | ---: | ---: |
| Regresión logística | Sin retardo | 0,4428 | 0,4894 | 0,5129 | 0,5419 |
| Bosque aleatorio | Sin retardo | 0,4873 | 0,4975 | 0,5299 | 0,5259 |
| Boosting | Sin retardo | 0,5002 | 0,4752 | 0,5159 | 0,5331 |
| Regresión logística | Con retardo | 0,4434 | 0,4955 | 0,5092 | 0,5399 |
| Bosque aleatorio | Con retardo | 0,4631 | 0,4868 | 0,5342 | 0,5306 |
| Boosting | Con retardo | 0,5117 | 0,4768 | 0,5145 | 0,5532 |

El desglose muestra heterogeneidad, no una mejora universal: TSLA y NVDA son
descriptivamente más favorables que AAPL y MSFT. El valor 0,5532 de TSLA no se
elige como nuevo objetivo ni se usa para reabrir el ajuste. Los valores de todos
los enfoques, empresas y bloques están en `metrics/` y en el cuaderno 09.

## Qué aporta al TFG

La ampliación completa la conexión entre el funcionamiento interno de un modelo
de lenguaje, su comportamiento lingüístico y su utilidad en una tarea posterior.
No se limita a usar una API: se conserva el checkpoint, tokenización, representación,
probabilidades, huellas de textos, caché reanudable, metadatos numéricos, agregación
causal y comparación controlada. Se preservan resultados desfavorables y controles
que permiten separar el cambio de cobertura del cambio de sentimiento.

Una explicación plausible del escaso avance es la combinación de sentimiento
general no dirigido, resúmenes cortos, pérdida de cobertura y un objetivo diario
ruidoso. Son hipótesis, no mecanismos demostrados por esta tabla. Un modelo puede
comprender mejor una frase y no aportar una señal nueva sobre el retorno siguiente.

## Evidencias y reproducción

- `manifest.json`: versiones, código, parámetros, entradas y huellas de salidas.
- `sentiment/text_scores.csv`: una inferencia por texto exacto.
- `sentiment/news_scores.csv`: trazabilidad por noticia, empresa y cierre.
- `quality/news_eligibility.csv`: decisiones de filtrado y deduplicación.
- `metrics/`: resultados completos y 150 contrastes por empresa o promedio macro.
- `tuning/`: 66 selecciones externas y 378 filas de validación interna.
- `predictions/all_predictions.csv`: predicciones de las 22 combinaciones únicas.
- `verification.json`: comprobación de entradas, fuentes, modelos y probabilidades.
- `notebooks/09_finbert_predictive_evaluation.ipynb`: lectura explicada, figuras y recálculo de AUC.

Las 104 pruebas pasan tanto en el entorno CPU original como en el entorno separado
con PyTorch 2.8.0+cu128. Los modelos `.joblib`, pesos, caché SQLite y paneles locales
siguen excluidos de Git. No se modifican resultados anteriores ni `reports/final/`.
El tiempo de máquina no se contabiliza como dedicación personal en la memoria.

Para reproducir, instalar los requisitos generales y opcionales y disponer de
las entradas locales y los pesos fijados. Usar un identificador de ejecución nuevo:

```powershell
python -m src.experiments.run_finbert --run-dir reports/experiments/finbert-predictive-NUEVA --device cpu --cache data/experiments/finbert-cache-cpu/scores.sqlite
```

La ejecución registrada utiliza `--device cuda` en un entorno separado. La caché
rechaza metadatos incompatibles: no reutilizar una caché GPU como si fuese CPU.
Los pesos deben descargarse explícitamente mediante la guía del cuaderno 07.
La caché permite recuperar lotes de inferencia terminados; los ajustes financieros
se vuelven a ejecutar en un directorio nuevo si se interrumpe una ejecución.
