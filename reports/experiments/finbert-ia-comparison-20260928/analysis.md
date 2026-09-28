# Valoración de noticias realizada por IA

**28 de septiembre de 2026. Issue #50. Sin validación humana.**

El autor delegó la revisión para evitar la carga de etiquetar 400 noticias en
CSV. El asistente leyó cada ticker, titular y resumen y dejó una razón por
etiqueta. No se entrenó FinBERT ni se cambiaron los resultados financieros.
Las [400 valoraciones](valoraciones.md) están disponibles como texto legible.

## Qué se ha valorado

| Valoración del asistente | Noticias |
| --- | ---: |
| Positiva | 94 |
| Negativa | 38 |
| Neutral | 57 |
| Información insuficiente para la empresa | 211 |
| Total | 400 |

Las 211 insuficientes representan el 52,75 % de esta muestra equilibrada. En
muchos casos el texto trata de otra empresa, un ETF o del sector sin aportar
contexto específico del activo asignado. No se convierten en neutrales ni se
supone que el artículo completo carezca de información adicional.

Quedan 100 casos evaluables de desarrollo y 89 del periodo posterior. Los 207
casos sin alias explícito resultan insuficientes en esta revisión; otros cuatro
tienen mención pero no contenido para decidir. Las 189 filas evaluables tienen
contexto extraído, de modo que aquí la comparación pareada y la general elegible
coinciden. La regla no cubre todo el panel: ofrece contexto en 193/400, no en
400/400. El 100 % de cobertura que aparece en las métricas se refiere únicamente
a las 189 filas elegibles tras excluir insuficientes.

El contenido semántico se valoró como inglés en 399 casos; una ficha con
encabezado multilingüe se marcó incierta y también insuficiente. No se suman
dos veces las exclusiones. Esta revisión no certifica el idioma del corpus entero.

## Comparación frente a la referencia IA

**Acuerdo y F1 se calculan contra las etiquetas del asistente, no contra verdad
humana.** Alpha Vantage puede tener acceso a texto distinto o más extenso.

| Partición | Sistema | N | Coincidencia | F1 macro de tres clases |
| --- | --- | ---: | ---: | ---: |
| Desarrollo | FinBERT, titular | 100 | 40,00 % | 0,4134 |
| Desarrollo | FinBERT, titular y resumen | 100 | 61,00 % | 0,5895 |
| Desarrollo | FinBERT, contexto | 100 | 65,00 % | 0,6241 |
| Desarrollo | Alpha Vantage | 100 | 67,00 % | 0,5857 |
| Desarrollo | Siempre positiva | 100 | 54,00 % | 0,2338 |
| Desarrollo | Siempre neutral | 100 | 24,00 % | 0,1290 |
| Evaluación | FinBERT, titular | 89 | 44,94 % | 0,4760 |
| Evaluación | FinBERT, titular y resumen | 89 | 67,42 % | 0,6588 |
| Evaluación | FinBERT, contexto | 89 | 61,80 % | 0,6144 |
| Evaluación | Alpha Vantage | 89 | 65,17 % | 0,6468 |
| Evaluación | Siempre positiva | 89 | 44,94 % | 0,2067 |
| Evaluación | Siempre neutral | 89 | 37,08 % | 0,1803 |

En evaluación, texto completo coincide en 60 casos y Alpha Vantage en 58: solo
dos casos de diferencia. En desarrollo, la comparación de coincidencia favorece
al proveedor, mientras F1 macro es prácticamente igual. El contexto mejora la
coincidencia en desarrollo, pero la empeora en evaluación. No se elige una
variante ganadora ni se ajusta el modelo con este panel.

Estos resultados no son comparables directamente con el 49 % inicial: aquel
valor era el acuerdo entre FinBERT y Alpha Vantage sobre las 200 noticias de
desarrollo, incluyendo las que ahora se consideran insuficientes para la empresa.

## Evaluación por empresa

| Empresa | N | Coincidencia FinBERT completo | Coincidencia Alpha Vantage | F1 FinBERT | F1 Alpha Vantage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Apple | 22 | 54,55 % | 50,00 % | 0,5473 | 0,3383 |
| Microsoft | 19 | 78,95 % | 73,68 % | 0,7765 | 0,7104 |
| NVIDIA | 27 | 74,07 % | 77,78 % | 0,6667 | 0,6601 |
| Tesla | 21 | 61,90 % | 57,14 % | 0,6157 | 0,6278 |

Los soportes son pequeños y desiguales. F1 macro pondera por igual las tres
clases, no las cuatro empresas; no debe confundirse con el promedio macro por
activo de los experimentos financieros. Las matrices y métricas por clase se
conservan en `comparison.json` y `reference_metrics.json`.

## Casos que ayudan a entender los desacuerdos

- **Netflix evita las comisiones de iTunes, fila 43:** el asistente valora el
  perjuicio para Apple como negativo; FinBERT completo y con contexto devuelven
  positivo. El éxito de una empresa puede ser el perjuicio de otra.
- **Google facilita abandonar Microsoft 365, fila 275:** el asistente asigna
  negativo para Microsoft; ambas variantes de FinBERT devuelven positivo.
  Extraer frases con el nombre de la empresa no resuelve necesariamente el
  sentimiento dirigido.
- **Demanda contra Microsoft por precios de IA, fila 260:** tanto el asistente
  como FinBERT completo y con contexto asignan negativo.
- **Nuevos vuelos de JetBlue, fila 180:** la noticia tiene tono favorable, pero
  no informa sobre Tesla. FinBERT completo devuelve positivo; la regla de
  contexto se abstiene; el asistente marca insuficiente. La predicción previa
  conocida correspondía al titular, que daba neutral, no al texto completo.

Las valoraciones de IA también pueden ser discutibles. Por ejemplo, la fila 5
combina un contratiempo técnico del iPhone con ausencia de retraso de entregas;
la fila 359 combina recomendaciones de compra y venta. Las razones se conservan
para que esos criterios sean revisables, no para convertirlos en hechos.

## Advertencia sobre fechas

Se observan discrepancias que deben investigarse antes de usar estos textos para
predecir el mercado de forma retrospectiva:

| Fila | Publicación registrada | Contenido del resumen |
| --- | --- | --- |
| 126, FIAX | 08/07/2023 | Histórico entre febrero y marzo de 2026 |
| 147, IWY | 04/04/2023 | Posiciones a 31/12/2025 |
| 369, TSLL | 31/03/2025 | Posiciones a 17/03/2026 |
| 382, RONB | 18/12/2025 | Posiciones a 26/02/2026 |

Una causa posible son páginas dinámicas actualizadas; no se ha verificado el
origen. No se corrigen fechas ni se excluyen retroactivamente estos casos para
mejorar métricas. La clasificación lingüística no demuestra disponibilidad del
texto en la fecha registrada. No se extrapola el número de ejemplos al corpus.

## Procedencia y limitaciones

Las etiquetas se fijaron en `d46e2d6` y el evaluador en `d15b9a6`. La vinculación
entre número de fila e ID se comprueba mediante la huella de la plantilla
original. El manifiesto guarda entradas, salidas, código y pesos.

Es una única referencia de IA, sin doble anotación ni validación humana. No se
conoce el corpus de preentrenamiento del asistente y no se descarta conocimiento
previo de noticias. El contexto incluía estadísticas agregadas de desarrollo y
una predicción individual anterior (fila 180), lo que impide afirmar cegamiento
total. El panel posterior ya tiene predicciones generadas y consultadas: no
permanece sin inspeccionar para decisiones futuras.

**Conclusión:** la muestra permite diagnosticar problemas de contexto por empresa
y casos concretos de sentimiento dirigido. No acredita que FinBERT supere al
proveedor de manera general, ni que mejore la predicción bursátil. El trabajo
manual del autor se evita sin simular una evaluación humana que no se realizó.
