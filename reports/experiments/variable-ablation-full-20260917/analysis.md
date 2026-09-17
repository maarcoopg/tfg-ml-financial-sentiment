# Ablación individual de las variables de sentimiento

Se han ajustado 180 modelos nuevos y verificado 36 referencias. Las 159.264
predicciones incluyen ambas referencias y todas las ablaciones, sobre 553 sesiones
por empresa (AAPL, MSFT, NVDA y TSLA), desde 2023-10-16 hasta 2025-12-29.

Se conserva la validación temporal externa en tres bloques, el historial completo,
la purga y el umbral 0,5. Cada ajuste hereda los hiperparámetros seleccionados mediante
validación interna de su referencia: sin extras al añadir, bloque completo al retirar.
No se realiza otra búsqueda. Las dos referencias pueden tener parámetros distintos.
En la variante con retardo se añade o retira conjuntamente la variable y su retardo
de una sesión. No se vuelve a ajustar la base financiera, que no cambia.

## Lectura de las diferencias

Una diferencia positiva al **añadir** sugiere utilidad de esa variable aislada.
Una diferencia positiva al **retirar** sugiere que esa variable perjudica dentro
del bloque completo. No son conclusiones causales ni necesariamente generalizables.
El AUC macro da el mismo peso a cada empresa. La tabla siguiente promedia además
los tres algoritmos únicamente como resumen descriptivo; no es un modelo combinado
ni tiene un intervalo de confianza propio.

| Variable | Añadir: híbrido | Añadir: retardo | Retirar: híbrido | Retirar: retardo |
| --- | --- | --- | --- | --- |
| Dispersión del sentimiento | 0.002037 | -0.004899 | 0.000015 | 0.002836 |
| Intensidad absoluta | 0.003609 | -0.002369 | 0.002242 | 0.001187 |
| Sorpresa del sentimiento | 0.001186 | -0.003151 | -0.000233 | 0.001029 |
| Historia disponible | 0.001750 | -0.000591 | -0.000290 | -0.000543 |
| Sorpresa del volumen de noticias | 0.006484 | -0.001424 | 0.001661 | 0.002466 |

## Qué aporta cada variable

- **Sorpresa del volumen:** es la adición con mayor mejora media en el híbrido
  (+0,006484), con diferencias positivas en los tres algoritmos. Sus tres
  intervalos incluyen cero. Con retardo empeora en los tres.
- **Intensidad absoluta:** mejora el bosque híbrido de 0,504481 a 0,520192
  (+0,015711; intervalo de diferencia [0,000408; 0,026242]). Es el único
  contraste macro nominalmente positivo, sin corregir por multiplicidad;
  el intervalo del propio AUC [0,498521; 0,540210] todavía contiene 0,5.
  No mejora todos los algoritmos ni se mantiene en la variante del bosque que
  incorpora su retardo. Por empresa, el bosque híbrido mejora en AAPL, NVDA y
  TSLA, pero empeora en MSFT: tampoco es un beneficio uniforme entre compañías.
- **Dispersión del sentimiento:** pequeño aumento medio sin retardo, pero la
  mayor caída media de las cinco adiciones con retardo (-0,004899).
- **Sorpresa del sentimiento:** cambios medios pequeños, positivos sin retardo
  y negativos con retardo. No hay un intervalo macro que excluya cero.
- **Historia disponible:** cambios pequeños y sin evidencia clara de utilidad
  general o de perjuicio. Retirarla del bloque completo empeora ligeramente
  las medias de ambas variantes.

Retirar intensidad o sorpresa del volumen aumenta modestamente la media del
bloque completo; retirar dispersión también mejora la media con retardo. Ninguno
de los 30 contrastes de retirada excluye cero. Que una variable ayude sola y
perjudique en un bloque no es una contradicción: depende de interacciones,
redundancia y del ajuste de referencia. No se elimina ninguna automáticamente.

## Incertidumbre y límites

Entre los 60 contrastes macro hay un intervalo nominal completamente positivo
y ninguno completamente negativo. Entre las 15 comparaciones primarias
(añadir al híbrido), también hay uno positivo y ninguno negativo.
Los intervalos del 95 % usan 1.000 réplicas pareadas de bloques móviles de 20 fechas,
sin reajustar modelos y sin corrección por comparaciones múltiples. No son 60 pruebas
independientes. Los 240 intervalos por empresa están disponibles en el CSV completo.

Estos periodos ya se han inspeccionado repetidamente. Los resultados sirven para
entender el comportamiento de las variables, no para certificar una mejora fuera
de muestra nueva ni rentabilidad. No se selecciona automáticamente una combinación
ganadora con estas mismas fechas. La variación entre algoritmos y empresas importa
más que escoger el máximo aislado.

## Los 60 contrastes macro

| Operación | Variable | Variante | Algoritmo | AUC referencia | AUC | Diferencia | IC inferior | IC superior |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Añadir | Dispersión del sentimiento | hybrid | Potenciación por histogramas | 0.514705 | 0.514085 | -0.000620 | -0.014170 | 0.010694 |
| Añadir | Dispersión del sentimiento | hybrid | Regresión logística | 0.497700 | 0.492333 | -0.005367 | -0.015110 | 0.008172 |
| Añadir | Dispersión del sentimiento | hybrid | Bosque aleatorio | 0.504481 | 0.516580 | 0.012099 | -0.002238 | 0.023349 |
| Añadir | Dispersión del sentimiento | lag_1 | Potenciación por histogramas | 0.517173 | 0.509274 | -0.007898 | -0.022886 | 0.002218 |
| Añadir | Dispersión del sentimiento | lag_1 | Regresión logística | 0.497372 | 0.493131 | -0.004241 | -0.013612 | 0.007762 |
| Añadir | Dispersión del sentimiento | lag_1 | Bosque aleatorio | 0.509038 | 0.506478 | -0.002560 | -0.015804 | 0.012918 |
| Añadir | Intensidad absoluta | hybrid | Potenciación por histogramas | 0.514705 | 0.509864 | -0.004841 | -0.020400 | 0.009886 |
| Añadir | Intensidad absoluta | hybrid | Regresión logística | 0.497700 | 0.497656 | -0.000044 | -0.002567 | 0.003557 |
| Añadir | Intensidad absoluta | hybrid | Bosque aleatorio | 0.504481 | 0.520192 | 0.015711 | 0.000408 | 0.026242 |
| Añadir | Intensidad absoluta | lag_1 | Potenciación por histogramas | 0.517173 | 0.515220 | -0.001953 | -0.019248 | 0.010636 |
| Añadir | Intensidad absoluta | lag_1 | Regresión logística | 0.497372 | 0.498835 | 0.001463 | -0.004257 | 0.008824 |
| Añadir | Intensidad absoluta | lag_1 | Bosque aleatorio | 0.509038 | 0.502420 | -0.006618 | -0.017854 | 0.004224 |
| Añadir | Sorpresa del sentimiento | hybrid | Potenciación por histogramas | 0.514705 | 0.511835 | -0.002870 | -0.017366 | 0.011063 |
| Añadir | Sorpresa del sentimiento | hybrid | Regresión logística | 0.497700 | 0.498236 | 0.000535 | -0.003005 | 0.005289 |
| Añadir | Sorpresa del sentimiento | hybrid | Bosque aleatorio | 0.504481 | 0.510375 | 0.005894 | -0.011294 | 0.018389 |
| Añadir | Sorpresa del sentimiento | lag_1 | Potenciación por histogramas | 0.517173 | 0.504820 | -0.012353 | -0.030002 | 0.003601 |
| Añadir | Sorpresa del sentimiento | lag_1 | Regresión logística | 0.497372 | 0.497984 | 0.000612 | -0.003636 | 0.006956 |
| Añadir | Sorpresa del sentimiento | lag_1 | Bosque aleatorio | 0.509038 | 0.511326 | 0.002288 | -0.011080 | 0.015180 |
| Añadir | Historia disponible | hybrid | Potenciación por histogramas | 0.514705 | 0.511367 | -0.003338 | -0.010048 | 0.003319 |
| Añadir | Historia disponible | hybrid | Regresión logística | 0.497700 | 0.498345 | 0.000644 | -0.000736 | 0.003310 |
| Añadir | Historia disponible | hybrid | Bosque aleatorio | 0.504481 | 0.512424 | 0.007943 | -0.007401 | 0.018345 |
| Añadir | Historia disponible | lag_1 | Potenciación por histogramas | 0.517173 | 0.516765 | -0.000408 | -0.006453 | 0.004844 |
| Añadir | Historia disponible | lag_1 | Regresión logística | 0.497372 | 0.497396 | 0.000024 | -0.000961 | 0.001844 |
| Añadir | Historia disponible | lag_1 | Bosque aleatorio | 0.509038 | 0.507648 | -0.001390 | -0.013763 | 0.011984 |
| Añadir | Sorpresa del volumen de noticias | hybrid | Potenciación por histogramas | 0.514705 | 0.519734 | 0.005028 | -0.007617 | 0.019081 |
| Añadir | Sorpresa del volumen de noticias | hybrid | Regresión logística | 0.497700 | 0.499305 | 0.001604 | -0.007781 | 0.008979 |
| Añadir | Sorpresa del volumen de noticias | hybrid | Bosque aleatorio | 0.504481 | 0.517299 | 0.012818 | -0.001484 | 0.025277 |
| Añadir | Sorpresa del volumen de noticias | lag_1 | Potenciación por histogramas | 0.517173 | 0.513718 | -0.003455 | -0.016672 | 0.005419 |
| Añadir | Sorpresa del volumen de noticias | lag_1 | Regresión logística | 0.497372 | 0.496895 | -0.000476 | -0.009870 | 0.006755 |
| Añadir | Sorpresa del volumen de noticias | lag_1 | Bosque aleatorio | 0.509038 | 0.508696 | -0.000341 | -0.013573 | 0.014433 |
| Retirar | Dispersión del sentimiento | hybrid | Potenciación por histogramas | 0.506631 | 0.513863 | 0.007232 | -0.006754 | 0.019636 |
| Retirar | Dispersión del sentimiento | hybrid | Regresión logística | 0.497144 | 0.500272 | 0.003128 | -0.006342 | 0.013443 |
| Retirar | Dispersión del sentimiento | hybrid | Bosque aleatorio | 0.510126 | 0.499811 | -0.010315 | -0.022347 | 0.002140 |
| Retirar | Dispersión del sentimiento | lag_1 | Potenciación por histogramas | 0.499149 | 0.511940 | 0.012790 | -0.001470 | 0.028507 |
| Retirar | Dispersión del sentimiento | lag_1 | Regresión logística | 0.496009 | 0.495276 | -0.000734 | -0.007557 | 0.009091 |
| Retirar | Dispersión del sentimiento | lag_1 | Bosque aleatorio | 0.506838 | 0.503289 | -0.003549 | -0.016157 | 0.008051 |
| Retirar | Intensidad absoluta | hybrid | Potenciación por histogramas | 0.506631 | 0.516490 | 0.009859 | -0.002704 | 0.025761 |
| Retirar | Intensidad absoluta | hybrid | Regresión logística | 0.497144 | 0.497030 | -0.000114 | -0.001803 | 0.003206 |
| Retirar | Intensidad absoluta | hybrid | Bosque aleatorio | 0.510126 | 0.507107 | -0.003019 | -0.012419 | 0.008653 |
| Retirar | Intensidad absoluta | lag_1 | Potenciación por histogramas | 0.499149 | 0.504669 | 0.005519 | -0.006564 | 0.019400 |
| Retirar | Intensidad absoluta | lag_1 | Regresión logística | 0.496009 | 0.493620 | -0.002389 | -0.007371 | 0.004041 |
| Retirar | Intensidad absoluta | lag_1 | Bosque aleatorio | 0.506838 | 0.507268 | 0.000430 | -0.011330 | 0.010578 |
| Retirar | Sorpresa del sentimiento | hybrid | Potenciación por histogramas | 0.506631 | 0.512643 | 0.006013 | -0.007871 | 0.018791 |
| Retirar | Sorpresa del sentimiento | hybrid | Regresión logística | 0.497144 | 0.495698 | -0.001447 | -0.004364 | 0.003159 |
| Retirar | Sorpresa del sentimiento | hybrid | Bosque aleatorio | 0.510126 | 0.504861 | -0.005265 | -0.012766 | 0.005267 |
| Retirar | Sorpresa del sentimiento | lag_1 | Potenciación por histogramas | 0.499149 | 0.506473 | 0.007323 | -0.005371 | 0.018171 |
| Retirar | Sorpresa del sentimiento | lag_1 | Regresión logística | 0.496009 | 0.492041 | -0.003968 | -0.008499 | 0.003590 |
| Retirar | Sorpresa del sentimiento | lag_1 | Bosque aleatorio | 0.506838 | 0.506570 | -0.000268 | -0.012315 | 0.010606 |
| Retirar | Historia disponible | hybrid | Potenciación por histogramas | 0.506631 | 0.509655 | 0.003024 | -0.000092 | 0.008222 |
| Retirar | Historia disponible | hybrid | Regresión logística | 0.497144 | 0.497373 | 0.000229 | -0.000812 | 0.002594 |
| Retirar | Historia disponible | hybrid | Bosque aleatorio | 0.510126 | 0.506004 | -0.004122 | -0.013689 | 0.007255 |
| Retirar | Historia disponible | lag_1 | Potenciación por histogramas | 0.499149 | 0.500565 | 0.001416 | -0.000356 | 0.003376 |
| Retirar | Historia disponible | lag_1 | Regresión logística | 0.496009 | 0.494741 | -0.001269 | -0.002727 | 0.001276 |
| Retirar | Historia disponible | lag_1 | Bosque aleatorio | 0.506838 | 0.505062 | -0.001776 | -0.012655 | 0.009550 |
| Retirar | Sorpresa del volumen de noticias | hybrid | Potenciación por histogramas | 0.506631 | 0.515398 | 0.008767 | -0.004979 | 0.021739 |
| Retirar | Sorpresa del volumen de noticias | hybrid | Regresión logística | 0.497144 | 0.498328 | 0.001184 | -0.004419 | 0.004755 |
| Retirar | Sorpresa del volumen de noticias | hybrid | Bosque aleatorio | 0.510126 | 0.505159 | -0.004967 | -0.014567 | 0.007008 |
| Retirar | Sorpresa del volumen de noticias | lag_1 | Potenciación por histogramas | 0.499149 | 0.505300 | 0.006150 | -0.008154 | 0.022046 |
| Retirar | Sorpresa del volumen de noticias | lag_1 | Regresión logística | 0.496009 | 0.496924 | 0.000915 | -0.004360 | 0.006917 |
| Retirar | Sorpresa del volumen de noticias | lag_1 | Bosque aleatorio | 0.506838 | 0.507171 | 0.000333 | -0.008253 | 0.010895 |

## Artefactos y reproducción

Verificación: 58 pruebas automatizadas superadas; los 180 modelos guardados
reproducen sus probabilidades; las 36 referencias, parámetros heredados,
columnas, purga, 159.264 filas y AUC recalculados han sido comprobados.
El notebook tiene 14 celdas, se ha ejecutado desde su directorio y desde la
raíz, y sus dos figuras, tildes y enlaces se han revisado.

- [Protocolo previo](../../../docs/variable_ablation_protocol.md).
- [Notebook ejecutado](../../../notebooks/06_sentiment_variable_ablation.ipynb).
- [Métricas por empresa](metrics/company_metrics.csv), [por bloque](metrics/all_metrics.csv)
  e [intervalos pareados](metrics/paired_intervals.csv).
- [Parámetros heredados](tuning/inherited_params.csv) y [manifiesto](manifest.json).

Modelos locales bajo `models/experiments/variable-ablation-full-20260917/`, excluidos
de Git. Entradas verificadas contra la primera ronda, que permanece intacta.
