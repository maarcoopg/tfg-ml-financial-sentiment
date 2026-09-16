# Modelos conjuntos con identidad de empresa y variables relativas

**Autor:** Marco Padilla Gómez

**Fecha:** 16 de septiembre de 2026

**Issue:** [#47](https://github.com/maarcoopg/tfg-ml-financial-sentiment/issues/47)

**Rama:** `feature/per-company-temporal-evaluation`

**Ejecución:** `ticker-aware-full-20260916`

## Conclusión

Hay una mejora descriptiva pequeña al utilizar **variables relativas**, pero no una mejora global demostrada con los intervalos calculados. Añadir solo el identificador de empresa apenas cambia el promedio. La mejora más destacada aparece en **NVDA** y está relacionada principalmente con las variables relativas, no con el identificador.

Este experimento nuevo se conserva separado de `per-company-20260916`. No sobrescribe su notebook, sus informes ni `reports/final/`, y no se fusiona a `main`.

## Diseño y control

Se mantienen AAPL, MSFT, NVDA y TSLA, las mismas 553 sesiones externas por empresa, tres bloques externos y tres divisiones internas crecientes con purga. El objetivo, umbral 0,5 y rejillas de dos configuraciones por algoritmo son los mismos. La selección interna sigue utilizando el AUC conjunto; la comparación principal externa utiliza el AUC macro por empresa.

Se comparan cinco algoritmos para: identificador con variables originales, variables relativas sin identificador y variables relativas con identificador. Para regresión logística se añaden dos variantes con interacciones explícitas empresa-variable. Los árboles ya pueden modelar interacciones mediante sus divisiones.

Los tres indicadores binarios representan MSFT, NVDA y TSLA, con AAPL como referencia. En la regresión logística, los indicadores cambian el nivel de partida, mientras que las interacciones permiten pendientes distintas. Esta parametrización y su regularización conjunta no equivalen a un modelo jerárquico bayesiano ni garantizan una especialización óptima.

La base relativa tiene 9 variables financieras; el híbrido relativo, 22; y el relativo con retardo, 24. Se reutiliza la transformación relativa del proyecto: distancias a medias móviles, MACD dividido por precio y volúmenes frente a sus medias previas. No se introducen noticias en la base. Las medias y retardos son por empresa, y los volúmenes relativos usan historia anterior a la sesión actual. Las noticias siguen conservando el sentimiento específico de cada activo.

Las referencias conjunta e individual se reutilizan tras verificar entradas, panel reconstruido byte a byte, código principal, versiones, rejillas, umbral, predicciones y fronteras temporales. No se han descargado datos nuevos. Hay 168 modelos nuevos guardados y 918 evaluaciones internas, además de las referencias anteriores.

## Resumen comparable

AUC macro: se calcula el AUC dentro de cada empresa y después se promedian las cuatro con igual peso. La tabla siguiente promedia además los cinco algoritmos, únicamente como resumen descriptivo. **No es el rendimiento de un modelo adicional ni debe compararse directamente con un máximo de una sola empresa.**

| Variante | Conjunto original | Individual | Conjunto + empresa | Relativas | Relativas + empresa |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0.4947 | 0.4930 | 0.4947 | 0.5031 | 0.5010 |
| Híbrido | 0.4986 | 0.4964 | 0.4984 | 0.5064 | 0.5069 |
| Con retardo | 0.4986 | 0.4969 | 0.4989 | 0.5058 | 0.5037 |

Las relativas sin identificador mejoran el AUC macro observado en las 15 combinaciones algoritmo-variante respecto al conjunto original. Relativas con identificador mejoran 12 de 15. Añadir identificador a las relativas solo mejora 6 de 15; por tanto, **no atribuiría la mejora principalmente a saber qué empresa es**.

Ninguno de los 81 intervalos macro de los contrastes definidos excluye cero. Son intervalos percentiles del 95 % con 1.000 réplicas de bloques de 20 sesiones, compartidos entre empresas y enfoques. No se corrige por comparaciones múltiples ni se vuelve a entrenar dentro de cada réplica. La ausencia de una diferencia clara no prueba equivalencia.

## Todos los algoritmos

| Variante | Algoritmo | Conjunto original | Individual | Conjunto + empresa | Relativas | Relativas + empresa |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base | Regresión logística | 0.5015 | 0.4839 | 0.4981 | 0.5019 | 0.5010 |
| Base | Bosque aleatorio | 0.5057 | 0.4884 | 0.4956 | 0.5068 | 0.4990 |
| Base | Potenciación por histogramas | 0.4846 | 0.4965 | 0.4917 | 0.5030 | 0.5011 |
| Base | XGBoost | 0.4918 | 0.4992 | 0.4948 | 0.5054 | 0.5008 |
| Base | LightGBM | 0.4899 | 0.4972 | 0.4935 | 0.4983 | 0.5029 |
| Híbrido | Regresión logística | 0.4987 | 0.4934 | 0.4964 | 0.5000 | 0.4980 |
| Híbrido | Bosque aleatorio | 0.5054 | 0.4944 | 0.5074 | 0.5112 | 0.5139 |
| Híbrido | Potenciación por histogramas | 0.4961 | 0.5000 | 0.4937 | 0.5127 | 0.5029 |
| Híbrido | XGBoost | 0.4988 | 0.5016 | 0.4980 | 0.5079 | 0.5095 |
| Híbrido | LightGBM | 0.4938 | 0.4929 | 0.4967 | 0.5000 | 0.5103 |
| Con retardo | Regresión logística | 0.4961 | 0.4968 | 0.4942 | 0.4998 | 0.4981 |
| Con retardo | Bosque aleatorio | 0.5091 | 0.4911 | 0.5046 | 0.5099 | 0.5157 |
| Con retardo | Potenciación por histogramas | 0.4934 | 0.4960 | 0.4935 | 0.5024 | 0.4993 |
| Con retardo | XGBoost | 0.4996 | 0.4978 | 0.5061 | 0.5049 | 0.5082 |
| Con retardo | LightGBM | 0.4950 | 0.5028 | 0.4963 | 0.5121 | 0.4972 |

El mayor AUC macro nuevo es **0,5157**, con bosque aleatorio, variables relativas, identificador y retardo. Su intervalo es [0,4863; 0,5386]. Frente al mismo bosque aleatorio conjunto original, la diferencia es +0,0066, con intervalo [−0,0163; 0,0294]. Es un máximo identificado después de observar la prueba, no una selección validada.

Con umbral 0,5, esa combinación tiene exactitud macro 52,35 %, exactitud equilibrada 50,13 % y MCC medio 0,0019. La referencia mayoritaria obtiene 53,89 % de exactitud. Por tanto, el pequeño aumento de AUC no representa una ventaja global clara en aciertos.

## Diferencias entre empresas

Media de AUC de los cinco algoritmos, comparando el conjunto original con relativas e identificador:

| Empresa | Base original | Base nueva | Híbrido original | Híbrido nuevo | Retardo original | Retardo nuevo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AAPL | 0.4747 | 0.4563 | 0.4799 | 0.4635 | 0.4840 | 0.4664 |
| MSFT | 0.4965 | 0.4940 | 0.4934 | 0.5075 | 0.4870 | 0.5029 |
| NVDA | 0.4923 | 0.5440 | 0.5013 | 0.5481 | 0.5136 | 0.5346 |
| TSLA | 0.5153 | 0.5095 | 0.5197 | 0.5087 | 0.5099 | 0.5108 |

AAPL empeora en la media de las tres variantes. MSFT presenta cambios pequeños y mixtos. NVDA mejora claramente en términos descriptivos. TSLA no muestra un beneficio uniforme. En la comparación principal hay ocho intervalos por empresa totalmente positivos, todos en NVDA, y uno negativo en AAPL; los demás incluyen cero. Estas comparaciones son numerosas y correlacionadas: ocho resultados favorables no son ocho confirmaciones independientes.

### Caso de NVDA

Comparación del **mismo bosque aleatorio híbrido** en las mismas fechas:

| Enfoque | AUC en NVDA |
| --- | ---: |
| Conjunto original | 0.5180 |
| Individual | 0.5026 |
| Conjunto + empresa | 0.5132 |
| Relativas | 0.5711 |
| Relativas + empresa | 0.5717 |

El modelo con relativas e identificador alcanza **0,5717 de AUC**, con intervalo [0,5314; 0,6158]. La diferencia frente al conjunto original es +0,0537, con intervalo [0,0147; 0,0980]; frente al modelo individual del mismo algoritmo, +0,0692, con intervalo [0,0181; 0,1262].

Sin embargo, las relativas **sin identificador** ya alcanzan 0,5711. El incremento adicional por incluir la empresa es solo +0,00065 y su intervalo [−0,0307; 0,0322] incluye cero. El resultado apunta a la representación relativa como cambio más relevante, sin demostrar un mecanismo causal único.

El AUC de la versión con relativas e identificador es 0,5783, 0,5429 y 0,5936 en los tres bloques, respectivamente. La mejora no se limita a un único bloque. Aun así, el caso se destaca después de examinar muchas combinaciones sobre un histórico ya conocido; requiere confirmación en fechas nuevas.

Su **exactitud es 55,88 %**, frente a 54,97 % de predecir la clase mayoritaria: cinco aciertos adicionales sobre 553 sesiones. La exactitud equilibrada es 54,75 % y MCC 0,0972. Un AUC de 0,5717 no significa acertar el 57,17 %.

## Interacciones en regresión logística

Esta tabla compara únicamente regresión logística; no mezcla una media de un algoritmo con la de cinco.

| Variante | Identificador original | Interacciones originales | Identificador relativo | Interacciones relativas |
| --- | ---: | ---: | ---: | ---: |
| Base | 0.4981 | 0.4887 | 0.5010 | 0.4978 |
| Híbrido | 0.4964 | 0.4978 | 0.4980 | 0.5059 |
| Con retardo | 0.4942 | 0.4987 | 0.4981 | 0.5076 |

Las interacciones relativas mejoran descriptivamente híbrido y retardo, pero no la base. Ninguno de los seis intervalos macro de interacciones frente a identificador excluye cero. Añadir pendientes por empresa no garantiza una mejora: aumenta el número de parámetros y mantiene la misma rejilla limitada de regularización.

## Qué haría con estos resultados

Conservaría ambos experimentos y destacaría en la memoria que **cambiar la representación de los datos aporta más que añadir únicamente el identificador** en esta comparación. Mantendría las variables relativas como hipótesis prioritaria para una validación futura, sin declarar vencedor definitivo ni sustituir automáticamente el enfoque anterior.

No ampliaría la búsqueda sobre las mismas fechas solo para elevar las métricas. Antes de elegir una combinación, fijaría una regla de selección y la probaría en fechas nuevas. Tampoco extrapolaría el caso de NVDA al resto de empresas o a rentabilidad: aquí no hay simulación económica ni costes de negociación.

## Verificación y reproducción

- 41 pruebas automatizadas superadas: identidad, interacciones, aislamiento por empresa, causalidad de medias y retardos, columnas booleanas, escalas, referencias incompatibles y cálculo de incertidumbre.
- El cálculo optimizado reproduce los 60 intervalos del experimento anterior con una diferencia máxima de 2,6 × 10⁻¹⁶.
- 194.656 predicciones identificadas por enfoque; 405 intervalos pareados, de los cuales 81 son macro y 324 por empresa.
- Los 168 modelos guardados reproducen las probabilidades registradas con tolerancia absoluta de 10⁻¹². Para recargar `features.csv`, se utiliza `pd.read_csv(..., float_precision="round_trip")`: el redondeo del lector predeterminado puede cambiar una decisión de un árbol cuando un valor está junto a un umbral.
- Los informes conservan sus bytes mediante `.gitattributes`. Modelos, datos de ejecución y copias de código permanecen fuera de Git.
- La primera ejecución técnica se detuvo antes del entrenamiento por una conversión de tipos. Se corrigió y añadió una prueba de regresión; ese intento se conserva localmente como diagnóstico y no se mezcla con esta ejecución completa.

```powershell
python -m src.experiments.run_ticker_aware --run-dir reports/experiments/ticker-aware-NUEVA-EJECUCION
python -m unittest discover -s tests -v
```

El identificador debe ser nuevo. Si cambia el código principal o la referencia, el ejecutor se detiene en lugar de reutilizar resultados incompatibles.

El [notebook 04](../../../notebooks/04_ticker_aware_models.ipynb) contiene tablas, figuras, métricas por bloque y empresa, hiperparámetros y verificaciones. El [experimento individual anterior](../per-company-20260916/analysis.md) permanece intacto.
