# Primera ronda controlada de mejoras

**Autor:** Marco Padilla Gómez

**Revisión:** 17 de septiembre de 2026

**Issue:** [#48](https://github.com/maarcoopg/tfg-ml-financial-sentiment/issues/48)

**Rama:** `feature/controlled-improvement-round`

**Ejecución:** `first-round-full-20260916`

## Conclusión principal

La ronda **no consigue una mejora general de discriminación**. La combinación completa de deduplicación, representación enriquecida, selección macro y búsqueda ampliada reduce el AUC macro medio de los tres algoritmos: de 0,5039 a 0,4995 en base, de 0,5080 a 0,5015 en híbrido y de 0,5040 a 0,4996 con retardo. Los tres contrastes híbridos principales presentan diferencias negativas, aunque sus intervalos incluyen cero: no se demuestra ni mejora ni empeoramiento concluyente con ese procedimiento.

La deduplicación conserva un resultado puntual favorable en potenciación por histogramas con retardo: AUC macro 0,5172 frente a 0,5024. Es el único intervalo macro de diferencia nominalmente positivo entre 63 contrastes y su límite inferior está muy cerca de cero. No acredita un ganador robusto, rentabilidad ni capacidad predictiva absoluta claramente superior a 0,5.

Se conservan todos los resultados y no se reemplazan los experimentos anteriores. La rama permanece pendiente de revisión del autor antes de fusionar con `main`.

## Qué se ha probado y por qué

La referencia de esta ronda es **relativas sin identidad de empresa**, no el modelo original con precios absolutos. Se eligió para investigar la representación que resultó más prometedora en el experimento anterior sin añadir identidad innecesariamente.

Se mantienen AAPL, MSFT, NVDA y TSLA, objetivo de la siguiente sesión, umbral 0,5, 553 sesiones externas por empresa entre el 16/10/2023 y el 29/12/2025, tres bloques externos y tres particiones internas con purga. No se han descargado datos ni consumido llamadas de pago. No se ha examinado un periodo nuevo reservado.

Se comparan tres algoritmos prefijados: regresión logística, bosque aleatorio y potenciación por histogramas. **Las medias de tres algoritmos no deben compararse directamente con medias de cinco algoritmos de informes anteriores.** El AUC macro promedia el AUC calculado dentro de cada una de las cuatro empresas, no las probabilidades de distintas empresas. Promediarlo después entre algoritmos es solo un resumen descriptivo, no un modelo adicional.

| Etapa | Propósito y control |
| --- | --- |
| Referencia | Reproducir relativas sin identidad, dos configuraciones y AUC agrupado interno |
| Deduplicación | Reducir sobreponderación de títulos idénticos, manteniendo todo lo demás |
| Ventana de tres años | Evaluar adaptación con menos historia; comparar contra deduplicación |
| Ventana de cinco años | Evaluar una reducción intermedia de historia |
| Sentimiento enriquecido | Añadir intensidad, desacuerdo y sorpresa; comparar contra deduplicación con todo el pasado |
| Selección macro | Alinear el criterio interno con AUC medio por empresa, sin cambiar candidatos |
| Ajuste ampliado | Elegir parámetros y ventana internamente con un presupuesto mayor y prefijado |

El [protocolo](../../../docs/first_round_protocol.md) se publicó antes de la ejecución completa. La referencia reentrenada reproduce sus 22.120 probabilidades anteriores con tolerancia absoluta de 10⁻¹².

## Auditoría de noticias

El corpus alineado contiene 46.014 registros. La política conservadora retiene 42.734 y elimina 3.280, aproximadamente un 7,13 %. Conserva la primera publicación de un título idéntico tras normalizar Unicode, mayúsculas y espacios, para la misma empresa en 24 horas. Mantiene números, signos y negaciones y nunca utiliza noticias futuras para cambiar una decisión anterior.

| Empresa | Registros eliminados |
| --- | ---: |
| AAPL | 274 |
| MSFT | 755 |
| NVDA | 2.017 |
| TSLA | 234 |

3.103 eliminaciones corresponden a 2025. Esto es compatible con una mayor repetición en el periodo de mayor volumen, pero no explica todo el salto de cobertura. Las tablas `quality/monthly_coverage.csv` y `quality/monthly_sources.csv` conservan recuentos y distribución de fuentes antes y después. Sus ratios se restringen a sesiones del panel; el recuento general anterior incluye todo el corpus alineado.

La deduplicación no es una anotación semántica: dos títulos distintos pueden repetir el mismo acontecimiento y un título idéntico puede acompañar una actualización válida. Las entradas originales y el registro de correspondencias eliminado-conservado permanecen disponibles localmente. No se certifica exhaustividad de noticias ni se seleccionan fuentes por su rendimiento externo.

## Resultados de todas las etapas

Media de AUC macro entre los tres algoritmos:

| Variante | Referencia | Deduplicación | Tres años | Cinco años | Enriquecido | Selección macro | Ajuste ampliado |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 0,5039 | 0,5039 | 0,4970 | 0,5023 | 0,5039 | 0,5039 | 0,4995 |
| Híbrido | 0,5080 | 0,5056 | 0,4951 | 0,5039 | 0,5046 | 0,5046 | 0,5015 |
| Con retardo | 0,5040 | 0,5079 | 0,4979 | 0,5070 | 0,5007 | 0,5007 | 0,4996 |

La base financiera no cambia con la deduplicación o las variables adicionales de noticias. Se ha verificado la igualdad de sus probabilidades, no solo de los AUC redondeados. Las ventanas, el criterio de selección o los candidatos sí pueden modificar el ajuste del modelo base.

### Qué aporta cada cambio

- **Deduplicación:** mejora dos de las seis combinaciones con noticias, ambas de potenciación por histogramas; las otras cuatro empeoran. La media híbrida baja y la de retardo sube. No basta para sustituir el corpus operativo automáticamente.
- **Tres años:** las nueve combinaciones algoritmo-variante empeoran frente a deduplicación con todo el pasado. No hay evidencia de que recortar tanto la historia ayude en este diseño.
- **Cinco años:** cuatro de nueve combinaciones mejoran descriptivamente, pero ninguna diferencia macro tiene intervalo completamente positivo. Tampoco supera uniformemente la historia completa.
- **Representación enriquecida:** solo mejora una de las seis combinaciones con noticias, el bosque aleatorio híbrido. La caída es mayor en potenciación por histogramas con retardo, de 0,5172 a 0,4991.
- **Selección macro:** selecciona exactamente los mismos parámetros que el criterio agrupado en las 27 combinaciones externas predictivas. Las predicciones y métricas no cambian. La alineación metodológica no produce aquí una mejora numérica con esos dos candidatos.
- **Búsqueda ampliada:** eleva la puntuación interna en 23 de 27 ajustes predictivos frente a la rejilla pequeña con selección macro, pero solo mejora dos de las nueve combinaciones externas agregadas frente a esa etapa. Más puntuación interna no se traduce en mayor generalización.

### Todos los algoritmos: referencia frente a ajuste completo

| Variante | Algoritmo | Referencia | Ajuste ampliado |
| --- | --- | ---: | ---: |
| Base | Regresión logística | 0,5019 | 0,4950 |
| Base | Bosque aleatorio | 0,5068 | 0,5025 |
| Base | Potenciación por histogramas | 0,5030 | 0,5008 |
| Híbrido | Regresión logística | 0,5000 | 0,4897 |
| Híbrido | Bosque aleatorio | 0,5112 | 0,5069 |
| Híbrido | Potenciación por histogramas | 0,5127 | 0,5081 |
| Con retardo | Regresión logística | 0,4998 | 0,4894 |
| Con retardo | Bosque aleatorio | 0,5099 | 0,5061 |
| Con retardo | Potenciación por histogramas | 0,5024 | 0,5034 |

## Comparación principal e incertidumbre

La comparación principal prefijada es ajuste ampliado menos referencia para el híbrido, manteniendo algoritmo:

| Algoritmo | Diferencia AUC macro | Intervalo del 95 % |
| --- | ---: | --- |
| Regresión logística | −0,01034 | [−0,03237; 0,01532] |
| Bosque aleatorio | −0,00438 | [−0,02430; 0,01321] |
| Potenciación por histogramas | −0,00467 | [−0,02921; 0,02094] |

Se calcularon 315 intervalos pareados: 63 macro y 252 por empresa. Utilizan 1.000 réplicas de bloques de veinte sesiones, compartidos entre empresas y enfoques. No vuelven a entrenar ni corrigen por multiplicidad. Ninguno de los tres contrastes principales indica mejora.

El único intervalo macro completamente positivo pertenece al contraste secundario de **deduplicación con potenciación por histogramas y retardo**: diferencia +0,014758, intervalo [0,000080; 0,029403]. Su AUC es 0,517173 y el intervalo de AUC [0,495506; 0,546076] contiene 0,5. El límite inferior de la diferencia es casi cero y hay 63 contrastes macro sin ajuste: el resultado es débil y exploratorio. Los otros 62 intervalos macro contienen cero.

## Diferencias entre empresas y aciertos

Media de AUC híbrido entre tres algoritmos:

| Empresa | Referencia | Deduplicación | Enriquecido | Ajuste ampliado |
| --- | ---: | ---: | ---: | ---: |
| AAPL | 0,4738 | 0,4607 | 0,4771 | 0,4768 |
| MSFT | 0,5072 | 0,5106 | 0,5002 | 0,5051 |
| NVDA | 0,5462 | 0,5354 | 0,5249 | 0,5160 |
| TSLA | 0,5048 | 0,5158 | 0,5164 | 0,5081 |

No hay un efecto uniforme. En particular, NVDA pierde AUC medio en esta secuencia, aunque había mostrado el resultado más atractivo en la ronda anterior. Eso limita cualquier extrapolación a partir de un único activo favorable.

| Híbrido con ajuste ampliado | Exactitud macro | Exactitud equilibrada | MCC medio |
| --- | ---: | ---: | ---: |
| Regresión logística | 50,09 % | 49,24 % | −0,0152 |
| Bosque aleatorio | 51,31 % | 49,37 % | −0,0158 |
| Potenciación por histogramas | 51,22 % | 50,67 % | 0,0131 |
| Referencia mayoritaria común | 53,89 % | 50,00 % | 0,0000 |

El mejor AUC macro descriptivo de toda la ronda, 0,5172, tiene exactitud 52,49 %, exactitud equilibrada 50,74 % y MCC medio 0,0150. No supera la exactitud de la referencia mayoritaria. Un AUC mayor no equivale a un porcentaje de aciertos del mismo valor ni demuestra rentabilidad.

## Parámetros y ventanas

Los árboles prueban veinte candidatos totales cada uno: las dos configuraciones previas y dieciocho adicionales con semilla 42, repartidos entre ventanas. Regresión logística compara sus dos valores de C con tres ventanas, seis candidatos. Los espacios completos están en el protocolo y en `manifest.json`; las decisiones, en `tuning/selected_params.csv`.

En la etapa ampliada se eligen tres años en 13 ajustes, cinco años en siete y todo el pasado en siete. Esto describe lo que prefirió la validación interna; no demuestra que las ventanas cortas sean mejores externamente. El criterio macro y la expansión de candidatos se fijaron antes de consultar esta evaluación completa.

## Decisión e interpretación

Conservaría la auditoría y los controles implementados como mejoras de ingeniería. Mantendría la deduplicación y las nuevas variables como variantes experimentales, no como reemplazos automáticos del conjunto anterior. No adoptaría el ajuste completo por sus resultados actuales ni ampliaría otra vez la búsqueda sobre las mismas fechas para perseguir un máximo.

La combinación de más puntuación interna y menor AUC externo es compatible con ruido de selección, señal débil o cambios de distribución; no identifica una causa única ni demuestra un fallo del programa. Tampoco estas comparaciones prueban que las noticias no influyan en los mercados. Evalúan esta representación, horizonte, universo y búsqueda limitada.

Una siguiente decisión razonable sería validar hipótesis fijadas en fechas nuevas o investigar otra fuente de señal con un protocolo independiente. Cambiar horizonte o añadir contexto de mercado supondría un experimento distinto, no una reparación retrospectiva de esta ronda.

## Verificación y reproducción

- 56 pruebas automatizadas superadas.
- 210 modelos guardados reproducen sus probabilidades con tolerancia absoluta de 10⁻¹², recargando variables con `float_precision="round_trip"`.
- 2.214 ajustes internos verificados: fronteras purgadas, ventanas propias de cada partición y selección del mejor candidato según el criterio registrado.
- 154.840 predicciones pareadas sobre las mismas claves, etiquetas y bloques; no son observaciones independientes adicionales.
- Referencia anterior reproducida; base financiera inalterada por cambios exclusivos de noticias; intervalos contrastados con la implementación anterior.
- Huellas de entradas y salidas verificadas; modelos, datos e instantáneas de código locales excluidos de Git.
- Corpus original, experimentos anteriores y `reports/final/` sin modificar.

```powershell
python -m unittest discover -s tests -v
python -m src.experiments.run_first_round --run-dir reports/experiments/first-round-NUEVA-EJECUCION
```

El identificador debe ser nuevo y las entradas y versiones deben coincidir con la referencia. El [cuaderno 05](../../../notebooks/05_controlled_improvement_round.ipynb) contiene las tablas completas, auditoría, figuras, candidatos y verificaciones sin volver a entrenar.
