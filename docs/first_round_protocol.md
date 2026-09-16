# Primera ronda controlada de mejoras

Issue [#48](https://github.com/maarcoopg/tfg-ml-financial-sentiment/issues/48).
Rama: `feature/controlled-improvement-round`.

## Pregunta y límites

¿Cambian los resultados al controlar repeticiones, limitar historia, enriquecer sentimiento y ajustar la selección al AUC macro? Se conserva el objetivo diario, cuatro empresas, umbral 0,5 y las fechas externas de los experimentos anteriores. Esta ronda no es una prueba independiente: el histórico ya se ha examinado. No se descargan noticias ni se consulta un nuevo periodo reservado.

Se utilizan regresión logística, bosque aleatorio y potenciación por histogramas. Es un subconjunto prefijado para limitar coste, no una selección posterior del ganador. Los promedios de esta ronda abarcan tres algoritmos, no los cinco anteriores. Se comparan siempre base, híbrido y retardo de una sesión.

## Comparaciones prefijadas

| Etapa | Noticias y variables | Historia | Selección | Búsqueda |
| --- | --- | --- | --- | --- |
| `reference` | Originales, relativas sin identidad | Todo el pasado | AUC agrupado | Dos configuraciones |
| `deduplicated` | Títulos normalizados deduplicados, mismas relativas | Todo el pasado | AUC agrupado | Dos |
| `window_3y` | Igual a deduplicadas | Tres años | AUC agrupado | Dos |
| `window_5y` | Igual a deduplicadas | Cinco años | AUC agrupado | Dos |
| `enhanced` | Deduplicadas y nuevas variables de sentimiento | Todo el pasado | AUC agrupado | Dos |
| `macro` | Igual a enriquecidas | Todo el pasado | AUC macro por empresa | Dos |
| `tuned` | Igual a enriquecidas | Elegida internamente entre todo, tres y cinco años | AUC macro | 20 candidatos por algoritmo de árboles; control logístico reducido |

La comparación principal es `tuned` menos `reference` para el híbrido de cada algoritmo. Los contrastes restantes describen cambios aislados: deduplicación frente a referencia; cada ventana frente a deduplicadas; enriquecidas frente a deduplicadas; macro frente a enriquecidas; ajuste ampliado frente a macro. La etapa final combina decisiones y no permite atribuir su diferencia a una sola causa.

## Controles

- Tres bloques externos y tres particiones internas por fechas completas, con purga de `target_end`. Las fronteras de validación son comunes a todos los candidatos; las ventanas limitan únicamente las filas de entrenamiento de cada ajuste.
- Los tres y cinco años son ventanas de calendario retrospectivas respecto al inicio del bloque evaluado, no respecto al final del conjunto de datos.
- La deduplicación es causal y conservadora: primer título idéntico tras normalización Unicode, mayúsculas y espacios, por empresa en 24 horas. No se eliminan números, signos o negaciones. No equivale a deduplicación semántica perfecta; las entradas originales y el registro de eliminaciones se conservan.
- Referencias históricas de las nuevas variables calculadas por empresa con desplazamiento de una sesión. Se pueden usar noticias actuales disponibles al cierre, nunca posteriores. Sin noticias en las variables base.
- Los candidatos ampliados y sus semillas se fijan antes de consultar resultados externos. La ventana se elige dentro de la validación interna, nunca por el resultado externo.
- AUC macro y por empresa, exactitud, exactitud equilibrada, F1 macro y MCC. Intervalos pareados de 1.000 réplicas de bloques de veinte sesiones compartidos entre empresas. Sin corrección por comparaciones múltiples y sin reentrenamiento dentro del remuestreo.
- No se modifica el objetivo, el instante de predicción ni la selección de empresas. Modelos y datos pesados permanecen fuera de Git; informes nuevos con identificador único y huellas verificables.

## Entrega incremental

La búsqueda ampliada conserva las dos configuraciones originales con toda la historia y añade dieciocho candidatos aleatorios únicos, con semilla 42, repartidos entre las tres ventanas. No son veinte candidatos por ventana: son veinte en total por algoritmo de árboles. La regresión logística compara sus dos valores de C con las tres ventanas (seis candidatos). Las configuraciones completas se guardan en el manifiesto antes del entrenamiento.

| Algoritmo | Espacio adicional |
| --- | --- |
| Bosque aleatorio | 100/200 árboles; profundidad 3/4/6/8; mínimo por hoja 5/10/20/40; variables por división `sqrt`/0,5/1,0 |
| Potenciación por histogramas | 100/200 iteraciones; 3/7/15/31 hojas; tasa 0,025/0,05/0,1; mínimo por hoja 10/20/40/80; regularización L2 0/1/10; sin parada temprana |
| Regresión logística | C 0,1/10; misma representación y preprocesamiento, distinta ventana |

Las fronteras internas parten del pasado completo del bloque externo y son comunes a todas las ventanas. Si en un ajuste temprano hay menos de tres o cinco años disponibles, la ventana utiliza solo esa historia existente; no se fabrican observaciones ni se trasladan fronteras. La referencia mayoritaria usa todo el pasado en las siete etapas para conservar una referencia común.

La representación enriquecida añade cinco variables: desviación típica del tono dentro de la sesión, media de su valor absoluto, diferencia del tono actual respecto a su media de las veinte sesiones anteriores con noticias (mínimo cinco observaciones), indicador de historia suficiente y sorpresa del logaritmo del número de noticias respecto a las veinte sesiones anteriores. Esta última se estandariza con desviación histórica y se limita a [−5; 5]; si no hay variación histórica, se utiliza cero. La media de tono usa una ventana de veinte sesiones bursátiles, no las últimas veinte noticias. El híbrido tiene 27 variables y el de retardo 34, al añadir los cinco retardos nuevos a las variables actuales. La base relativa permanece en nueve.

Commits separados para auditoría, ventanas, representación, selección y resultados. Antes de cada commit se comunica el contenido, su verificación y el beneficio esperado u observado. Los resultados desfavorables también se conservan. La issue permanece abierta y la rama no se fusiona hasta revisar la ejecución con el autor.
