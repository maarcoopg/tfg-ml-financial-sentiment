# Diagnóstico del sentimiento de FinBERT

**Autor:** Marco Padilla Gómez. **Issue #50, en curso.** La evaluación humana
está pendiente. No se han calculado macro-F1, exactitud ni resultados bursátiles.

## Datos y reserva

Se auditan 46.223 registros noticia–empresa, con 38.704 textos distintos y 35.709
grupos conectados por URL, titular o texto normalizado. Hay 13 resúmenes vacíos,
ningún texto vacío y ninguna entrada supera las 512 posiciones de FinBERT. La
mediana es 114 tokens y el máximo 493, incluidos los especiales. No se modifica
el corpus original. Las 102 alertas lingüísticas son heurísticas, no idiomas
confirmados; falta revisión humana.

Se excluyen de la muestra 14 grupos que cruzan el corte temporal (46 filas).
Se seleccionan 400 parejas, 50 por empresa y partición. Hay 200 de desarrollo
y 200 de evaluación; una pareja por grupo. Las plantillas ocultan resultados
del modelo y proveedor. Se preparan 96 casos para doble anotación independiente.
El muestreo equilibrado no representa la distribución real de noticias.

En la muestra, 115 parejas pertenecen a grupos asociados a varias empresas.
La regla de contexto encuentra alias explícitos en 193 de las 400 parejas:
100 de desarrollo y 93 de evaluación. La falta de alias no demuestra irrelevancia,
pero impide atribuir automáticamente el tono de toda la noticia a la empresa.

## Desarrollo: sensibilidad a la entrada

| Comparación | Parejas | Coincidencia de etiqueta |
| --- | ---: | ---: |
| Titular frente a titular y resumen | 200 | 55,5 % |
| Titular y resumen frente a contexto explícito | 100 | 83,0 % |
| Titular y resumen frente a Alpha Vantage | 200 | 49,0 % |

**Estas proporciones no son exactitud.** Sin etiquetas humanas no puede saberse
qué sistema acierta. La comparación de contexto se restringe a las filas con
contexto disponible, no oculta sus abstenciones. El proveedor podría utilizar
contenido distinto al nuestro.

## Pruebas controladas

Seis pares tienen una expectativa de ordenación y uno intercambia empresas con
efectos opuestos. Cinco de los seis presentan una diferencia de tono con el signo
esperado, pero eso no implica clasificación correcta: el modelo mantiene una
etiqueta positiva al negar que se espere una subida de beneficios, al aumentar
las pérdidas y al incumplir expectativas pese a subir los beneficios. El par
de litigios invierte la ordenación esperada. Se conservan los siete casos.

Son ejemplos sintéticos prefijados, no una muestra representativa ni un benchmark
anotado. Sus resultados justifican analizar estos fenómenos en noticias reales,
no estimar una tasa general de aciertos a partir de seis contrastes.

## Atribuciones y perturbaciones

Cuatro textos predefinidos (tres sintéticos y el primer titular de desarrollo por
fecha e ID) se inspeccionan con dos referencias, PAD y MASK. Las ocho integraciones
cumplen la tolerancia de completitud prefijada; requieren entre 32 y 128 pasos.
El mayor residuo absoluto es 0,016535, tolerado por el criterio relativo al cambio
de logit. Esto verifica aproximación numérica, no verdad de la explicación.

La referencia cambia las contribuciones. Por ejemplo, en el primer caso, retirar
los tokens de mayor contribución produce una caída de logit de aproximadamente
1,074 con la selección basada en PAD y 0,451 con MASK; el control aleatorio
medio da 0,639. No todas las atribuciones superan al control aleatorio. En el
titular real, la perturbación elegida con PAD aumenta ligeramente el logit
(caída -0,052). Se presentan estos contraejemplos en lugar de concluir que el
método siempre identifica las causas de la decisión.

Los controles tienen el mismo número de tokens que la perturbación dirigida en
cada atribución. Se enmascaran subpalabras, lo que puede generar entradas
artificiales. La inspección de cuatro textos no valida globalmente la fidelidad.

## Procedencia temporal

El SHA-256 de los pesos locales coincide con el objeto LFS de la publicación de
diciembre de 2020. La revisión fijada de mayo de 2023 modifica documentación.
Esto apoya la disponibilidad del modelo antes del corte de evaluación; no permite
enumerar el corpus de entrenamiento ni descartar solapamientos en desarrollo.

## Pendiente y reproducción

Faltan etiquetas humanas, revisión de idioma, doble anotación y adjudicación.
El panel de evaluación no tiene predicciones generadas. Las copias editables
están en `data/annotations/finbert-sentiment-20260925/`; los informes permanecen
inmutables. El evaluador rechaza las plantillas vacías.

[Protocolo y comandos](../../../docs/finbert_evaluacion_protocolo.md).
[Notebook 08](../../../notebooks/08_finbert_sentiment_evaluation.ipynb).

La issue sigue abierta y la rama no se fusiona a `main`. La tarea #51 no ha
comenzado y todos los experimentos financieros anteriores se conservan intactos.
