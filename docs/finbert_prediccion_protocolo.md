# Utilidad predictiva del sentimiento de FinBERT

**Issue #51. Protocolo fijado antes de obtener resultados bursátiles nuevos.**
La #50 se cerró con autorización del autor, utilizando una referencia de IA y
declarando que no se realizó validación humana. No se usa esa referencia como
etiqueta de entrenamiento ni como verdad lingüística en esta fase.

## Pregunta y panel

¿Cambiar el sentimiento del proveedor por FinBERT aporta información para
clasificar la dirección de la siguiente sesión? Se conservan AAPL, MSFT, NVDA y
TSLA, sin SPY. Se mantienen el objetivo, cierre NASDAQ, ajuste UTC y purga de
etiquetas del flujo existente. No se modifica ningún archivo original.

Se calculan variables con historia previa para calentamiento, pero solo se
entrenan filas desde el 01/01/2021: fecha posterior a la publicación de los pesos
en diciembre de 2020. El panel externo mantiene el comienzo 16/10/2023 y tres
bloques consecutivos hasta el final disponible de 2025. Esta restricción de
entrenamiento se aplica a todos los enfoques, incluida la base. Por ello no se
atribuyen diferencias respecto a tablas antiguas exclusivamente a FinBERT.

La separación temporal verifica disponibilidad respecto a la marca registrada,
no certifica cuándo existía el resumen en su versión actual. Se excluyen del
panel de noticias emparejadas fichas dinámicas reconocibles por titular o ruta
de URL, y se exige una mención explícita de la empresa mediante los alias ya
definidos. Estas reglas son conservadoras e imperfectas, no un detector de
relevancia validado. No se utiliza una etiqueta del asistente ni un retorno para
aceptar una noticia. No se elimina una sesión bursátil por falta de noticias.

## Modelo e inferencia

Se fija `ProsusAI/finbert`, revisión
`4556d13015211d73dccd3fdd39d39232506f3e43`, sin ajustar pesos. La entrada será el
texto existente de titular y resumen, sin prefijos de ticker ni seleccionar
frases. Se preserva la información disponible y se evita introducir un segundo
experimento de representación. La elección es posterior al diagnóstico
lingüístico exploratorio, pero anterior a esta comparación bursátil; no se
presenta como una selección sobre un conjunto lingüístico independiente.

Puntuación `P(positive) - P(negative)` y etiqueta de mayor probabilidad. No se
interpreta la puntuación como probabilidad de subida de la acción. Alpha Vantage
conserva sus puntuaciones y etiquetas; ambas escalas y calibraciones difieren.

Inferencia FP32, atención eager y `eval`, sin gradientes. La GPU es opcional y se
comprueba frente a CPU en textos prefijados con tolerancia absoluta 2e-5 y mismas
etiquetas. TF32 queda desactivado. No se emplea cuantización ni otro modelo.
La caché usa la huella del texto exacto y metadatos de checkpoint, pesos,
tokenizador y entorno. Las puntuaciones exportadas incluyen IDs y huellas.

Para GPU se utiliza un entorno separado, sin sustituir PyTorch CPU del proyecto.
La [documentación oficial de versiones de PyTorch](https://pytorch.org/get-started/previous-versions/)
publica el paquete 2.8.0 con CUDA 12.8 para Windows. Se instala esa variante y las
dependencias fijadas del repositorio. No se instala ni cambia el controlador.

## Comparaciones fijadas

Primero se deduplican titulares idénticos por empresa dentro de 24 horas,
conservando la primera observación, mediante el módulo existente.

| Enfoque | Datos de noticias | Sentimiento |
| --- | --- | --- |
| `financial` | Ninguno como predictor | Ninguno |
| `alpha_all` | Todas las deduplicadas del periodo | Alpha Vantage |
| `alpha_matched` | Con contexto explícito y sin ficha dinámica detectada | Alpha Vantage |
| `finbert_matched` | Exactamente los mismos IDs de `alpha_matched` | FinBERT |

Cada híbrido tiene variante contemporánea y con un retardo de una sesión de
sentimiento medio y volumen relativo. Se utilizan las variables relativas del
módulo `sentiment_representation.relative_panel`, sin las cinco variables
adicionales ni identidad de ticker. La base tiene 9 variables, el híbrido 22 y
la variante con retardo 24. Se comprueban los mismos precios, objetivos,
calendarios, conteos y presencia de noticias entre los híbridos emparejados.

La predicción de la base se calcula una sola vez por algoritmo y bloque; se
reutiliza en ambos contrastes de variante, sin contar las copias como modelos
distintos. También se incluye una referencia de clase mayoritaria entrenada
solo con el pasado.

## Selección y presupuesto

Tres algoritmos de las últimas rondas: regresión logística, bosque aleatorio y
histogram gradient boosting. Se usan dos configuraciones iniciales por algoritmo
de `run_review.GRIDS`: C 0,1/10; bosque con 100 árboles, profundidad 4/8 y mínimo
20 por hoja; boosting con 100 iteraciones, 7/15 hojas y sin parada temprana.
Los demás parámetros son los de `build_model`. No se amplía la búsqueda tras
ver resultados. Semilla 42, umbral fijo 0,5.

La selección maximiza AUC macro por empresa en tres bloques temporales internos
expansivos. Imputación y escalado se ajustan dentro de cada entrenamiento. Se
purga cualquier etiqueta cuyo final alcance la fecha inicial de validación.
Con tres bloques externos y siete combinaciones únicas por algoritmo hay 63
ajustes externos y 378 internos, más tres referencias mayoritarias. El conjunto
externo nunca decide hiperparámetros, ventana, umbral o representación.

## Métricas e interpretación

Se informa AUC por empresa y su promedio, exactitud, exactitud equilibrada,
F1 macro de clases y MCC. Los contrastes predefinidos son:

1. `finbert_matched` menos `alpha_matched`: efecto de cambiar sentimiento.
2. `alpha_matched` menos `alpha_all`: efecto del filtro de noticias.
3. Cada híbrido menos `financial`: aportación frente a precios solos.

Se calculan intervalos pareados con 1.000 réplicas de bloques de 20 sesiones,
compartiendo las fechas entre empresas y enfoques. No incluyen incertidumbre de
reentrenamiento ni corrigen multiplicidad. Se publican todos los algoritmos y
variantes, incluso si empeoran. No se selecciona solo el mejor valor externo.

El histórico ha sido inspeccionado repetidamente. Hay riesgos de cobertura,
versiones de resúmenes, adaptación de criterios tras diagnóstico y procedencia
del preentrenamiento. Esta es una comparación retrospectiva exploratoria,
no una prueba final independiente, una demostración de causalidad, rentabilidad
o ejecución viable al precio exacto de cierre. Los experimentos anteriores y
`reports/final/` permanecen intactos. Se presentarán resultados antes de merge.
