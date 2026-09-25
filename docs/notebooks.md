# Notebooks de análisis

Los dos primeros cuadernos se leen en orden y fijan explícitamente la ejecución
`review-full-20260908`. Incluyen explicaciones antes de los análisis, tablas,
gráficos y conclusiones. No entrenan modelos ni realizan llamadas a proveedores.

## 01. Datos y calidad

[01_dataset_exploration.ipynb](../notebooks/01_dataset_exploration.ipynb) explica
qué representa cada fila, el objetivo, las variables, los precios, el balance
de clases, la cobertura de noticias, las sesiones sin registros y la purga.

Si existe `data/experiments/review-full-20260908/dataset.csv`, lo consulta.
Si falta, reconstruye el panel con las funciones de `src/` y las entradas
guardadas en `data/raw/` y `data/processed/`, usando un directorio temporal.
No modifica esas entradas ni sustituye resultados. Si faltan, indica sus rutas.
Cambiar `REBUILD_DATASET` a `True` permite comprobar esta alternativa.

Los hashes de las entradas se contrastan al reconstruir. Una diferencia genera
un aviso: el panel reconstruido con datos o código distintos no debe confundirse
con la instantánea original. La cobertura mostrada sigue siendo el informe
conservado de la ejecución seleccionada.

## 02. Resultados e incertidumbre

[02_model_results.ipynb](../notebooks/02_model_results.ipynb) comprueba las claves
de las predicciones y recalcula sus AUC. Después compara base e híbrido, muestra
el baseline, las doce variantes, el desglose externo y por empresa, y los
intervalos pareados. La evolución histórica aparece al final, claramente separada.

Este cuaderno solo necesita los informes versionados; no requiere `.joblib`,
datasets de ejecución ni instantáneas locales de código. Los AUC agrupados y
los promedios por bloque se presentan como medidas diferentes.

## 03. Modelos independientes por empresa

[03_per_company_models.ipynb](../notebooks/03_per_company_models.ipynb) analiza
el experimento separado `per-company-20260916`: AAPL, MSFT, NVDA y TSLA,
cada una con su propio entrenamiento y noticias, para base, híbrido e híbrido
con retardo de una sesión. Incluye cinco algoritmos y una referencia mayoritaria.

La referencia conjunta se vuelve a entrenar con el mismo protocolo. El cuaderno
verifica que se comparan las mismas observaciones, recalcula los AUC y comprueba
los hashes de los informes. Muestra cobertura, parámetros, métricas por empresa
y bloque, intervalos pareados y promedios macro, sin necesitar modelos `.joblib`
ni datos locales de la ejecución. No sustituye los resultados de los cuadernos
anteriores ni convierte esta exploración histórica en una validación final.

Para regenerar el entrenamiento, desde la raíz y con un identificador nuevo:

```powershell
python -m src.experiments.run_per_company --run-dir reports/experiments/per-company-NUEVA-EJECUCION
```

## 04. Modelos conjuntos con identidad de empresa

[04_ticker_aware_models.ipynb](../notebooks/04_ticker_aware_models.ipynb) estudia
`ticker-aware-full-20260916` sin sobrescribir el experimento anterior. Compara
identificador de empresa, variables relativas y ambas modificaciones para los
cinco algoritmos. En regresión logística añade interacciones empresa-variable.

Incluye base, híbrido y retardo de una sesión, referencias conjunta e individual,
AUC macro y por empresa, estabilidad por bloque e intervalos pareados. Comprueba
las claves, recalcula las métricas y verifica hashes usando solo informes
versionados. Las comparaciones siguen siendo exploratorias.

```powershell
python -m src.experiments.run_ticker_aware --run-dir reports/experiments/ticker-aware-NUEVA-EJECUCION
```

La reutilización de las referencias exige las mismas entradas, panel, código
principal, versiones, rejillas y protocolo; el ejecutor se detiene si no coinciden.

## 05. Primera ronda controlada de mejoras

[05_controlled_improvement_round.ipynb](../notebooks/05_controlled_improvement_round.ipynb)
compara las siete etapas de `first-round-full-20260916`: referencia relativa,
deduplicación, ventanas de tres y cinco años, sentimiento enriquecido, selección
macro y ajuste ampliado. Mantiene separados los experimentos anteriores.

Incluye auditoría de cobertura y fuentes, tamaños de variables, fronteras,
candidatos y selecciones internas, AUC macro y por empresa, aciertos e intervalos
pareados. Los resúmenes promedian tres algoritmos prefijados, no cinco. Verifica
huellas, emparejamiento y métricas sin cargar modelos ni consumir API.

El [protocolo](first_round_protocol.md) explica los controles y las limitaciones.
Para repetir la ejecución completa con las entradas locales y un identificador nuevo:

```powershell
python -m src.experiments.run_first_round --run-dir reports/experiments/first-round-NUEVA-EJECUCION
```

La referencia anterior debe estar disponible y conservar entradas, versiones y
fronteras. El ejecutor comprueba que reproduce sus predicciones antes de evaluar cambios.

## 06. Ablación individual de variables

[06_sentiment_variable_ablation.ipynb](../notebooks/06_sentiment_variable_ablation.ipynb)
separa las cinco variables añadidas en la primera ronda. Compara añadir cada una
a la referencia sin extras y retirarla del bloque completo, con parámetros fijos
heredados de cada referencia. En retardo se estudia la variable y su retardo juntos.

Incluye los 60 contrastes macro, intervalos pareados, diferencias por empresa y
por bloque, comprobaciones de hashes y recálculo de AUC. Lee exclusivamente
`variable-ablation-full-20260917`; no entrena ni necesita los modelos locales.
El [protocolo previo](variable_ablation_protocol.md) explica los controles y
el carácter exploratorio sobre fechas ya inspeccionadas.

## 07. Funcionamiento interno de FinBERT

[07_finbert_model_understanding.ipynb](../notebooks/07_finbert_model_understanding.ipynb)
recorre tokenización, embeddings, atención, conexiones residuales, normalización,
redes internas, pooler y clasificación. Usa un titular real seleccionado por una
regla fija y ejemplos sintéticos identificados. Verifica 28 operaciones con los
pesos reales y explica qué no puede concluirse de las visualizaciones.

La lectura puede hacerse con las salidas conservadas. Para repetir las celdas
requiere las dependencias opcionales, el CSV local de noticias y los pesos en
caché. Véase [la guía de FinBERT](finbert_modelo.md). No compara otros modelos de
lenguaje, no evalúa calidad lingüística ni entrena clasificadores financieros.

```powershell
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-finbert.txt
python -m src.nlp.inspect_finbert --download
```

## 08. Diagnóstico del sentimiento de FinBERT

[08_finbert_sentiment_evaluation.ipynb](../notebooks/08_finbert_sentiment_evaluation.ipynb)
lee `finbert-sentiment-20260925`: auditoría, muestra temporal ciega, disponibilidad
de contexto por empresa, sensibilidad a las representaciones, siete pares
sintéticos y gradientes integrados con dos referencias y perturbaciones.

No ejecuta el modelo, no descarga pesos ni requiere las dependencias opcionales
para leer y reproducir sus tablas y figuras. La inferencia del informe se limita
a 200 parejas de desarrollo. Las otras 200 permanecen reservadas. La calidad
lingüística sigue pendiente de anotación humana: coincidencia entre sistemas no
equivale a exactitud. El [protocolo](finbert_evaluacion_protocolo.md) explica las
plantillas y el comando que rechaza etiquetas incompletas.

## Preparación y ejecución

Desde la raíz, con el entorno del proyecto activado:

```powershell
python -m pip install -r requirements.txt
python -m ipykernel install --user --name tfg-finanzas --display-name "Python (TFG Finanzas)"
python -m jupyter lab notebooks/
```

Selecciona el kernel **Python (TFG Finanzas)** y ejecuta todas las celdas en
orden. El kernel debe usar el mismo intérprete que tiene las dependencias;
abrir Jupyter desde un entorno no garantiza que un kernel antiguo lo utilice.
Los cuadernos reconocen tanto la raíz del repositorio como `notebooks/`.

Para ejecutarlos sin interfaz y conservar las salidas:

```powershell
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=tfg-finanzas --ExecutePreprocessor.timeout=180 notebooks/01_dataset_exploration.ipynb notebooks/02_model_results.ipynb notebooks/03_per_company_models.ipynb notebooks/04_ticker_aware_models.ipynb notebooks/05_controlled_improvement_round.ipynb notebooks/06_sentiment_variable_ablation.ipynb
```

Las salidas incluidas se han generado mediante ejecución completa, no mediante
tablas o gráficos simulados. Pueden regenerarse con las mismas entradas.
El único aviso del entorno Windows observado durante la ejecución se refiere
al bucle de eventos de Jupyter; no representa una excepción de las celdas.

## Interpretación

Una ausencia de noticias registradas no certifica ausencia de noticias reales.
Un F1 positivo alto puede corresponder a predecir siempre subida. Un AUC cercano
a 0,5 no demuestra rentabilidad; tampoco los máximos entre muchas variantes
constituyen una validación independiente. La evidencia y sus límites se
desarrollan en [el borrador de memoria](borrador_memoria.md).
