# Implementación de la revisión metodológica

## Estado de publicación

La integración de la issue #45 en main fue autorizada por el usuario. El 8 de septiembre no se pudo publicar por restricciones de escritura del entorno. El 15 de septiembre se confirmó el acceso de escritura a Git y el usuario solicitó publicar e integrar el trabajo acumulado desde `feature/lagged-sentiment-features`, cerrando la issue #45 al llegar a main.

Las cinco propuestas están conservadas con la plantilla de imprevistos en `docs/issues/revision-01.md` a `revision-05.md`. Esos nombres son identificadores locales, no números de GitHub. Las ramas de los borradores documentan la planificación original; sus implementaciones se incluyen en la entrega acumulada de la rama de la issue #45. No se deben interpretar como cinco issues publicadas o cerradas en GitHub.

## Ejecución recomendada

Instalar las dependencias fijadas en `requirements.txt` en un entorno virtual. En Windows:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m src.experiments.run_review
```

La ejecución lee los precios y noticias ya descargados. No hace consultas a Alpha Vantage ni consume créditos. Cada ejecución crea un directorio nuevo en `reports/experiments/`; `--run-dir` permite elegir otro directorio dentro del proyecto, que debe no existir. Los modelos, datos e instantáneas se guardan fuera de reports, con el mismo identificador y rutas registradas en el manifiesto. Véase `docs/report_organization.md`. Las ejecuciones fallidas conservan su estado y no se mezclan con las completadas.

Para una comprobación reducida:

```powershell
.venv/Scripts/python.exe -m src.experiments.run_review --run-dir artifacts/verification/comprobacion-nueva --models logistic_regression --variants base hybrid lag_1 --outer-splits 2 --inner-splits 2 --bootstrap-repeats 100
```

## Instante de predicción

Se plantea clasificación después de conocer el cierre de la sesión t, sobre la dirección cierre(t)-cierre(t+1). No se afirma que sea posible ejecutar una orden al cierre exacto utilizado como característica.

Cada noticia se asigna al primer cierre mayor o igual que su timestamp. El calendario NASDAQ incluye festivos, horario de verano y cierres anticipados. El caso de igualdad se considera disponible; la publicación se usa como aproximación de disponibilidad, pues el histórico no proporciona la hora de recepción ni la latencia de cálculo del sentimiento.

Se interpreta el timestamp original sin zona como UTC; la opción `--source-timezone` hace explícita esta hipótesis. La documentación de Alpha Vantage describe los filtros horarios de NEWS_SENTIMENT en UTC, pero no se dispone de una auditoría independiente de cada timestamp ni de las revisiones del proveedor. El calendario se obtiene de una librería fijada, no de reglas bursátiles escritas a mano.

Fuentes: [Alpha Vantage](https://www.alphavantage.co/documentation/#news-sentiment) y [pandas_market_calendars](https://pandas-market-calendars.readthedocs.io/en/latest/usage.html).

## Validación temporal

El periodo externo empieza por defecto el 16/10/2023 y se divide en tres bloques de fechas. Cada bloque ajusta el modelo con datos anteriores y selecciona parámetros mediante tres folds internos expansivos. Las observaciones de una fecha permanecen juntas. Se exige que el final del horizonte de todas las etiquetas de entrenamiento sea estrictamente anterior al inicio de validación.

Se ensayan dos configuraciones por algoritmo, seleccionadas por la media de AUC de los folds internos. El umbral principal es 0,5. El baseline `dummy` aprende la clase mayoritaria solo en entrenamiento. No se decide ningún parámetro a partir de las etiquetas del bloque externo.

Se guardan AUC agrupado y media/desviación de AUC por fold externo; no son intercambiables. Los intervalos se calculan para el AUC agrupado mediante bloques de 20 sesiones y réplicas pareadas. La comparación predefinida `lag_1/lightgbm` frente a `hybrid/lightgbm` añade sensibilidad a bloques de 5 y 60 sesiones. Los intervalos son percentiles del 95 %, no corrigen múltiples comparaciones y no prueban una mejora por sí solos.

Todo el histórico ya ha sido inspeccionado en trabajos anteriores. La validación anidada mejora el procedimiento, pero no convierte estas fechas en un holdout nuevo. Una confirmación independiente requiere un periodo no utilizado y un protocolo congelado antes de evaluarlo.

## Comparaciones controladas

Se mantienen iguales las filas y el protocolo para las doce variantes:

- `base`: variables financieras originales.
- `news_volume`: base más cantidad y presencia de registros de noticias.
- `sentiment_only`: scores, ratios de tono y presencia de noticias, sin precios ni recuentos.
- `financial_sentiment`: base más esas variables de tono, sin recuentos.
- `hybrid`: base y bloque de sentimiento original completo.
- `lag_1`, `lag_2`, `lag_3`: híbrido más un único retardo de media de sentimiento y cantidad de noticias.
- `rolling_3`, `rolling_5`: híbrido más una única ventana móvil de esas dos variables, incluyendo la sesión actual disponible.
- `weighted_sentiment`: sustituye la media simple por la media ponderada por relevancia.
- `relative_features`: sustituye niveles de precio por distancias a medias, MACD relativo y volumen relativo a su historia; es una hipótesis de representación conjunta, no permite atribuir el efecto a cada transformación individual.

Se comparan regresión logística, Random Forest, HistGradientBoosting, XGBoost y LightGBM. Con la configuración completa son 1.080 ajustes internos y 183 entrenamientos externos, incluyendo tres baselines. Los estimadores finales de cada bloque quedan guardados.

## Salidas

- `manifest.json`: configuración previa, versiones, estado, hashes de entradas y salidas y estado Git.
- `source/`: copia del código y requisitos al comenzar las nuevas ejecuciones.
- `data/`: dataset temporal corregido, calendario y asignación de noticias.
- `coverage/`: cobertura por año y empresa, fuentes y relevancia.
- `inner_cv.csv`, `selected_params.csv`: búsquedas y fronteras efectivas.
- `predictions.csv`, `metrics.csv`, `global_metrics.csv`: predicciones y métricas globales, por empresa, año y fold.
- `auc_intervals.csv`: intervalos pareados y referencias de comparación.
- `models/`: modelos de cada bloque externo.
- `figures/`: figuras separadas de AUC, balanced accuracy, F1 macro y MCC.

El primer ensayo completo se inició antes de añadir la copia `source/`; conserva hashes del código al inicio, pero no una copia de cada fuente. Las ejecuciones posteriores incorporan esa copia. Esta limitación de trazabilidad queda explícita y no afecta al emparejamiento de predicciones ni al cálculo de sus métricas.

## Compatibilidad con scripts anteriores

Los informes anteriores se conservan como históricos. El camino recomendado es el comando completo anterior, que reconstruye los datos con la alineación corregida.

Los scripts de entrenamiento tradicionales admiten una ruta por ejecución:

```powershell
python src/models/train_models.py --dataset base hybrid --run-dir reports/experiments/entrenamiento-nuevo
python src/models/evaluate_models.py --run-dir reports/experiments/entrenamiento-nuevo
python src/models/compare_models.py --run-dir reports/experiments/entrenamiento-nuevo
python src/models/tune_temporal_cv.py --datasets base hybrid lagged_hybrid --run-dir reports/experiments/tuning-nuevo
```

Estos scripts consumen los CSV de modelado disponibles: no se debe confundir ajustar un CSV histórico con reconstruir la alineación temporal. El tuning parcial no compara contra artefactos de otra ejecución automáticamente. Para visualizar la revisión completa:

```powershell
python -m src.visualization.plot_experiment reports/experiments/NOMBRE-DE-EJECUCION
```

## Límites que siguen abiertos

La cobertura completa del proveedor no puede demostrarse con los registros descargados. Se informa `coverage_verified=false`; no se equipara falta de registros a ausencia de noticias. Se identifican títulos repetidos, pero no se eliminan automáticamente porque pueden ser actualizaciones reales. Una descarga que satura un único día se detiene con explicación en lugar de afirmar que está completa.

El sentimiento sigue siendo el score externo de Alpha Vantage. No se ha entrenado un modelo textual ni se dispone de etiquetas humanas para validar ese score. No se ha construido un backtest económico ni se ha cambiado el horizonte objetivo. Son extensiones distintas que no se necesitan para corregir la comparación actual.
