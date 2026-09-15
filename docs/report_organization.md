# Organización de resultados

El README permanece en la raíz. No hay un README adicional en `reports/final/`.

## Directorios

| Ruta | Contenido |
| --- | --- |
| `reports/final/tables/` | Métricas globales e intervalos del experimento seleccionado para la memoria. |
| `reports/final/figures/` | Cuatro figuras del mismo experimento, sin mezclar pruebas técnicas. |
| `reports/final/provenance.json` | Ejecución de origen, rutas, hashes y límites de la evaluación. |
| `reports/experiments/<id>/` | Métricas, predicciones, tuning, cobertura, figuras y manifiesto de cada experimento. |
| `reports/historical/` | Resultados anteriores, conservados sin modificar su contenido. |
| `models/experiments/<id>/` | Modelos serializados, excluidos de Git. |
| `data/experiments/<id>/` | Datos, calendario y asignaciones temporales utilizados, excluidos de Git. |
| `artifacts/snapshots/<id>/` | Código y dependencias de cada ejecución, excluidos de Git. |
| `artifacts/verification/<id>/` | Salidas de pruebas técnicas, excluidas de Git. |

Los modelos históricos se conservan en `models/experiments/legacy-trained/` y
`models/experiments/legacy-tuned/`. No se modifican los datos de entrada de
`data/raw/` ni de `data/processed/`.

Los directorios de informes, modelos, datos e instantáneas comparten el mismo
identificador. Los manifiestos de versión 2 contienen `artifact_paths` y
`output_sha256`, con rutas relativas a la raíz del proyecto. Los manifiestos
anteriores se conservan como `original_manifest.json` en las instantáneas.
Las fechas, parámetros y hashes del código original no se reemplazan por los
de la reorganización. La migración de rutas queda registrada por separado.

Los registros `reports/historical/migration_manifest.json` y
`reports/historical/model_migration_manifest.json` relacionan las rutas antiguas
con las nuevas y sus hashes. Las carpetas antiguas vacías que Windows no permita
eliminar no contienen resultados y no vuelven a utilizarse.

## Ejecución

```powershell
python -m src.experiments.run_review
python -m src.experiments.run_review --run-dir artifacts/verification/comprobacion-nueva --models logistic_regression --variants base hybrid lag_1 --outer-splits 2 --inner-splits 2 --bootstrap-repeats 100
python -m src.visualization.plot_experiment reports/experiments/IDENTIFICADOR
```

Las nuevas ejecuciones se guardan por defecto en `reports/experiments/`.
`--run-dir` debe apuntar a una carpeta nueva dentro del proyecto y su identificador
no puede coincidir con el de otra ejecución. Para las pruebas técnicas se debe
elegir explícitamente `artifacts/verification/<id>`.

El flujo tradicional también permite usar `--run-dir` en entrenamiento,
evaluación, comparación, importancia de variables y gráficos. Los lectores
tradicionales sin ese argumento consultan los resultados históricos; no buscan
automáticamente el último experimento. Para nuevos resultados se debe pasar
el mismo directorio a todas las fases.

La selección actual en `reports/final/` procede de `review-full-20260908`.
"Final" designa los archivos seleccionados para redactar la memoria, no una
prueba independiente ni evidencia concluyente de capacidad predictiva. Se
mantiene el carácter exploratorio y la reutilización del histórico evaluado.

`python -m src.experiments.organize_reports --publish-final reports/experiments/IDENTIFICADOR`
crea una selección con procedencia verificable únicamente si no existe ya
`reports/final/`. Nunca reemplaza silenciosamente una selección anterior.
