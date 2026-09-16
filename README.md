# tfg-ml-financial-sentiment

Trabajo de Fin de Grado sobre un modelo híbrido de aprendizaje automático para la predicción de tendencias bursátiles mediante análisis técnico y sentimiento financiero.

## Objetivo

El proyecto compara dos enfoques de clasificación binaria:

- Modelo financiero base: usa precios, volumen, retornos e indicadores técnicos.
- Modelo híbrido: usa las mismas variables financieras y añade sentimiento financiero agregado por día.

La variable objetivo vale `1` cuando el precio ajustado de cierre de la siguiente sesión es superior al de la sesión actual, y `0` en caso contrario.

## Activos

Los datos financieros iniciales incluyen AAPL, MSFT, NVDA, TSLA y SPY. Para el entrenamiento y la comparación principal se usan AAPL, MSFT, NVDA y TSLA. SPY queda excluido porque es un ETF y no existe una señal de sentimiento corporativo comparable con la de las empresas.

## Ejecución

La revisión metodológica se ejecuta con `python -m src.experiments.run_review`.
Usa horarios reales del mercado, validación temporal anidada con purga y salidas
aisladas por ejecución. Véase [la guía de ejecución](docs/full_pipeline_execution.md). Los informes
anteriores a esta revisión son históricos exploratorios.

La guía completa de ejecución está en:

```text
docs/full_pipeline_execution.md
```

El protocolo corregido se ejecuta con:

```bash
python -m src.experiments.run_review
```

## Documentación

- [Borrador de la memoria del TFG](docs/borrador_memoria.md): problema, decisiones, metodología, resultados y limitaciones.
- [Ejecución del pipeline](docs/full_pipeline_execution.md): preparación y reproducción.
- [Resultados de la revisión temporal](docs/resultados_revision_temporal.md): evidencia del experimento principal.
- [Organización de artefactos](docs/report_organization.md): informes, modelos, datos y trazabilidad.
- [Notebooks de análisis](docs/notebooks.md): exploración guiada y comparación de modelos.

La memoria consolida la documentación anteriormente dividida por tareas. Las plantillas de GitHub se conservan en `.github/ISSUE_TEMPLATE/`; la documentación versionada en `docs/` ya no contiene borradores de issues.
