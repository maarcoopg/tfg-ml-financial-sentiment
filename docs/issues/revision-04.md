# [IMPREVISTO] Asegurar reproducibilidad y pruebas del pipeline

### Resumen del imprevisto

Asegurar reproducibilidad y pruebas del pipeline.

### Tipo de imprevisto

Error de código

### Contexto

Revisión objetiva del proyecto del 08/09/2026, posterior al experimento de la issue #45.

### Qué ocurre actualmente

Las ejecuciones parciales mezclan artefactos, se pierde configuración de retardos y las figuras y la memoria siguen usando resultados antiguos.

### Qué debería ocurrir

Crear ejecuciones aisladas con manifiestos y hashes; corregir los fallos identificados y añadir pruebas automáticas.

### Evidencias

Véase docs/revision_objetiva_2026-09-08.md y los resultados guardados en reports/.

### Impacto en el TFG

Afecta a una parte importante

### Prioridad

Alta

### Posible solución

- [ ] Guardar configuración, versiones, estado Git, hashes y predicciones por ejecución.
- [ ] Proteger configuración de características y activar muestreo de LightGBM.
- [ ] Corregir gráfico que promedia métricas y permitir visualizar cada ejecución.
- [ ] Añadir pruebas de contratos, calendario, purga, descarga y evaluación con CI.

### Rama asociada

`feature/reproducible-experiments`
