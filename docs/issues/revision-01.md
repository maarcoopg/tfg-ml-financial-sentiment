# [IMPREVISTO] Corregir la alineación temporal de noticias y etiquetas

### Resumen del imprevisto

Corregir la disponibilidad temporal de noticias y etiquetas para emitir predicciones al cierre de cada sesión.

### Tipo de imprevisto

Problema metodológico

### Contexto

La revisión objetiva del 08/09/2026 detecta que la alineación utiliza solo la fecha de publicación y que las fronteras temporales no purgan etiquetas.

### Qué ocurre actualmente

Las noticias posteriores al cierre se incorporan a la misma sesión y la última etiqueta de entrenamiento alcanza el primer día de validación.

### Qué debería ocurrir

Usar timestamps normalizados, cierres reales del calendario bursátil y fronteras que excluyan etiquetas no disponibles antes de validar.

### Evidencias

src/data/align_news_to_trading_days.py y src/models/tune_temporal_cv.py; docs/revision_objetiva_2026-09-08.md.

### Impacto en el TFG

Afecta a una parte importante

### Prioridad

Alta

### Posible solución

- [ ] Definir el instante de predicción al cierre y documentar la zona horaria de origen.
- [ ] Asignar cada noticia al primer cierre posterior a su publicación, incluidos festivos y cierres anticipados.
- [ ] Purgar etiquetas en train/test y validación temporal.
- [ ] Probar los casos límite y regenerar datos en una ejecución aislada.

### Rama asociada

`feature/point-in-time-alignment`
