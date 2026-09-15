# [IMPREVISTO] Auditar la cobertura y reforzar la descarga de noticias

### Resumen del imprevisto

Auditar la cobertura y reforzar la descarga de noticias.

### Tipo de imprevisto

Problema con datos

### Contexto

Revisión objetiva del proyecto del 08/09/2026, posterior al experimento de la issue #45.

### Qué ocurre actualmente

La cobertura crece de 3.675 noticias en 2024 a 27.381 en 2025; una respuesta vacía puede romper la reanudación.

### Qué debería ocurrir

Generar auditoría por año, empresa y fuente; distinguir cobertura desconocida de ausencia de registros; tratar errores, respuestas vacías y saturación sin afirmar exhaustividad.

### Evidencias

Véase docs/revision_objetiva_2026-09-08.md y los resultados guardados en reports/.

### Impacto en el TFG

Afecta a una parte importante

### Prioridad

Alta

### Posible solución

- [ ] Auditar volumen, fuentes, relevancia y duplicados.
- [ ] Guardar informes de cobertura junto a cada ejecución.
- [ ] Guardar chunks vacíos con esquema y reintentar fallos transitorios.
- [ ] Detener la descarga con explicación si un único día alcanza el límite.

### Rama asociada

`feature/news-coverage-audit`
