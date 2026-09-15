# [IMPREVISTO] Evaluar modelos con validación temporal anidada e incertidumbre

### Resumen del imprevisto

Evaluar modelos con validación temporal anidada e incertidumbre.

### Tipo de imprevisto

Problema metodológico

### Contexto

Revisión objetiva del proyecto del 08/09/2026, posterior al experimento de la issue #45.

### Qué ocurre actualmente

El F1 positivo beneficia al predictor siempre alcista; las mejoras no incluyen incertidumbre y el test ya se ha consultado.

### Qué debería ocurrir

Separar ajuste interno y evaluación externa cronológica, purgar etiquetas y comparar con baseline e intervalos pareados por bloques.

### Evidencias

Véase docs/revision_objetiva_2026-09-08.md y los resultados guardados en reports/.

### Impacto en el TFG

Afecta a una parte importante

### Prioridad

Alta

### Posible solución

- [ ] Implementar folds externos e internos con purga.
- [ ] Seleccionar por media de AUC de folds internos, sin usar etiquetas externas.
- [ ] Reportar balanced accuracy, MCC, F1 macro, matrices y resultados por empresa y año.
- [ ] Calcular intervalos pareados por bloques y registrar que el histórico no es un holdout nuevo.

### Rama asociada

`feature/nested-temporal-evaluation`
