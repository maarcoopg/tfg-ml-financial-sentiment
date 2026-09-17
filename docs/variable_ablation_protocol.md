# Ablación individual de las nuevas variables

Ampliación de la issue #48, en la misma rama. Protocolo fijado antes de ejecutar
los nuevos modelos. No sustituye la primera ronda ni modifica sus resultados.

## Pregunta

¿Qué aporta cada una de las cinco nuevas variables, individualmente y en presencia
de las demás? Se estudian dispersión del sentimiento (`news_sentiment_std`),
intensidad absoluta (`news_abs_sentiment`), sorpresa del tono (`sentiment_surprise_20`),
disponibilidad de historia (`sentiment_history_ready`) y sorpresa del volumen
de noticias (`news_log_surprise_20`).

## Comparaciones

- Añadir cada variable al híbrido sin extras (`deduplicated`).
- Retirar cada variable del bloque completo (`enhanced`).
- Repetir en híbrido y en híbrido con retardo. En este último, cada variable y
  su retardo de una sesión se añaden o retiran juntos: se estudia una familia,
  no se separa el efecto contemporáneo del retardado.
- Regresión logística, bosque aleatorio y potenciación por histogramas;
  tres bloques externos, mismas empresas, fechas, etiquetas, noticias depuradas,
  purga temporal, historial completo y umbral 0,5 que en la primera ronda.
- Hiperparámetros fijos heredados de la referencia correspondiente, por bloque,
  algoritmo y variante. Se habían seleccionado mediante validación temporal
  interna. No se realiza otra búsqueda. Las referencias de añadir y retirar
  pueden tener parámetros distintos. No se busca el mejor ajuste posible para
  cada subconjunto, sino una comparación controlada con su referencia.

Son 180 ajustes nuevos, contrastados con 36 modelos de referencia cuyas
predicciones se verifican antes de entrenar. Se comprueban hashes de entradas,
versiones, columnas, parámetros, tamaños de entrenamiento y emparejamiento.
No se vuelve a entrenar la base financiera, que no cambia.

## Interpretación

Métrica principal: ROC AUC macro entre las cuatro empresas. Las 15 comparaciones
de añadir una variable al híbrido (cinco variables por tres algoritmos) son
primarias; las restantes, secundarias. Se publican todas, también por empresa
y bloque. Intervalos pareados por bloques móviles de 20 fechas, 1.000 réplicas,
sin reajuste de modelos ni corrección por comparaciones múltiples.

Una diferencia positiva al añadir sugiere utilidad en ese contexto. Una diferencia
positiva al retirar sugiere perjuicio dentro del bloque completo. No son efectos
causales ni pruebas independientes: las variables pueden interactuar y los
algoritmos responden al cambio de dimensionalidad. No se concluirá utilidad
general por el mejor resultado aislado ni se ensamblará automáticamente una
combinación ganadora seleccionada sobre estas fechas.

Los periodos externos ya se han inspeccionado repetidamente. Es un análisis
exploratorio, no un conjunto de prueba nuevo ni evidencia de rentabilidad.

## Ejecución

```powershell
.venv\Scripts\python.exe -m src.experiments.run_variable_ablation --run-dir reports/experiments/variable-ablation-full-20260917
```

Los modelos permanecen en `models/experiments/` y no se suben a GitHub.
