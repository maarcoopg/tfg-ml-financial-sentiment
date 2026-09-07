# Tratamiento de dias sin noticias

## Problema

Despues de unir el dataset financiero con el sentimiento diario, algunos dias bursatiles no tienen noticias asociadas al ticker.

Estos casos no deben quedar como valores nulos sin interpretar, porque el modelo podria tratarlos como datos perdidos en vez de como ausencia de noticias.

## Criterio aplicado

Los dias bursatiles sin noticias se tratan como dias sin informacion textual nueva:

- `news_count = 0`
- `positive_news_count = 0`
- `negative_news_count = 0`
- `neutral_news_count = 0`
- `non_trading_news_count = 0`
- medias y extremos de sentimiento igual a `0.0`
- ratios de sentimiento igual a `0.0`
- `has_news = False`

Cuando un dia tiene noticias, `has_news = True` y se conservan los valores agregados reales.

## Justificacion

La puntuacion `0.0` representa ausencia de sesgo positivo o negativo. La columna `has_news` permite al modelo distinguir entre neutralidad por falta de noticias y neutralidad calculada a partir de noticias reales.

Esta decision evita perdida de filas financieras y mantiene una serie temporal continua por ticker.
