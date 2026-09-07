# Alineacion de noticias con calendario bursatil

## Problema

Las noticias financieras pueden publicarse en fines de semana, festivos o despues del ultimo dia bursatil disponible en el dataset de precios.

Para unir noticias con datos financieros es necesario asignar cada noticia a una fecha bursatil valida.

## Criterio aplicado

Cada noticia se asigna al siguiente dia de cotizacion disponible para su ticker.

Ejemplos:

- Una noticia publicada en sabado se asigna al lunes siguiente si hay sesion.
- Una noticia publicada en festivo se asigna al siguiente dia bursatil.
- Una noticia publicada en un dia bursatil se mantiene en ese mismo dia.

## Casos no alineables

Si una noticia se publica despues del ultimo dia bursatil disponible en el dataset financiero, no se puede asignar sin incorporar datos futuros de precios.

Estos casos se guardan separados en:

```text
data/processed/news/financial_news_unaligned.csv
```

El dataset alineado se guarda en:

```text
data/processed/news/financial_news_trading_days.csv
```

## Variables generadas

- `trading_date`: fecha bursatil asignada.
- `original_published_date`: fecha original de publicacion.
- `is_non_trading_day`: indica si la noticia fue reasignada a otro dia.

Esta estrategia evita perder noticias de fines de semana o festivos y mantiene el dataset alineado con el calendario bursatil usado por los datos financieros.
