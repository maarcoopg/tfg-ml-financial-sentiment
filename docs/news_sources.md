# Fuentes de noticias financieras

## Objetivo

El objetivo es seleccionar una fuente inicial de noticias financieras que permita obtener titulares o textos asociados a los activos del proyecto.

Los activos principales para noticias seran las empresas `AAPL`, `TSLA`, `NVDA` y `MSFT`. `SPY` se mantiene como referencia de mercado, ya que al ser un ETF sus noticias no representan una empresa concreta.

## Alternativas revisadas

### Alpha Vantage News Sentiment

- Fuente: https://www.alphavantage.co/documentation/
- Endpoint: `NEWS_SENTIMENT`.
- Permite filtrar por ticker con el parametro `tickers`.
- Permite acotar el periodo con `time_from` y `time_to`.
- Devuelve hasta `limit=1000` noticias por peticion.
- Formato: JSON con `title`, `summary`, `url`, `source`, `time_published`, `overall_sentiment_score`, `overall_sentiment_label` y `ticker_sentiment`.
- Ventaja principal: fuente documentada, estable y alineada con datos financieros.
- Limitacion principal: requiere API key y control de limites de peticiones.

### Finnhub Company News

- Fuente: https://finnhub.io/docs/api/company-news
- Permite consultar noticias por simbolo y rango de fechas.
- Formato sencillo para descargar noticias por empresa.
- Ventaja principal: buena cobertura de empresas norteamericanas.
- Limitacion principal: requiere otra API key y no aporta una ventaja clara frente a Alpha Vantage para este proyecto.

### Yahoo Finance mediante yfinance

- Fuente: https://ranaroussi.github.io/yfinance/
- Permite obtener noticias recientes asociadas a busquedas o tickers.
- Ventaja principal: ya se usa `yfinance` para precios.
- Limitacion principal: es una fuente no oficial y menos adecuada para construir un dataset historico reproducible.

### Datasets publicos

- Plataformas como Kaggle o datasets academicos pueden contener noticias financieras ya recopiladas.
- Ventaja principal: pueden evitar limites de API.
- Limitacion principal: menor control sobre cobertura exacta de tickers, fechas, fuentes y actualizacion.

## Fuente seleccionada

La fuente inicial seleccionada es Alpha Vantage News Sentiment.

La seleccion se basa en estos criterios:

- Permite descargar noticias filtradas por las empresas del proyecto.
- Permite trabajar con el periodo temporal definido en el dataset financiero.
- Devuelve metadatos suficientes para trazabilidad y limpieza.
- Incluye informacion de sentimiento por noticia y por ticker, aunque el pipeline del proyecto podra recalcular sentimiento con un modelo propio.
- Es viable para el alcance del TFG si se descarga por bloques temporales y se guarda progreso local.

## Estrategia de descarga

Para evitar perder noticias por el limite de `1000` resultados por peticion, la descarga se hara por bloques temporales.

La estrategia inicial sera:

- Descargar por ticker.
- Empezar con bloques anuales.
- Si un bloque devuelve un numero cercano al limite, dividirlo en trimestres.
- Si sigue siendo demasiado amplio, dividirlo en meses.
- Guardar las respuestas en `data/raw/news/`.
- Deduplicar posteriormente por `url`, `title`, `published_at`, `source` y `ticker`.

Esta estrategia permite obtener todas las noticias disponibles en la fuente seleccionada para los activos y periodo del proyecto sin depender de una descarga manual.
