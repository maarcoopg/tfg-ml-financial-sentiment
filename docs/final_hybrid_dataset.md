# Dataset hibrido final

## Archivo final

El dataset final se guarda en:

```text
data/processed/hybrid/final_hybrid_dataset.csv
```

## Columnas incluidas

El dataset contiene:

- Identificacion: `ticker`, `Date`.
- Precios y volumen: `Adj Close`, `Close`, `High`, `Low`, `Open`, `Volume`.
- Indicadores tecnicos: `daily_return`, `sma_5`, `sma_20`, `sma_50`, `RSI`, `MACD`, `MACD_signal`, `volatility_20`.
- Sentimiento diario: `sentiment_mean`, `sentiment_median`, `sentiment_min`, `sentiment_max`.
- Volumen de noticias: `news_count`, `non_trading_news_count`, `positive_news_count`, `negative_news_count`, `neutral_news_count`.
- Ratios de sentimiento: `positive_news_ratio`, `negative_news_ratio`, `neutral_news_ratio`.
- Indicador auxiliar: `has_news`.
- Variable objetivo: `target`.

## Columnas excluidas

`next_adj_close` se excluye del dataset final porque se uso solo para construir `target`. Mantenerla como variable predictora introduciria informacion futura.

El dataset final queda preparado para dividir entrenamiento y prueba respetando el orden temporal.
