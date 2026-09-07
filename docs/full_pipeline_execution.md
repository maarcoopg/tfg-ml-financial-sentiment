# Ejecución completa del pipeline

Esta guía resume cómo reproducir el flujo principal del proyecto desde la descarga de datos hasta las métricas y figuras finales.

## 1. Entorno

Instalar dependencias:

```bash
pip install -r requirements.txt
```

La descarga de noticias requiere un archivo `.env` con:

```text
ALPHA_VANTAGE_API_KEY=...
```

## 2. Datos financieros

Descargar precios diarios:

```bash
python src/data/download_prices.py
```

Crear la variable objetivo:

```bash
python src/data/create_target.py
```

Calcular variables financieras:

```bash
python src/data/calculate_daily_returns.py
python src/data/calculate_moving_averages.py
python src/data/calculate_rsi.py
python src/data/calculate_macd.py
python src/data/calculate_volatility.py
```

## 3. Noticias y sentimiento

Descargar noticias financieras:

```bash
python src/data/download_financial_news.py
```

Limpiar, extraer y clasificar sentimiento:

```bash
python src/data/clean_financial_news.py
python src/data/extract_news_sentiment.py
python src/data/classify_news_sentiment.py
```

Alinear noticias con calendario bursátil y agregarlas por día:

```bash
python src/data/align_news_to_trading_days.py
python src/data/aggregate_daily_sentiment.py
```

## 4. Dataset híbrido

Unir datos financieros y sentimiento:

```bash
python src/data/merge_financial_sentiment_data.py
```

Rellenar días sin noticias:

```bash
python src/data/handle_days_without_news.py
```

Crear el dataset híbrido final:

```bash
python src/data/create_final_hybrid_dataset.py
```

## 5. División temporal

Generar datasets de entrenamiento y prueba:

```bash
python src/data/split_train_test.py
```

El split usa el 80% inicial de fechas únicas para entrenamiento y el 20% final para prueba. No se usa división aleatoria.

## 6. Modelado

Entrenar modelos base:

```bash
python src/models/train_models.py --dataset base
```

Entrenar modelos híbridos:

```bash
python src/models/train_models.py --dataset hybrid
```

Los modelos disponibles son:

- `dummy`
- `logistic_regression`
- `random_forest`
- `hist_gradient_boosting`

## 7. Evaluación y comparación

Consolidar métricas:

```bash
python src/models/evaluate_models.py
```

Comparar modelo base e híbrido:

```bash
python src/models/compare_models.py
```

Analizar importancia de variables:

```bash
python src/models/analyze_feature_importance.py
```

Generar visualizaciones:

```bash
python src/visualization/plot_results.py
```

## 8. Salidas principales

- Datos procesados: `data/processed/`
- Modelos entrenados: `models/trained/`
- Predicciones: `reports/predictions/`
- Métricas: `reports/metrics/`
- Comparación: `reports/comparison/`
- Importancia de variables: `reports/feature_importance/`
- Figuras: `reports/figures/`

## Nota sobre SPY

SPY se descargó como referencia de mercado dentro de los datos financieros iniciales. No entra en los entrenamientos ni en la comparación principal, porque el análisis de sentimiento se plantea sobre empresas concretas y no sobre ETFs. Esta exclusión evita comparar modelos entrenados sobre universos de activos distintos.
