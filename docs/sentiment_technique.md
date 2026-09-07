# Tecnica de analisis de sentimiento financiero

## Objetivo

El objetivo es seleccionar una tecnica reproducible para clasificar noticias financieras como positivas, negativas o neutras y generar variables de sentimiento para el modelo hibrido.

## Alternativas revisadas

### Diccionarios generales

Ejemplos: TextBlob o VADER.

- Ventaja: faciles de integrar y rapidos.
- Limitacion: no estan entrenados especificamente para lenguaje financiero.
- Riesgo: palabras con significado financiero pueden interpretarse mal en un contexto general.

### Diccionarios financieros

Ejemplo: Loughran-McDonald.

- Ventaja: vocabulario adaptado al dominio financiero.
- Limitacion: requiere construir reglas y calibrar umbrales.
- Riesgo: menor sensibilidad al contexto completo de la frase.

### Modelos transformer financieros

Ejemplo: `ProsusAI/finbert`.

- Fuente: https://huggingface.co/ProsusAI/finbert
- Modelo basado en BERT y ajustado para sentimiento financiero.
- Devuelve probabilidades para tres clases: `positive`, `negative` y `neutral`.
- Puede integrarse en Python mediante `transformers`.

## Tecnica seleccionada

La tecnica operativa seleccionada para la primera version del pipeline es el sentimiento por ticker devuelto por Alpha Vantage News Sentiment.

La seleccion se justifica por:

- Esta calculado para lenguaje financiero y asociado al ticker concreto de la noticia.
- Genera una puntuacion continua y una etiqueta interpretable.
- Evita volver a procesar decenas de miles de textos con un modelo pesado en local.
- Es reproducible porque queda almacenado junto con la noticia descargada.
- Encaja con el objetivo del TFG de combinar analisis tecnico y sentimiento financiero.

FinBERT, usando el modelo `ProsusAI/finbert`, queda como alternativa avanzada si se quiere recalcular el sentimiento directamente sobre `title` y `summary`.

## Integracion prevista

El pipeline usara como entrada principal:

- `ticker_sentiment_score`
- `ticker_sentiment_label`

Para cada noticia se guardaran:

- `sentiment_label`
- `sentiment_score`
- `sentiment_label_raw`
- `sentiment_source`

La puntuacion `sentiment_score` se tomara de:

```text
ticker_sentiment_score
```

Este valor mantiene una interpretacion sencilla:

- valores positivos: tono favorable
- valores cercanos a cero: tono neutro
- valores negativos: tono desfavorable

## Prueba inicial

La prueba inicial se hara sobre un subconjunto pequeno de noticias limpias para comprobar que:

- La puntuacion numerica esta disponible.
- La etiqueta original esta disponible.
- No se generan valores nulos.
- Los ejemplos clasificados son coherentes de forma manual.
