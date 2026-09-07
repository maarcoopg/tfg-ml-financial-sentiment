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

La tecnica seleccionada es FinBERT, usando el modelo `ProsusAI/finbert`.

La seleccion se justifica por:

- Esta orientado especificamente a textos financieros.
- Genera directamente las tres clases necesarias para el proyecto.
- Permite obtener una puntuacion continua ademas de una clase.
- Se puede aplicar de forma reproducible sobre `title` y `summary`.
- Encaja con el objetivo del TFG de combinar analisis tecnico y sentimiento financiero.

## Integracion prevista

El pipeline aplicara FinBERT sobre una columna de texto construida a partir de:

- `title`
- `summary`, cuando este disponible

Para cada noticia se guardaran:

- `sentiment_label`
- `sentiment_score`
- `sentiment_positive`
- `sentiment_negative`
- `sentiment_neutral`

La puntuacion `sentiment_score` se calculara como:

```text
sentiment_positive - sentiment_negative
```

Este valor mantiene una interpretacion sencilla:

- valores positivos: tono favorable
- valores cercanos a cero: tono neutro
- valores negativos: tono desfavorable

## Prueba inicial

La prueba inicial se hara sobre un subconjunto pequeno de noticias limpias para comprobar que:

- El modelo carga correctamente.
- Las tres clases aparecen en el resultado.
- Las probabilidades suman aproximadamente 1.
- Los ejemplos clasificados son coherentes de forma manual.

Si no se dispone del modelo localmente, el script debera fallar con un mensaje claro indicando que faltan las dependencias `transformers` y `torch`.
