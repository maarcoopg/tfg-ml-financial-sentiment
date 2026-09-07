# Metodología de datos y preprocesamiento

## Fuentes de datos

El proyecto utiliza dos tipos de datos:

- Datos financieros diarios descargados con `yfinance`.
- Noticias financieras obtenidas mediante la API de Alpha Vantage News Sentiment.

Los activos financieros descargados inicialmente son AAPL, MSFT, NVDA, TSLA y SPY. Para el entrenamiento y la comparación principal se utilizan AAPL, MSFT, NVDA y TSLA. SPY se conserva como referencia de mercado, pero se excluye del modelado porque es un ETF y no dispone de una señal de sentimiento corporativo comparable a la de las empresas.

## Horizonte temporal

El periodo de trabajo cubre datos desde 2015 hasta 2025. La información se organiza a frecuencia diaria y se alinea con los días de negociación bursátil.

## Variable objetivo

La variable objetivo se define como una clasificación binaria de la tendencia de la siguiente sesión:

```text
target = 1 si Adj Close(t+1) > Adj Close(t)
target = 0 en caso contrario
```

Se utiliza `Adj Close` porque incorpora ajustes por dividendos y splits, lo que permite calcular una señal de evolución de precio más consistente. La columna auxiliar `next_adj_close` se usa solo para construir `target` y se elimina antes del entrenamiento para evitar fuga de información futura.

## Variables financieras

Sobre los precios diarios se calculan variables financieras y técnicas:

- Retorno diario.
- Medias móviles simples de 5, 20 y 50 sesiones.
- RSI.
- MACD y señal MACD.
- Volatilidad móvil de 20 sesiones.

Estas variables se calculan únicamente con información disponible hasta la fecha observada.

## Noticias financieras

Las noticias se descargan por ticker y rango temporal. Para cada noticia se conserva información como título, resumen, fuente, URL, fecha de publicación y puntuaciones de sentimiento proporcionadas por la fuente.

El proceso elimina duplicados por ticker, URL, título y fecha de publicación. Después, las noticias se ordenan cronológicamente para mantener trazabilidad.

## Sentimiento

El sentimiento se transforma en variables numéricas y categóricas. A nivel diario se agregan las noticias por ticker y día bursátil, generando:

- Media, mediana, mínimo y máximo de sentimiento.
- Número total de noticias.
- Número de noticias positivas, negativas y neutrales.
- Ratios de noticias positivas, negativas y neutrales.
- Indicador binario de existencia de noticias.

## Alineación con calendario bursátil

Las noticias publicadas en días no bursátiles se asignan al siguiente día de negociación del ticker correspondiente. De esta forma, la información queda disponible en la primera sesión en la que el mercado puede reaccionar.

Las noticias que no pueden alinearse con una sesión posterior dentro del periodo disponible se separan en un archivo específico de noticias no alineadas.

## Días sin noticias

Cuando una fecha bursátil no tiene noticias asociadas, las variables de sentimiento se rellenan con valores neutros:

- Puntuaciones de sentimiento igual a `0`.
- Conteos de noticias igual a `0`.
- Ratios de sentimiento igual a `0`.
- `has_news` igual a `0`.

Este tratamiento evita eliminar observaciones financieras válidas y permite entrenar modelos sobre una serie temporal diaria continua por activo.

## Dataset final

El dataset híbrido final combina:

- Identificación: `ticker`, `Date`.
- Precios y volumen.
- Indicadores técnicos.
- Variables de sentimiento agregadas.
- Variable objetivo.

El dataset final no contiene nulos relevantes, no incluye duplicados por ticker-fecha y queda preparado para la división temporal de entrenamiento y prueba.

## División entrenamiento-prueba

La división se realiza por fechas, no de forma aleatoria. Se usa el 80% inicial de fechas únicas para entrenamiento y el 20% final para prueba. Esta decisión evita que información futura entre en el entrenamiento y mantiene la coherencia temporal del experimento.
