# Evaluación del sentimiento de FinBERT

**Autor:** Marco Padilla Gómez. **Issue:** #50. **Estado inicial:** protocolo fijado
antes de obtener resultados; etiquetas humanas todavía no disponibles.

## Preguntas y límites

Se mantiene exclusivamente `ProsusAI/finbert` y la revisión de la issue #49.
No se ajustan sus pesos, no se prueban otros modelos de lenguaje ni se entrenan
clasificadores bursátiles. Se separan diagnóstico del comportamiento, evaluación
lingüística con referencia humana y futura utilidad financiera (#51).

## Separación de datos

Se agrupan componentes conectados por URL sin fragmento, titular normalizado o
texto completo normalizado idénticos, también entre empresas. Se conservan las
consultas de las URL porque pueden identificar artículos. La normalización usa
Unicode NFKC, minúsculas y espacios; no elimina números ni signos financieros.
Agrupar titulares idénticos a lo largo de años es una exclusión conservadora:
puede agrupar eventos diferentes. No se garantiza detectar todas las paráfrasis.

El corte es el 16 de octubre de 2023 UTC. Los grupos que cruzan esa fecha se
excluyen de la muestra, sin desplazarlos a una fecha falsa. Se seleccionan
determinísticamente 50 parejas por empresa y partición: 200 de desarrollo y 200
de evaluación, 400 en total. Se admite una pareja por grupo en la muestra; las
empresas se recorren en orden AAPL, MSFT, NVDA, TSLA. Esto no constituye una
muestra proporcional al tráfico real: se informa el desequilibrio y no se
extrapola una exactitud global a toda la población. Selección por hash, semilla
50, sin usar puntuaciones, etiquetas del proveedor ni cotizaciones futuras.

Solo se ejecuta el diagnóstico de representaciones sobre desarrollo. La
evaluación permanece reservada hasta recibir y validar anotaciones. Las fechas
bursátiles históricas ya se inspeccionaron: una reserva lingüística nueva no
convierte la futura evaluación financiera en un test independiente.

## Referencia humana

Las plantillas ocultan puntuaciones de FinBERT y Alpha Vantage. Se etiqueta el
sentimiento financiero **hacia la empresa indicada** utilizando únicamente el
titular y resumen proporcionados, sin consultar cotizaciones ni el artículo
completo. Etiquetas admitidas en el archivo:

| Valor | Criterio |
| --- | --- |
| `positive` | Información favorable a la empresa: mejora, superación de expectativas o reducción de un perjuicio. |
| `negative` | Información desfavorable: deterioro, incumplimiento de expectativas o aumento de un perjuicio. |
| `neutral` | Información suficiente sin orientación favorable o desfavorable identificable. |
| `insufficient` | Texto insuficiente, empresa sin contexto atribuible o ambigüedad que impide elegir una clase. |

Neutral no significa desconocido. Se anota también idioma (`en`, `other` o
`uncertain`), justificación breve, identificador del anotador y fecha ISO de
revisión. Una misma noticia puede ser favorable a una empresa y desfavorable
a otra. No se deduce la etiqueta por palabras aisladas ni por el retorno siguiente.

Un segundo anotador revisa independientemente 12 ejemplos por empresa y
partición: 96 de los 400, seleccionados por ID. Se calcula acuerdo y kappa antes
de resolver discrepancias; se guarda la adjudicación sin borrar los originales.
Si no se dispone de un segundo anotador, debe declararse la limitación, no
simularlo. El autor deberá confirmar la muestra y realizar u organizar la anotación.

Se copian las plantillas a una carpeta de trabajo antes de rellenarlas; no se
editan los informes inmutables. Las métricas exigen etiquetas y procedencia
completas, ID únicos, empresa intacta y partición tomada del manifiesto, no del
archivo editable. Unas etiquetas vacías no producen métricas ficticias.

## Representaciones predefinidas

1. Titular completo.
2. Titular y resumen, conservando el campo `text` existente.
3. Contexto: titular y frases del resumen con un alias explícito de la empresa.

El tercer enfoque es una regla diagnóstica, no un modelo entrenado para sentimiento
dirigido. Los alias son Apple/AAPL, Microsoft/MSFT, NVIDIA/NVDA y Tesla/TSLA.
No resuelven pronombres, subsidiarias ni homónimos; la segmentación por puntuación
puede fallar con abreviaturas. Sin coincidencias se registra ausencia y se abstiene,
sin sustituir silenciosamente por todo el texto. La comparación pareada informa
la cobertura y usa el subconjunto con contexto disponible.

La referencia humana describe titular y resumen: el titular solo dispone de
menos información. Una diferencia no demostraría superioridad intrínseca del
modelo. Tampoco conocemos si Alpha Vantage usa más contenido que nuestro resumen.

## Pruebas y atribuciones

Los pares sintéticos predefinidos estudian negación, beneficios, pérdidas,
expectativas y efectos opuestos sobre empresas. Sus expectativas de ordenación
son hipótesis de diagnóstico, no etiquetas humanas independientes. Se conservan
también los casos que contradicen la expectativa.

Se utiliza Integrated Gradients de Captum sobre embeddings de palabras y el
logit de la clase elegida en la entrada original. Se mantienen fijos posición,
segmentos, máscara y tokens especiales. Se comparan referencias PAD y MASK para
los tokens de contenido, con todos esos lugares aún visibles a la atención.
Son referencias artificiales, no noticias neutrales. Integración Gauss-Legendre
con 32, 64, 128 y hasta 256 pasos; tolerancia de completitud
`max(0.005, 0.01 * abs(logit_entrada - logit_referencia))`.

Las atribuciones no convergentes se marcan, no se descartan ni se presentan como
explicaciones fiables. Se enmascaran tokens de mayor contribución positiva y se
contrastan con igual número de tokens de menor magnitud y diez controles
aleatorios fijos. El cambio de logit es un diagnóstico de sensibilidad, no
causalidad económica; el enmascaramiento puede producir entradas artificiales.

## Métricas y estado pendiente

Tras la anotación: macro-F1 de tres clases, precisión y sensibilidad por clase,
matriz de confusión, cobertura de abstención y revisión de errores por empresa.
`insufficient` e idiomas distintos o inciertos se excluyen de las métricas de
tres clases y se cuentan por separado. Se informa soporte, no solo promedios.
El acuerdo con Alpha Vantage es desacuerdo entre sistemas, no exactitud.

Fuentes: [Captum Integrated Gradients](https://captum.ai/api/integrated_gradients.html),
[tutorial de interpretación de BERT](https://captum.ai/tutorials/Bert_SQUAD_Interpret),
[artículo de gradientes integrados](https://proceedings.mlr.press/v70/sundararajan17a.html).
La revisión publicada del checkpoint es de mayo de 2023, anterior al corte,
según el [historial del modelo](https://huggingface.co/ProsusAI/finbert/commits/main).
Esto no identifica todos los documentos de preentrenamiento ni descarta
solapamiento con las noticias antiguas de desarrollo.
