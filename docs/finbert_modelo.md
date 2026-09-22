# FinBERT: del texto a las probabilidades

**Autor:** Marco Padilla Gómez. **Tarea:** [#49](https://github.com/maarcoopg/tfg-ml-financial-sentiment/issues/49).

Este capítulo técnico estudia un único modelo, `ProsusAI/finbert`. No mide todavía
la calidad de su sentimiento ni su utilidad bursátil. Esas preguntas pertenecen
a las issues #50 y #51. No se ha ajustado ni entrenado un modelo de lenguaje nuevo.

## Reproducir el estudio

Desde la raíz, utilizando Python 3.13 y el entorno del proyecto:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements-finbert.txt
.venv/Scripts/python.exe -m ipykernel install --user --name tfg-finanzas --display-name "Python (TFG Finanzas)"
.venv/Scripts/python.exe -m src.nlp.inspect_finbert --download
.venv/Scripts/python.exe -m unittest tests.test_finbert -v
```

La primera ejecución descarga aproximadamente 438 MB de pesos y los archivos de
configuración y tokenización. Las siguientes pueden omitir `--download`; por
defecto solo se permite la caché local. No se llama a una API de inferencia ni se
envían noticias al proveedor. Los pesos quedan en `models/pretrained/`, fuera de
Git. La salida se crea con un identificador nuevo en `reports/experiments/` y no
sobrescribe ejecuciones anteriores. El estudio usa una noticia local real;
requiere `data/processed/news/financial_news_with_sentiment.csv`.

El [notebook 07](../notebooks/07_finbert_model_understanding.ipynb) permite leer las
salidas conservadas y repetir la inspección con el modelo en caché. Funciona desde
la raíz o desde `notebooks/`, seleccionando el kernel **Python (TFG Finanzas)**. No necesita anotaciones humanas para verificar las
operaciones matemáticas, pero no puede estimar calidad lingüística sin ellas.

## Qué artefacto utilizamos

Se fija la revisión `4556d13015211d73dccd3fdd39d39232506f3e43`, no una referencia
móvil `main`. Tokenizador y pesos comparten revisión. Se utiliza la implementación
`BertForSequenceClassification` de Transformers 4.57.6 con atención explícita
`eager`, precisión de 32 bits y modo `eval()`. Este modo desactiva el dropout;
`no_grad()` evita construir el grafo de gradientes. Ninguno de ellos modifica pesos.

El repositorio publicado contiene `pytorch_model.bin`, no pesos Safetensors en la
revisión fijada. Se carga con `weights_only=True` en PyTorch 2.8.0 y
`trust_remote_code=False`; no se ejecuta código Python remoto del modelo. El
manifiesto conserva SHA-256 de pesos, configuración, vocabulario, entradas y código.
Esto identifica el artefacto; no reconstruye por sí solo toda su historia de entrenamiento.

Hugging Face proporciona almacenamiento y bibliotecas, no un certificado de
calidad del checkpoint. La ficha declara clasificación financiera en inglés.
Los identificadores de clases se leen de la configuración: 0 positivo, 1 negativo
y 2 neutral; no se presupone el orden habitual de otro modelo.

Fuentes: [checkpoint fijado](https://huggingface.co/ProsusAI/finbert/tree/4556d13015211d73dccd3fdd39d39232506f3e43),
[implementación BERT utilizada](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/bert/modeling_bert.py).

## Qué aprendió antes de nuestro proyecto

Conviene separar tres etapas: aprendizaje general de representaciones, adaptación
al vocabulario y contexto financieros, y aprendizaje supervisado de las clases de
sentimiento. El trabajo de Araci describe adaptación con TRC2-financial y ajuste
con Financial PhraseBank. No hemos reproducido esos entrenamientos ni debemos
confundir su evaluación publicada con nuestros resultados.

El BERT original aprende reconstruyendo tokens enmascarados (MLM) y discriminando
si dos segmentos son consecutivos (NSP). No es un generador causal que aprende
únicamente a predecir el siguiente token. En el ajuste de clasificación, una
etiqueta humana supervisa la salida mediante entropía cruzada:

$$\mathcal{L}=-\sum_{c=1}^{3} y_c\log p_c.$$

El artículo de FinBERT también estudia estrategias contra el olvido catastrófico.
Aquí distinguimos esas propuestas del funcionamiento del checkpoint descargado:
su configuración no acredita todos los detalles del entrenamiento que produjo
los pesos. Las fechas y el posible solapamiento del corpus deben auditarse antes
de interpretar resultados temporales; la fecha del artículo no basta.

Fuentes: [BERT, Devlin et al.](https://aclanthology.org/N19-1423/),
[FinBERT, Araci](https://arxiv.org/abs/1908.10063).

## 1. Texto y tokens

WordPiece representa el texto mediante unidades de un vocabulario aprendido.
Un token no equivale necesariamente a una palabra: nombres, decimales y palabras
poco frecuentes pueden dividirse. El prefijo `##` identifica continuaciones.
El tokenizador del checkpoint normaliza mayúsculas/minúsculas; el notebook lo
comprueba en su configuración y muestra un ejemplo, sin traducir las noticias.

Una entrada individual adopta la forma `[CLS] texto [SEP]`. `[CLS]` sirve como
posición de lectura para clasificación; no contiene inicialmente un resumen.
`[SEP]` delimita el final. En un lote, `[PAD]` iguala longitudes y la máscara marca
posiciones válidas. Suprimir claves de relleno no obliga a que las representaciones
de las consultas de relleno sean cero; sencillamente no las usamos como contenido.

FinBERT admite 512 posiciones, incluidos tokens especiales. El módulo registra
longitud original, utilizada y descartada. El ejemplo sintético de 520 repeticiones
de `profit` permite verificar el truncamiento sin simular una noticia real. No se
presenta una longitud en palabras como si fuera una longitud en tokens.

## 2. Representación de entrada

Para cada posición se suman vectores aprendidos de token, posición y segmento:

$$E_i=\operatorname{LayerNorm}(W_{token_i}+P_i+T_{segment_i}).$$

Todos tienen dimensión 768. Una entrada con $L$ tokens y un lote de $B$ textos
produce un tensor $B\times L\times768$. Al usar una única secuencia, todos los
segmentos valen cero. Estas posiciones son aprendidas y absolutas, no sinusoidales.
En entrenamiento se aplica también dropout; en nuestra inspección está desactivado.

## 3. Atención multicabeza

Cada bloque calcula proyecciones aprendidas $Q=XW_Q+b_Q$, $K=XW_K+b_K$ y
$V=XW_V+b_V$. Hay 12 cabezas de dimensión $d_k=64$. Para cada cabeza:

$$A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{64}}+M\right),\qquad C=AV.$$

Las filas representan posiciones que consultan; las columnas, posiciones
consultadas. $M$ impide atender a claves de relleno. No se aplica máscara causal:
una palabra puede incorporar contexto anterior y posterior dentro de la noticia.
Esto no supone usar noticias futuras; la causalidad bursátil se controla por
fecha y disponibilidad de los textos, fuera del Transformer.

Las cabezas no son categorías predefinidas como «beneficios» o «negaciones».
Son proyecciones aprendidas. El gráfico fija la primera capa y primera cabeza,
sin buscar una visualización favorable. Si se muestran solo 32 tokens, la imagen
es un recorte; sus filas pueden no sumar uno dentro de ese recorte.

## 4. Bloque Transformer

Se concatenan las cabezas y se proyectan al espacio de 768 dimensiones. Después:

$$H=\operatorname{LayerNorm}(X+\operatorname{Dense}(\operatorname{Concat}(C_1,\ldots,C_{12}))),$$
$$X'=\operatorname{LayerNorm}(H+W_2\operatorname{GELU}(W_1H+b_1)+b_2).$$

La red interna pasa de 768 a 3072 dimensiones y vuelve a 768. La atención mezcla
posiciones; esa red transforma cada posición con los mismos parámetros. Las
sumas residuales conservan una vía de información y la normalización estabiliza
la escala. BERT aplica normalización después de la suma residual. El proceso se
repite en 12 bloques: no son 12 modelos independientes.

El módulo reconstruye cada bloque desde la entrada registrada por la biblioteca.
Por tanto, mide discrepancia numérica local y no error acumulado al encadenar
una implementación alternativa. La biblioteca sigue produciendo las predicciones.

## 5. De [CLS] a sentimiento

Se toma el estado final de `[CLS]`, se aplica el pooler (capa lineal y tangente
hiperbólica), dropout inactivo durante inferencia y la capa lineal de tres clases:

$$h_p=\tanh(W_ph_{CLS}+b_p),\quad z=W_ch_p+b_c,\quad p_c=\frac{e^{z_c}}{\sum_j e^{z_j}}.$$

Los logits $z$ no son probabilidades. Para estabilidad, la exponenciación resta el
mayor logit. La clase elegida es el máximo de las tres probabilidades. Se conserva
toda la distribución y se calcula $s=p_{positivo}-p_{negativo}$, entre -1 y 1.
Un valor próximo a cero puede surgir por neutralidad o por reparto entre positivo
y negativo: no son situaciones equivalentes. Una probabilidad alta tampoco
garantiza calibración, corrección ni una subida bursátil.

Fuente de las operaciones y del pooler: [código BERT de Transformers 4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/bert/modeling_bert.py).

## Qué verificamos y qué no

El ejecutor contrasta embeddings, atención y salida de cada bloque, pooler, logits
y softmax. Conserva errores absolutos máximos, tolerancias y resultado de cada
comparación. Comprueba también el enmascaramiento del relleno y que una frase
produce las mismas probabilidades sola y acompañada de un titular más largo,
dentro de tolerancia numérica. Las pruebas unitarias usan un BERT diminuto
aleatorio: verifican código sin red y no se presentan como FinBERT entrenado.

La noticia real se selecciona como el primer titular no vacío de AAPL por fecha
UTC, URL y título; no por su puntuación ni por su efecto en precios. El ejemplo
`Profits increased.` es sintético y solo controla el relleno. No se etiquetan
automáticamente estos ejemplos como un conjunto de prueba humano.

La atención muestra un mecanismo interno, no una atribución causal de la decisión.
También la norma de `[CLS]` es una magnitud, no una medida de comprensión.
La [literatura sobre atención y explicación](https://aclanthology.org/N19-1357/)
motiva separar estas visualizaciones de los análisis de atribución de la issue #50.

Finalmente, el sentimiento de este checkpoint corresponde al texto presentado.
No tiene una entrada entrenada específica para la empresa objetivo. Un titular
con varias entidades necesita un tratamiento y una evaluación propios. Nada de
esta tarea demuestra todavía que mejore Alpha Vantage o los modelos financieros.
