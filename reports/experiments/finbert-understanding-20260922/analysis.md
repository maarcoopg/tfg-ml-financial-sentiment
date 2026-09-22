# Estudio interno de FinBERT

**Issue #49. Autor: Marco Padilla Gómez.** Esta ejecución no evalúa predicción bursátil.

Se utiliza `ProsusAI/finbert` en la revisión
`4556d13015211d73dccd3fdd39d39232506f3e43`, con Transformers 4.57.6 y PyTorch
2.8.0+cpu. No se actualizan pesos ni se comparan otros modelos de lenguaje.

## Evidencias

- 109.484.547 parámetros contabilizados por componente en `parameters.csv`.
- 28 reconstrucciones contrastadas en `reconstruction.csv`; todas pasan.
- Error absoluto máximo: 3,725290298461914 × 10⁻⁹. Tolerancia absoluta: 2 × 10⁻⁵;
  relativa: 10⁻⁵. Las coincidencias exactas en CPU no garantizan igualdad binaria
  en otro dispositivo o versión.
- Se verifican embeddings, atención y salida de cada bloque, pooler, logits y
  softmax. Cada bloque usa la entrada registrada por la biblioteca para medir
  error local; no se encadenan errores de una implementación alternativa.
- Relleno enmascarado e invariancia de la predicción al cambiar el lote verificados.
- Control artificial largo: 522 tokens originales, 512 utilizados, 10 descartados.

El primer titular no vacío de AAPL por fecha UTC, URL y título determina el
ejemplo real. Tiene 11 tokens incluidos los especiales. No se ha seleccionado
por su predicción; menciona varias empresas, ilustrando además la limitación de
un clasificador global. La frase sintética corta se utiliza como control técnico.
Las probabilidades en `predictions.json` no son etiquetas humanas ni probabilidades
de subida. `attention.png` muestra una cabeza predefinida, no atribuciones causales.

## Reproducibilidad

`manifest.json` conserva revisión, versiones, dispositivo, huellas de datos,
código, pesos y salidas. El código corresponde al commit indicado en ese manifiesto.
Las dependencias son opcionales y los pesos no se suben a Git. Para repetir:

```powershell
python -m src.nlp.inspect_finbert
```

Esto crea otra ejecución; se requiere la caché local. Solo en la primera descarga
se añade `--download`. El [cuaderno 07](../../../notebooks/07_finbert_model_understanding.ipynb)
desarrolla el recorrido y conserva sus salidas.

## Conclusión

El estudio documenta y verifica el funcionamiento del checkpoint. No demuestra
calidad lingüística, superioridad frente a Alpha Vantage, mejora de AUC ni
rentabilidad. La anotación, el análisis de errores, las atribuciones y la
integración temporal pertenecen a las issues #50 y #51 y no se han ejecutado aquí.
