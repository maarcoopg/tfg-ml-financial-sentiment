# Evaluación de modelos independientes por empresa

**Autor:** Marco Padilla Gómez

**Fecha:** 16 de septiembre de 2026

**Issue:** [#47](https://github.com/maarcoopg/tfg-ml-financial-sentiment/issues/47)

**Rama:** `feature/per-company-temporal-evaluation`

**Ejecución:** `per-company-20260916`

## Resumen

Separar las empresas **no produce una mejora general** en los AUC observados. AAPL empeora, MSFT obtiene pequeñas mejoras respecto al enfoque conjunto, y NVDA y TSLA presentan resultados mixtos según variante y algoritmo. Añadir noticias o su retardo tampoco ofrece una mejora uniforme.

Se conserva el experimento como una comparación adicional, no como sustituto automático del enfoque conjunto. No se han modificado `reports/final/`, los experimentos anteriores ni las conclusiones de la memoria. La rama permanece sin fusionar a `main`.

## Diseño de la comparación

- Universo: AAPL, MSFT, NVDA y TSLA; SPY continúa excluido.
- 2.764 observaciones por empresa, 11.056 en el panel. Evaluación externa: 553 sesiones por empresa, del 16/10/2023 al 29/12/2025.
- Tres bloques externos de 185, 184 y 184 sesiones. Entrenamientos individuales de 2.210, 2.395 y 2.579 observaciones, respectivamente, después de purgar.
- Tres divisiones temporales internas crecientes y selección por AUC medio, con dos configuraciones por algoritmo. Se conserva la misma rejilla que en la revisión corregida.
- Cinco algoritmos: regresión logística, bosque aleatorio, potenciación por histogramas, XGBoost y LightGBM. Referencia adicional de clase mayoritaria, aprendida separadamente en cada empresa y bloque.
- Base: 14 variables financieras. Híbrido: 27 variables. Híbrido con retardo: 29 variables, incluyendo sentimiento medio y número de noticias de la sesión anterior de la misma empresa. No es el antiguo diseño de 92 variables.
- Imputación, escalado, selección de parámetros y entrenamiento se ajustan únicamente con los datos pasados de la empresa correspondiente. El umbral permanece en 0,5.
- La referencia conjunta se ha **reentrenado en esta ejecución**, con el mismo código, variables y fechas. Sus 35.392 probabilidades coinciden exactamente con las de las tres variantes correspondientes de `review-full-20260908`.
- Ambos enfoques se comparan sobre las mismas claves empresa-fecha, etiquetas y bloques, sin cambiar el universo ni seleccionar fechas favorables.

La separación reduce a una cuarta parte los ejemplos disponibles para cada estimador. Permite especialización, pero también reduce la cantidad de información para ajustar los parámetros. Estos efectos se estudian conjuntamente; el experimento no identifica por sí solo cuál explica cada resultado.

## Separación de noticias

Las noticias se filtran por empresa antes de derivar retardos y de entrenar. El sentimiento procede de la puntuación específica del activo de Alpha Vantage. Si un mismo artículo menciona dos empresas, puede aparecer en ambas con sus propias puntuaciones; no se usa una puntuación global idéntica ni se exige exclusividad artificial.

Se guardan copias locales separadas bajo `data/experiments/per-company-20260916/<empresa>/`. Los registros alineados son 8.935 para AAPL, 15.295 para MSFT, 14.808 para NVDA y 6.976 para TSLA. Estos recuentos incluyen todos los registros alineados al calendario disponible, no solo los utilizados en las sesiones externas. La cobertura por año y empresa está en `coverage/`.

Se mantiene la alineación al siguiente cierre disponible, incluidos festivos y cierres anticipados. La publicación es una aproximación a disponibilidad real. La cobertura no está certificada: un día sin registros no prueba que no existieran noticias.

## Comparación descriptiva por empresa

La tabla siguiente es la **media de AUC de los cinco algoritmos** en cada empresa. No representa un sexto modelo ni una combinación de predicciones. Evita seleccionar un ganador distinto para cada celda.

| Empresa | Conjunto base | Individual base | Conjunto híbrido | Individual híbrido | Conjunto con retardo | Individual con retardo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AAPL | 0.4747 | 0.4498 | 0.4799 | 0.4489 | 0.4840 | 0.4591 |
| MSFT | 0.4965 | 0.5033 | 0.4934 | 0.5021 | 0.4870 | 0.4997 |
| NVDA | 0.4923 | 0.5074 | 0.5013 | 0.5120 | 0.5136 | 0.5135 |
| TSLA | 0.5153 | 0.5117 | 0.5197 | 0.5227 | 0.5099 | 0.5153 |

Promediando también las cuatro empresas con igual peso:

| Variante | AUC conjunto medio | AUC individual medio | Diferencia individual menos conjunto |
| --- | ---: | ---: | ---: |
| Base | 0.4947 | 0.4930 | -0.0017 |
| Híbrido | 0.4986 | 0.4964 | -0.0021 |
| Híbrido con retardo | 0.4986 | 0.4969 | -0.0017 |

Estos promedios son descriptivos, no contrastes estadísticos agregados. No deben compararse directamente con el AUC global del panel de los informes anteriores: aquí se promedian AUC calculados dentro de cada empresa.

## Todos los algoritmos

AUC calculados sobre las 553 sesiones externas de cada empresa. Los valores de cada pareja corresponden al mismo algoritmo, no a máximos seleccionados entre algoritmos.

### AAPL

| Algoritmo | Conjunto base | Individual base | Conjunto híbrido | Individual híbrido | Conjunto con retardo | Individual con retardo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regresión logística | 0.4495 | 0.4270 | 0.4688 | 0.4458 | 0.4694 | 0.4536 |
| Bosque aleatorio | 0.5292 | 0.4413 | 0.4926 | 0.4414 | 0.4947 | 0.4411 |
| Potenciación por histogramas | 0.4619 | 0.4512 | 0.4707 | 0.4709 | 0.4851 | 0.4702 |
| XGBoost | 0.4586 | 0.4663 | 0.4802 | 0.4507 | 0.4883 | 0.4639 |
| LightGBM | 0.4746 | 0.4632 | 0.4873 | 0.4355 | 0.4825 | 0.4668 |

### MSFT

| Algoritmo | Conjunto base | Individual base | Conjunto híbrido | Individual híbrido | Conjunto con retardo | Individual con retardo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regresión logística | 0.5220 | 0.4945 | 0.5039 | 0.4939 | 0.4949 | 0.4946 |
| Bosque aleatorio | 0.4889 | 0.5058 | 0.4852 | 0.5069 | 0.4865 | 0.5024 |
| Potenciación por histogramas | 0.4891 | 0.5141 | 0.4883 | 0.5123 | 0.4757 | 0.4996 |
| XGBoost | 0.5048 | 0.4923 | 0.5034 | 0.4948 | 0.4976 | 0.4941 |
| LightGBM | 0.4775 | 0.5097 | 0.4862 | 0.5028 | 0.4803 | 0.5075 |

### NVDA

| Algoritmo | Conjunto base | Individual base | Conjunto híbrido | Individual híbrido | Conjunto con retardo | Individual con retardo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regresión logística | 0.5180 | 0.4879 | 0.5189 | 0.5110 | 0.5192 | 0.5067 |
| Bosque aleatorio | 0.4875 | 0.4970 | 0.5180 | 0.5026 | 0.5309 | 0.5012 |
| Potenciación por histogramas | 0.4742 | 0.5224 | 0.4837 | 0.5048 | 0.4956 | 0.5193 |
| XGBoost | 0.4899 | 0.5190 | 0.4920 | 0.5304 | 0.5035 | 0.5242 |
| LightGBM | 0.4921 | 0.5106 | 0.4940 | 0.5112 | 0.5188 | 0.5158 |

### TSLA

| Algoritmo | Conjunto base | Individual base | Conjunto híbrido | Individual híbrido | Conjunto con retardo | Individual con retardo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regresión logística | 0.5167 | 0.5260 | 0.5033 | 0.5227 | 0.5008 | 0.5324 |
| Bosque aleatorio | 0.5173 | 0.5097 | 0.5256 | 0.5267 | 0.5242 | 0.5196 |
| Potenciación por histogramas | 0.5133 | 0.4984 | 0.5418 | 0.5118 | 0.5173 | 0.4949 |
| XGBoost | 0.5139 | 0.5193 | 0.5198 | 0.5304 | 0.5089 | 0.5088 |
| LightGBM | 0.5155 | 0.5052 | 0.5078 | 0.5221 | 0.4984 | 0.5210 |

## AUC y aciertos no son lo mismo

Los máximos individuales siguientes se identifican **después de observar los resultados externos**. Sirven para describir la ejecución, no para elegir un modelo validado ni estimar su rendimiento futuro.

| Empresa | Máximo descriptivo | Variante | AUC | Exactitud | Exactitud equilibrada | MCC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| AAPL | Potenciación por histogramas | Híbrido | 0.4709 | 46.84 % | 46.48 % | -0.0703 |
| MSFT | Potenciación por histogramas | Base | 0.5141 | 47.74 % | 49.63 % | -0.0079 |
| NVDA | XGBoost | Híbrido | 0.5304 | 50.99 % | 52.45 % | 0.0509 |
| TSLA | Regresión logística | Híbrido con retardo | 0.5324 | 51.18 % | 50.76 % | 0.0162 |

La referencia mayoritaria obtiene AUC 0,5 y exactitud equilibrada 0,5. Su exactitud es:

- AAPL: 54.43 %.
- MSFT: 54.97 %.
- NVDA: 54.97 %.
- TSLA: 51.18 %.

Ninguno de los máximos de AUC seleccionados en la tabla supera en exactitud a su referencia mayoritaria. Esto no invalida por sí solo una mejora de discriminación, pero muestra por qué un AUC ligeramente superior a 0,5 no equivale a muchos más aciertos.

El cuaderno también muestra el AUC de cada bloque y su media. Esta media no coincide necesariamente con el AUC de todas las sesiones: al cambiar de estimador entre bloques también puede cambiar la calibración de sus probabilidades. No se mezclan probabilidades de distintas empresas para presentar un único AUC individual.

## Incertidumbre de las diferencias

Se calculan intervalos percentiles del 95 % con 1.000 réplicas pareadas de bloques temporales de 20 sesiones. El remuestreo conserva las mismas fechas en los dos enfoques; no vuelve a entrenar los estimadores ni corrige por comparaciones múltiples.

De las 60 comparaciones individual menos conjunto, **28 tienen una diferencia observada positiva**. Tres intervalos quedan por encima de cero, todos en MSFT, y otros tres por debajo, todos en AAPL. Los otros 54 contienen cero.

| Empresa | Diferencias positivas observadas | Intervalos totalmente positivos | Intervalos totalmente negativos |
| --- | ---: | ---: | ---: |
| AAPL | 2/15 | 0 | 3 |
| MSFT | 9/15 | 3 | 0 |
| NVDA | 9/15 | 0 | 0 |
| TSLA | 8/15 | 0 | 0 |

Los seis intervalos que no contienen cero son:

| Empresa | Variante | Algoritmo | Diferencia de AUC | Límite inferior | Límite superior |
| --- | --- | --- | ---: | ---: | ---: |
| AAPL | Base | Bosque aleatorio | -0.087875 | -0.137764 | -0.041126 |
| AAPL | Híbrido | LightGBM | -0.051759 | -0.105581 | -0.000434 |
| AAPL | Híbrido con retardo | Bosque aleatorio | -0.053565 | -0.105339 | -0.009989 |
| MSFT | Base | LightGBM | 0.032208 | 0.001601 | 0.070623 |
| MSFT | Base | Bosque aleatorio | 0.016897 | 0.000663 | 0.039838 |
| MSFT | Híbrido | Bosque aleatorio | 0.021639 | 0.000024 | 0.051648 |

Las mejoras relativas de MSFT no demuestran capacidad predictiva absoluta: sus AUC individuales en esas tres comparaciones son aproximadamente 0,5058–0,5097 y sus intervalos de AUC incluyen 0,5. Además, el límite inferior del bosque aleatorio híbrido está muy cerca de cero (0,000024). Con 60 comparaciones no ajustadas pueden aparecer resultados nominalmente favorables por azar.

Dentro del enfoque individual se realizan otras **40 comparaciones**: híbrido menos base y retardo menos híbrido, para cada empresa y algoritmo. Ningún intervalo queda totalmente por encima de cero. Solo el retardo de XGBoost en TSLA queda totalmente por debajo: diferencia −0,021561, intervalo [−0,042256; −0,003479].

Los máximos descriptivos de NVDA y TSLA tampoco quedan claramente por encima de 0,5 de AUC con este procedimiento: XGBoost híbrido en NVDA tiene intervalo [0,492158; 0,576955] y regresión logística con retardo en TSLA [0,499562; 0,569233]. Todos estos intervalos son exploratorios, condicionados a las predicciones ya obtenidas.

## Interpretación y siguientes pasos

1. **AAPL:** los 15 modelos individuales quedan por debajo de 0,5 de AUC. Separar el activo no resuelve el problema y empeora la media de cada variante. Invertir las probabilidades después de observar este resultado sería otra decisión retrospectiva, no evidencia de una estrategia validada.
2. **MSFT:** pequeñas mejoras respecto al conjunto, pero discriminación agregada próxima a 0,5 y resultados basados en clases poco convincentes con el umbral fijo.
3. **NVDA:** hay señales descriptivas algo mejores, con XGBoost híbrido en 0,5304 de AUC. No constituyen por sí solas evidencia confirmatoria.
4. **TSLA:** algunos híbridos mejoran ligeramente; la mejor cifra descriptiva es 0,5324 con regresión logística y retardo. El efecto del retardo depende del algoritmo y no mejora la media frente al híbrido sin retardo.
5. **Decisión:** conservar este enfoque como experimento de contraste para la memoria, no sustituir automáticamente el modelo conjunto ni presentar una mejora general.

Como hipótesis futuras, sería más informativo estudiar un enfoque conjunto que identifique la empresa o permita interacciones por activo, y extender la variante de variables financieras relativas ya existente a la comparación individual/conjunta, sin descartar la información compartida. También conviene estudiar periodos con cobertura de noticias comparable. Cualquier elección derivada de esta exploración necesitará validarse en nuevas fechas no utilizadas para diseñarla.

## Reproducción

```powershell
python -m src.experiments.run_per_company --run-dir reports/experiments/per-company-NUEVA-EJECUCION
python -m unittest discover -s tests -v
```

El identificador debe ser nuevo. La ejecución reutiliza las entradas guardadas y no realiza descargas ni consume llamadas de pago. Los modelos `.joblib`, datos de ejecución y copias de código se mantienen fuera de Git.

El [notebook independiente](../../../notebooks/03_per_company_models.ipynb) contiene tablas, figuras, hiperparámetros, cobertura y comprobaciones reproducibles. Los resultados anteriores permanecen intactos.

## Verificación técnica

- 32 pruebas automatizadas superadas, incluidas separación por empresa, noticias compartidas con puntuaciones distintas, retardos y fronteras temporales.
- 240 modelos guardados recargados: sus probabilidades reproducen las publicadas con tolerancia absoluta de 1e-12.
- 1.350 evaluaciones internas revisadas y 35.392 predicciones por enfoque, con las mismas claves y etiquetas.
- Informes protegidos contra conversiones automáticas de finales de línea mediante `.gitattributes`, para conservar sus hashes entre sistemas operativos.
