# Resultados de la revisión temporal

## Ejecución y alcance

Ejecución completa: `reports/experiments/review-full-20260908`. Estado del manifiesto: `complete`. Se usaron doce variantes de variables, cinco algoritmos, tres folds externos y tres internos. Hubo 1.080 entrenamientos internos y 183 externos, contando tres baselines. Cada comparación contiene las mismas 2.212 observaciones externas, de AAPL, MSFT, NVDA y TSLA. SPY no participa.

Se corrigieron horarios de noticias, purga temporal y muestreo de LightGBM. También se cambió el protocolo a selección anidada con una búsqueda pequeña predefinida. Por tanto, la diferencia frente al antiguo AUC 0,5217 no puede atribuirse exclusivamente a una de esas correcciones. El nuevo experimento no reproduce exactamente las 65 variables de la issue #45: compara retardos y ventanas por separado.

## Resultados principales

El mayor AUC agrupado observado es 0,5108, con HistGradientBoosting y variables financieras relativas. Su balanced accuracy es 0,5024 y su MCC 0,0052. Estos valores no muestran una capacidad discriminativa robusta.

| Variante | Algoritmo | Accuracy | Balanced accuracy | AUC agrupado | Media AUC de folds externos |
| --- | --- | ---: | ---: | ---: | ---: |
| Base | Dummy | 0,5389 | 0,5000 | 0,5000 | 0,5000 |
| Base | HistGradientBoosting | 0,5000 | 0,4928 | 0,4868 | 0,4928 |
| Híbrido | HistGradientBoosting | 0,4932 | 0,4876 | 0,4932 | 0,5011 |
| Variables relativas | HistGradientBoosting | 0,5163 | 0,5024 | 0,5108 | 0,5099 |
| Base | LightGBM | 0,5045 | 0,4965 | 0,4920 | 0,4981 |
| Híbrido | LightGBM | 0,5041 | 0,5016 | 0,4954 | 0,5004 |
| Retardo de una sesión | LightGBM | 0,4959 | 0,4938 | 0,4946 | 0,5005 |
| Base | Random Forest | 0,5140 | 0,4976 | 0,5049 | 0,5146 |
| Híbrido | Random Forest | 0,5081 | 0,5034 | 0,5018 | 0,5110 |
| Sentimiento ponderado | Random Forest | 0,5140 | 0,5051 | 0,5075 | 0,5181 |

El AUC agrupado combina las predicciones externas de todos los bloques. La media de AUC calcula primero cada bloque y después promedia. Los modelos y las distribuciones pueden cambiar entre bloques, por lo que ambos resúmenes no son equivalentes. El mayor promedio externo corresponde a Random Forest con sentimiento ponderado, 0,5181, y no al ganador por AUC agrupado. Ambos se identifican descriptivamente; seleccionar el mayor entre muchas alternativas no demuestra generalización.

## Incertidumbre

Intervalos percentiles exploratorios del 95 %, calculados con 1.000 réplicas pareadas, cuatro empresas juntas por fecha y bloques de 20 sesiones:

| Comparación | Diferencia AUC agrupado | Intervalo de la diferencia |
| --- | ---: | --- |
| HistGradientBoosting relativo frente a híbrido | +0,0176 | [-0,0096; 0,0477] |
| Random Forest ponderado frente a híbrido | +0,0056 | [-0,0041; 0,0156] |
| LightGBM retardo 1 frente a híbrido | -0,0008 | [-0,0193; 0,0132] |

El intervalo del AUC de HistGradientBoosting relativo es [0,4855; 0,5406]. La comparación principal predefinida, LightGBM con retardo 1 frente a su híbrido, tampoco excluye cero al usar bloques de 5 o 60 sesiones.

De los 62 intervalos de diferencias calculados, 61 incluyen cero. El único que no lo incluye indica un empeoramiento de LightGBM con solo sentimiento frente al baseline: [-0,0520; -0,0017]. No hay ninguna mejora positiva cuyo intervalo excluya cero en este análisis. No se aplica corrección por múltiples comparaciones y los intervalos no incorporan toda la incertidumbre de reentrenar y seleccionar modelos.

## Fronteras temporales verificadas

| Fold | Última fecha train | Fin máximo de etiqueta train | Inicio externo | Final externo | Filas train | Filas externas |
| --- | --- | --- | --- | --- | ---: | ---: |
| 1 | 2023-10-12 | 2023-10-13 | 2023-10-16 | 2024-07-11 | 8.840 | 740 |
| 2 | 2024-07-10 | 2024-07-11 | 2024-07-12 | 2025-04-04 | 9.580 | 736 |
| 3 | 2025-04-03 | 2025-04-04 | 2025-04-07 | 2025-12-29 | 10.316 | 736 |

En todos los folds internos y externos el final de las etiquetas de entrenamiento es estrictamente anterior al periodo que se evalúa. Los bloques externos anteriores pasan a formar parte del entrenamiento de los siguientes, de acuerdo con el diseño expansivo.

## Calidad de noticias

La asignación horaria contiene 46.014 registros alineados dentro del calendario generado; los registros posteriores al último cierre se guardan aparte. De ellos, 17.308 se asignan a una fecha posterior a su fecha local de publicación, incluyendo fines de semana y publicaciones posteriores al cierre. Esta cifra no representa exclusivamente noticias mal asignadas en el experimento antiguo.

No hay duplicados exactos bajo la clave ticker, URL, título y timestamp. Se detectan 3.124 repeticiones de título dentro de la misma empresa y sesión; podrían incluir actualizaciones y no se han eliminado automáticamente.

El dataset modelado registra 3.675 noticias en 2024 y 27.336 en 2025. La cobertura sigue sin estar verificada por el proveedor. La nueva asignación cambia ligeramente los recuentos por año y el tratamiento del extremo final del histórico.

## Verificación técnica

- 22 pruebas automáticas locales superadas: calendario, cierres anticipados, UTC y horario de verano, purga, retardos, descargas vacías y errores, configuración y evaluación pareada.
- Los 183 modelos guardados reproducen sus probabilidades y clases externas. No se encontraron discrepancias al comparar con las predicciones almacenadas.
- Verificadas las fronteras de los 1.080 ajustes internos y de los 183 entrenamientos externos.
- Todos los grupos de variante y algoritmo tienen 2.212 predicciones externas.
- Verificados los hashes de todos los archivos registrados en el manifiesto de la ejecución completa.
- Ejecución reducida independiente con instantánea de código verificada: `artifacts/verification/review-verification-20260908`.
- Compatibilidad de entrenamiento, evaluación y comparación tradicional comprobada; tuning reducido verificado en `artifacts/verification/tuning-verification-20260908`.
- Figuras revisadas visualmente. La configuración de CI está preparada, pero no se ha ejecutado en GitHub porque la publicación permanece bloqueada.

El primer ensayo completo conserva hashes del código de inicio, pero comenzó antes de añadir las copias de fuente por ejecución. Esa mejora se verificó en el ensayo reducido posterior. Véase `docs/revision_implementation.md`.

## Conclusión

Las correcciones mejoran la consistencia temporal, la trazabilidad y la evaluación, pero no han producido una mejora predictiva concluyente en este histórico. El resultado científicamente defendible es que, con estas fuentes, variables, horizonte y protocolo, no se demuestra una ventaja robusta del sentimiento para predecir la dirección de la siguiente sesión.

Eso no prueba que las noticias carezcan de efecto sobre los precios, ni que cualquier modelo de sentimiento vaya a fallar. Sí limita lo que puede afirmarse a partir de este proyecto. El histórico ya se había consultado; una confirmación independiente requerirá datos no utilizados y decisiones experimentales fijadas de antemano. No se ha evaluado rentabilidad económica.
