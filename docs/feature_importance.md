# Importancia de variables

El análisis de importancia se genera con:

```bash
python src/models/analyze_feature_importance.py
```

## Modelos analizados

Se calcula importancia para los modelos que ofrecen una señal interpretable directa:

- `logistic_regression`: valor absoluto de los coeficientes.
- `random_forest`: importancia interna del estimador.

El modelo `dummy` no tiene variables explicativas reales. `hist_gradient_boosting` no expone una importancia directa equivalente en esta implementación, por lo que no se incluye en este análisis.

## Salidas

- `reports/feature_importance/feature_importance.csv`
- `reports/feature_importance/top_10_feature_importance.csv`
- `reports/feature_importance/feature_group_importance.csv`

## Lectura por grupos

| Dataset | Modelo | Grupo | Importancia agregada |
| --- | --- | --- | ---: |
| base | logistic_regression | financial | 0.3648 |
| base | random_forest | financial | 1.0000 |
| hybrid | logistic_regression | financial | 1.7803 |
| hybrid | logistic_regression | sentiment | 0.5023 |
| hybrid | random_forest | financial | 0.7498 |
| hybrid | random_forest | sentiment | 0.2502 |

En el modelo híbrido, las variables financieras siguen concentrando la mayor parte de la importancia. Aun así, las variables de sentimiento tienen peso medible, especialmente en `random_forest`, donde representan aproximadamente una cuarta parte de la importancia agregada.

## Limitaciones

Las importancias internas no prueban causalidad. Además, los coeficientes de regresión logística y las importancias de random forest no son directamente comparables entre sí. Este análisis se usa como apoyo interpretativo, no como demostración definitiva del efecto del sentimiento.
