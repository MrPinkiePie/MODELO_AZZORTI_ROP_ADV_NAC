# processed

`datos_estimar_5.csv` — dataset final a nivel referencia (CAMP_NOMBRE_SAP), con RATIO_FACTOR ya calculado. Es lo que consumen los notebooks 2-4 y el split train/test de `entrenar.py`.

`df_limpio.parquet` — train+test ya armados (columna `SPLIT`), que exporta `entrenar.py` (PASO 9) para `6.- experimentos_nube/`.

`PRUEBAS DE MODELO/` — resultados de cada corrida de `entrenar.py` (PASO 15), un csv por corrida (`df_<origen>_<destino>_<linea>_<comentario>.csv`). `consolidar_pruebas_modelo.py` los junta en `consolidado_pruebas_modelo.parquet`, que lee `2.- notebooks/4.- diagnostico_shap.py`. Los `df_historico_*.csv` son los históricos acumulados de antes de este cambio (venían de `5.- reports/excel/resultados_*.csv`).
