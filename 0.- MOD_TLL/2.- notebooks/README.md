# 2.- notebooks

Cuadernos exploratorios numerados por orden de uso. El flujo productivo vive en `3.- src/models/entrenar.py`; estos notebooks son para revisar/validar resultados, no para redefinir la logica.

1. **1.- exploracion.py** — shapes/columnas/valores unicos de las tablas de destino y origen antes de cruzarlas.
2. **2.- features.py** — inspeccion del dataset a nivel referencia (`datos_estimar_5.csv`): pivot tables, distribucion de `CAMBIO_PAG_LADO`.
3. **3.- modelado.py** — EDA de `RATIO_FACTOR` (base del target `LOG_RATIO_FACTOR`) antes de correr Optuna: distribucion, deteccion visual de atipicos, comparacion Train/Test.
4. **4.- diagnostico_shap.py** — explica con SHAP una predicción puntual del `.pkl` ya entrenado (mismo notebook que ROP_NAC_NAC/HOG_NAC_NAC; lee `0.- data/processed/PRUEBAS DE MODELO/consolidado_pruebas_modelo.parquet`).
5. **5.- diagnostico_ratio_precio.py** — distribucion y evolucion de `RATIO_PRECIO` (migrado de `analisis_ratio_precio.py`).
