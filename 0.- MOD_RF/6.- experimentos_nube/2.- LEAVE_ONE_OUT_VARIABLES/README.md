# LEAVE_ONE_OUT_VARIABLES

Mide el impacto individual de cada variable en el modelo ROP_ADV_NAC: entrena el combo completo (todas las variables) y luego, una a la vez, un combo sin esa variable (N+1 corridas en vez de las 2^n-1 de `RF NUBE.py`), comparando SVS/SBR contra el combo completo.

## De dónde toma la data

`RF NUBE QUITANDO X.py` lee `0.- data/processed/df_limpio.parquet` (`../../0.- data/processed/df_limpio.parquet` desde esta carpeta) — el mismo archivo que exporta el PASO 9 de `3.- src/models/entrenar.py`. Si no existe, correr `entrenar.py` primero.

## Cómo usarlo

1. Ajustar los parámetros al inicio del script: `TOP_K_FEATURES` (cuántas variables entran al leave-one-out), `N_ESTIMATORS_BUSQUEDA` (árboles del RF solo durante la búsqueda, más liviano que el modelo final) y `comentario`.
2. El script primero preselecciona variables por Mutual Information/correlación de Spearman (si `USAR_TOP_K_FEATURES=True`), arma los combos leave-one-out y corre cada uno en paralelo (`joblib.Parallel`).
3. Revisar `tabla_resumen.xlsx`: la fila `ALL` es el combo completo (referencia); en cada fila `VARIABLE`, `PERDIDA_SRV` es cuánto baja el nivel de servicio al quitar esa variable y `PERD_SOBR` cuánto sube el sobrante — mientras más alto `PERDIDA_SRV`, más importante es la variable para el modelo.

## Diferencia respecto a MODELO_AZZORTI_ROP_NAC_NAC

Se agrupa por `CAMP` en vez de `CAMP_DESTINO` (ADV_NAC no tiene esa columna), y las `numeric_features`/`categorical_features` salen de la lista real de `entrenar.py` de este modelo — ver el README de `6.- experimentos_nube/` un nivel arriba para el detalle completo.

## Archivos

| Archivo | Qué es |
|---|---|
| `RF NUBE QUITANDO X.py` | Corre el leave-one-out (combo completo + un combo por variable quitada) |
| `df_leave_one_out.csv` | Detalle de cada combo por percentil (SRV/SBR, MAE/MAPE train y test) |
| `tabla_resumen.xlsx` | Resumen SRV/SBR (percentil Q_50) por variable quitada, con la pérdida respecto al combo completo (`ALL`) |
