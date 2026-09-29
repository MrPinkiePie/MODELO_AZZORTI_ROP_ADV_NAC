# NUBE_COMBINACIONES_RF

Búsqueda combinatoria de variables para el modelo Random Forest de ROP_ADV_NAC. Cada combinación de variables se entrena y se mide su SVS/SBR, generando una "nube" de puntos para elegir el mejor set de variables. Adaptado del mismo esquema de `MODELO_AZZORTI_ROP_NAC_NAC/6.- experimentos_nube/`, con la lista de features real de este modelo.

## De dónde toma la data

`RF NUBE.py` lee `0.- data/processed/df_limpio.parquet` (`../0.- data/processed/df_limpio.parquet` desde esta carpeta) — el mismo archivo que exporta el PASO 9 de `3.- src/models/entrenar.py` (columna `SPLIT`, alias de `CLASIF`). Si no existe todavía, hay que correr `entrenar.py` primero.

## Cómo usarlo (orden de ejecución)

1. **`RF NUBE.py`** — corre TODAS las combinaciones posibles de variables (2^n - 1, sobre las top-`TOP_K_FEATURES` por Mutual Information si `USAR_TOP_K_FEATURES=True`). Exporta un `.txt` por corrida, nombrado con la variable `comentario` (ej. `prueba_1.txt`).
2. **`CONCAENADOR _DE_TXT.py`** — concatena todos los `.txt` generados por el paso 1 en `DATA.csv` / `DATA.parquet`, y agrega columnas `has_<variable>` para saber qué combinaciones incluyen cada variable.
3. **`DASH NUBES RF.pbix`** — dashboard de Power BI que lee `DATA.csv`/`DATA.parquet` para visualizar la nube de puntos (SVS vs SBR) y filtrar por variable. Es una copia del `.pbix` de `MODELO_AZZORTI_ROP_NAC_NAC`, así que **antes de abrirlo hay que actualizar el origen de datos** (Inicio → Transformar datos → Configuración del origen de datos) para que apunte a esta carpeta (`MODELO_AZZORTI_ROP_ADV_NAC/0.- MOD_RF/6.- experimentos_nube/DATA.csv` o `.parquet`), no a la de NAC_NAC.
4. **`2.- LEAVE_ONE_OUT_VARIABLES/RF NUBE QUITANDO X.py`** — en vez de todas las combinaciones, corre el modelo completo y luego quita una variable a la vez, para medir el impacto individual de cada una. Exporta `df_leave_one_out.csv` y `tabla_resumen.xlsx` (resumen SRV/SBR por variable quitada) en esa misma subcarpeta.

## Diferencias respecto a MODELO_AZZORTI_ROP_NAC_NAC

- Se agrupa por `CAMP` en vez de `CAMP_DESTINO` (ADV_NAC no tiene esa columna; la campaña destino ya es `CAMP`).
- La lista de `numeric_features`/`categorical_features` sale de `features` en `entrenar.py`, sin los factores regionales crudos (`FACT_BOG`/`FACT_LIMA`/etc., que cambian según `PAIS_ORIGEN` COL/PER) — esos ya quedan resumidos en `FACTOR_MAX`/`FACTOR_MIN`/`FACTOR_MEDIANA`. Si se quieren probar los crudos, agregarlos a mano según el país con el que se corrió `entrenar.py`.
- `entrenar.py` de ADV_NAC no exportaba `df_limpio.parquet` originalmente (solo lo hace NAC_NAC); se agregó esa exportación en el PASO 9 para que esta carpeta tenga de dónde leer.

## Archivos

| Archivo | Qué es |
|---|---|
| `RF NUBE.py` | Búsqueda combinatoria completa de variables |
| `CONCAENADOR _DE_TXT.py` | Junta los `.txt` de las corridas en `DATA.csv`/`DATA.parquet` |
| `DATA.csv` / `DATA.parquet` | Resultado consolidado de todas las corridas (insumo del dashboard) |
| `prueba_*.txt` | Resultado crudo de una corrida individual de `RF NUBE.py` |
| `DASH NUBES RF.pbix` | Dashboard de Power BI para explorar la nube de resultados |
| `2.- LEAVE_ONE_OUT_VARIABLES/` | Análisis leave-one-out: impacto de quitar cada variable individualmente |
