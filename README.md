# MODELO_AZZORTI_ROP_ADV_NAC

Pronostico de demanda de Ropa (ROP) en Ecuador (NAC), usando como predictor la
demanda Advance de Colombia (ADV) desfasada 3 campanas: la logica es que lo
que se vende en Colombia hoy "anticipa" lo que se va a vender en Ecuador
3 campanas despues.

Reorganizado al formato cookiecutter de `MODELO_AZZORTI_HOG_NAC_NAC` /
`MODELO_AZZORTI_ROP_NAC_NAC` (ver `0.- MOD_RF/README.md` para el detalle de
cada carpeta). El codigo original vivia en 4 archivos sueltos sin estructura
(`0.- MOD_0.py`, `analisis_ratio_precio.py`, `mis_funciones.py`, y un
notebook identico a `0.- MOD_0.py` que se elimino por ser duplicado).

## Estado del modelo

- `3.- src/models/entrenar.py` ya llega de punta a punta: carga, limpieza,
  features, Optuna (mismo enfoque simple que MODELO_AZZORTI_ROP_NAC_NAC, con
  Boruta opcional via `usar_boruta`), fit del pipeline final y guardado en
  `4.- models/`, percentiles/metricas (SVS/SBR) y export a
  `0.- data/processed/PRUEBAS DE MODELO/`.
  Validado corriendo con datos reales (ver `pipeline.py`/`evaluacion.py`).
  Falta todavia `predecir.py` (aplicar el `.pkl` guardado sobre campanas
  nuevas, no solo sobre el split train/test de la corrida de entrenamiento).
- El pronostico de `0.- MOD_RF` queda a **nivel de referencia** (codigo SAP
  del producto, `CAMP_NOMBRE_SAP`), agregando todas las tallas (`CODI_VENT`)
  del producto en una sola fila. El reparto **por talla** vive en
  `0.- MOD_TLL/`, que carga la base por talla (`interim/dat.parquet`) y el
  pronostico de MOD_RF, sin rehacer la limpieza. Ver `COMO_USAR_PLANTILLA.md`.
- Paises, desfase, campañas, corte y comentario de la corrida se editan solo
  en `config_proyecto.py` (raiz), compartido por MOD_RF y MOD_TLL.
- Hay un par de rutas de OneDrive con inconsistencias que se dejaron marcadas
  con `# TODO:` en `3.- src/config.py` (la ruta de CON_PRD no coincide con la
  de MODELO_AZZORTI_ROP_NAC_NAC, y el nombre de una carpeta intermedia
  cambiaba entre donde el script original guardaba y donde otro script leia)
  en vez de asumir cual es la correcta.
