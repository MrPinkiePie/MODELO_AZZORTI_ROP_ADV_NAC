# MODELO_AZZORTI_ROP_ADV_NAC — 0.- MOD_RF (desarrollo)

Pronostico de demanda de Ropa (ROP) en Ecuador (NAC), usando como predictor
la demanda Advance de Colombia (ADV) desfasada 3 campanas. A diferencia de
MODELO_AZZORTI_HOG_NAC_NAC / MODELO_AZZORTI_ROP_NAC_NAC (un solo pais), este
modelo cruza dos paises (ver PAIS_DESTINO/PAIS_ORIGEN en `3.- src/config.py`).

Migrado desde los archivos sueltos "0.- MOD_0.py" / "analisis_ratio_precio.py"
/ "mis_funciones.py" (sin estructura) a este formato. Ver README.md raiz de
MODELO_AZZORTI_ROP_ADV_NAC para el estado actual del modelo (no llega a
pipeline final entrenado) y la limitacion de nivel de agregacion (referencia
SAP, no talla).

## Estructura
- **0.- data/** — raw / interim / processed.
- **1.- references/** — documentacion de negocio (tabla de competidores).
- **2.- notebooks/** — EDA numerado, resultado de correr `3.- src/models/entrenar.py`.
- **3.- src/** — codigo productivo: `data/`, `features/`, `models/`, `visualization/`.
- **4.- models/** — cortes de quintiles (fitted artifacts); todavia no hay un pipeline final entrenado.
- **5.- reports/** — excel / figures (vacio por ahora).

No existe (todavia) una carpeta `1.- MOD_RF_1` de produccion: este modelo no
tiene pipeline entrenado ni `predecir.py`, asi que no hay nada que promover
a produccion aun.
