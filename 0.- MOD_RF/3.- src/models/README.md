# models

- **entrenar.py** — orquesta todo el flujo: carga, limpieza, features, Optuna, fit del pipeline final, guardado en `4.- models/`, percentiles/metricas (SVS/SBR) y export de cada corrida a `0.- data/processed/PRUEBAS DE MODELO/`.
- **entrenamiento_utils.py** — split train/test por corte de campana, metrica de negocio (servicio/sobrante, `rf_score_con_std`), armado de X/y, seleccion de variables con Boruta y el objective/estudio de Optuna (mismo enfoque simple que MODELO_AZZORTI_ROP_NAC_NAC).
- **pipeline.py** — arma el Pipeline (StandardScaler + CatBoostEncoder + RandomForestRegressor).
- **evaluacion.py** — predicciones por arbol/percentiles, importancia de variables, consolidacion de resultados.
- **predecir.py** — no implementado todavia (ver docstring): falta aplicar el `.pkl` guardado sobre campanas nuevas.
