# 3.- src

Codigo reutilizable y productivo (no exploratorio):

- **config.py** — rutas y variables centralizadas (paises destino/origen, desfase de campanas, rutas internas/externas). Ver TODO sobre la ruta de CON_PRD.
- **data/** — cargar (`cargar_datos.py`) y limpiar (`limpiar_datos.py`) las tablas de destino/origen; funciones compartidas (`mis_funciones.py`).
- **features/** — construccion de variables de competencia, cruce origen/destino, agregacion a nivel referencia, quintiles (`construir_features.py`).
- **models/** — split train/test, metrica de negocio y Optuna (`entrenamiento_utils.py`), orquestacion (`entrenar.py`); `pipeline.py`/`evaluacion.py`/`predecir.py` todavia no implementados (ver TODO en `entrenar.py`).
- **visualization/** — graficos reutilizables (`graficos.py`).
