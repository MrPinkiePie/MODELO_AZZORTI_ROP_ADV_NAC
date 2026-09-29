"""
Carga el modelo ya entrenado (guardado por entrenar.py en 4.- models/) y
genera el pronostico sobre datos nuevos. No implementado todavia: el codigo
original nunca llega a este paso -- se detiene en el mejor trial de Optuna,
sin fit() final ni guardado del pipeline (ver TODO en entrenar.py).

Deberia contener:
- Una funcion que cargue el .pkl/.joblib mas reciente desde 4.- models/.
- La preparacion de los datos nuevos: mismos pasos de carga/limpieza/features
  que en entrenar.py (incluyendo cruzar_origen_destino con el desfase de
  DESFASE_CAMPANAS campanas), para que las columnas coincidan con las que el
  Pipeline espera.
- La llamada a predict() del Pipeline cargado, y el guardado o retorno del
  pronostico resultante (a nivel referencia -- ver README.md sobre el
  pendiente de desagregar a nivel talla).
"""
