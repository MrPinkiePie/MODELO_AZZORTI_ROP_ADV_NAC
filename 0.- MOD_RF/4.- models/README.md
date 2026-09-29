# 4.- models

`PIPELINE_ROP_ADV_NAC.pkl` — pipeline final (StandardScaler + CatBoostEncoder + RandomForestRegressor) entrenado por `3.- src/models/entrenar.py` (PASO 11), con los mejores hiperparametros de Optuna. Se sobreescribe cada vez que se corre `entrenar.py` con `guardar_pkl=True`.
